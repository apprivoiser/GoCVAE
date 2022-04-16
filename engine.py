#!/usr/bin/env python

import os
import numpy as np
import torch
import paddle
from collections import defaultdict
from evaluation.eval import calc_bleu, calc_distinct, calc_f1
from decode import beam_search_infilling, post_process

class MetricsManager(object):

    def __init__(self):
        self.metrics_val = defaultdict(float)
        self.num_samples = 0
        self.lens = 0

    def update(self, metrics):

        num_samples = metrics.pop("num_samples", 1)
        lens = metrics.pop("lens", 0)
        self.num_samples += num_samples
        self.lens += lens

        for key, val in metrics.items():
            if val is not None:
                self.metrics_val[key] += val

    def get(self, name):
        val = self.metrics_val.get(name)
        return val / self.num_samples

    def report_val(self):
        metric_strs = []
        for key, val in self.metrics_val.items():
            if(key == "ppl"):
                metric_str = "{}-{:.3f}".format(key.upper(), np.exp(val / self.lens))
            else:
                metric_str = "{}-{:.3f}".format(key.upper(), val / self.num_samples)
            metric_strs.append(metric_str)
        metric_strs = "   ".join(metric_strs)
        return metric_strs


class Trainer(object):
    """
    Trainer
    """
    def __init__(self,
                 model,
                 tokenizer,
                 optimizer,
                 train_iter,
                 valid_iter,
                 logger,
                 num_epochs=1,
                 save_dir=None,
                 log_steps=None,
                 valid_steps=None,
                 lr_scheduler=None,
                 scaler=None,
                 accumulate_batchs_num=None):
        self.model = model
        self.tokenizer = tokenizer
        self.attn_id = self.tokenizer.vocab['[ATTN]'] if '[ATTN]' in self.tokenizer.vocab else self.tokenizer.vocab[
            '[MASK]']
        self.optimizer = optimizer
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.logger = logger
        self.is_decreased_valid_metric = True
        self.valid_metric_name = "nll_loss"
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.log_steps = log_steps
        self.valid_steps = valid_steps
        self.lr_scheduler = lr_scheduler
        self.scaler = scaler
        self.accumulate_batchs_num = accumulate_batchs_num

        self.best_valid_metric = float("inf") if self.is_decreased_valid_metric else -float("inf")
        self.epoch = 0
        self.batch_num = 0

        self.train_start_message = "\n".join(["*************",
                                              " Training: ",
                                              "*************"])
        self.valid_start_message = "\n".join(["*************",
                                              " Evaluating: ",
                                              "*************"])

    def summarize_train_metrics(self, metrics, global_step):
        """
        summarize_train_metrics
        """
        for key, val in metrics.items():
            if isinstance(val, (list, tuple)):
                val = val[0]
            if isinstance(val, torch.Tensor):
                self.train_writer.add_scalar(key, val, global_step)

    def summarize_valid_metrics(self, metrics_mm, global_step):
        """
        summarize_valid_metrics
        """
        for key in metrics_mm.metrics_cum.keys():
            val = metrics_mm.get(key)
            self.valid_writer.add_scalar(key, val, global_step)

    def train_epoch(self):

        self.epoch += 1
        train_mm = MetricsManager()
        num_batches = len(self.train_iter)
        self.logger.info(self.train_start_message)

        for batch_id, inputs in enumerate(self.train_iter, 1):
            # if batch_id == 2 : break
            self.model.train()
            # 创建AMP上下文环境，开启自动混合精度训练
            # with paddle.amp.auto_cast():
            # with paddle.amp.auto_cast(enable=True, custom_white_list=None, custom_black_list=None, level='O2'):
            (nll_loss, kl_loss, bagOfWord_loss, bow_loss), lens = self.model(inputs, self.attn_id)
            if self.epoch % 2:
                kl_weight = batch_id / len(self.train_iter)
            else:
                kl_weight = 1
            # kl_weight = min(
            #     1.0 * ((batch_id + (self.epoch - 1) * len(self.train_iter)) % self.kl_step) / self.kl_step, 1)

            # 使用 GradScaler 完成 loss 的缩放，用缩放后的 loss 进行反向传播
            # scaled = self.scaler.scale(nll_loss + kl_weight * kl_loss + bagOfWord_loss + bow_loss)
            # scaled.backward()
            (nll_loss + kl_weight * kl_loss + bagOfWord_loss + bow_loss).backward()
            # (nll_loss + kl_loss + bagOfWord_loss + bow_loss).backward()
            nll_loss.backward()

            # 当累计的 batch 为 accumulate_batchs_num 时，更新模型参数
            if batch_id % self.accumulate_batchs_num == 0:
                # 训练模型
                # self.scaler.minimize(self.optimizer, scaled)
                self.optimizer.step()
                self.optimizer.clear_grad()
                self.lr_scheduler.step()

            train_mm.update(
                # {'nll_loss': nll_loss.item(), 'ppl': nll_loss.item() * lens, 'lens': lens})
                {'nll_loss': nll_loss.item(), 'kl_loss': kl_loss.item(), 'bagOfWord_loss': bagOfWord_loss.item(),
                 'ppl': nll_loss.item() * lens, 'lens': lens})

            self.batch_num += 1

            if batch_id % self.log_steps == 0:
                message_prefix = "CurEpoch: {}    Batch: {}/{}".format(self.epoch, batch_id, num_batches)
                metrics_message = train_mm.report_val()
                self.logger.info("   ".join(
                    [message_prefix, metrics_message]))

            if batch_id % self.valid_steps == 0:
                self.logger.info(self.valid_start_message)
                valid_mm = self.evaluate(self.valid_iter)
                message_prefix = "CurEpoch: {}    Batch: {}/{}".format(self.epoch, batch_id, num_batches)
                metrics_message = valid_mm.report_val()
                self.logger.info("   ".join([message_prefix, metrics_message]))
                cur_valid_metric = valid_mm.get(self.valid_metric_name)
                if self.is_decreased_valid_metric:
                    is_best = cur_valid_metric < self.best_valid_metric
                else:
                    is_best = cur_valid_metric > self.best_valid_metric

                if is_best:
                    self.best_valid_metric = cur_valid_metric
                self.save(is_best)
                self.logger.info("*" * 13 + "\n")

    @paddle.no_grad()
    def evaluate(self, data_iter):
        self.model.eval()
        mm = MetricsManager()
        for inputs in data_iter:
            (nll_loss, _, _, bow_loss), lens = self.model(inputs, self.attn_id, test=True)
            mm.update(
                {'nll_loss': nll_loss.item(), 'ppl': nll_loss.item() * lens, 'lens': lens})
        return mm

    def train(self):
        for _ in range(self.epoch, self.num_epochs):
            self.train_epoch()

    def save(self, is_best=False):
        """
        save
        """
        if is_best:
            best_model_file = os.path.join(self.save_dir, "best_{}.model".format(self.epoch))
            best_train_file = os.path.join(self.save_dir, "best_{}.train".format(self.epoch))
            paddle.save(self.model.state_dict(), best_model_file)

            train_state = {"epoch": self.epoch,
                           "batch_num": self.batch_num,
                           "best_valid_metric": self.best_valid_metric,
                           "optimizer": self.optimizer.state_dict()}
            if self.lr_scheduler is not None:
                train_state["lr_scheduler"] = self.lr_scheduler.state_dict()
            paddle.save(train_state, best_train_file)

            self.logger.info(
                "Saved best model state to '{}' with new best valid metric {}-{:.3f}".format(
                    best_model_file, self.valid_metric_name.upper(), self.best_valid_metric))

    def load(self, file_prefix):
        """
        load
        """

        model_file = "{}.model".format(file_prefix)
        train_file = "{}.train".format(file_prefix)

        model_state_dict = paddle.load(model_file)
        self.model.set_state_dict(model_state_dict)
        self.logger.info("Loaded model state from '{}'".format(model_file))

        train_state_dict = paddle.load(train_file)
        self.epoch = train_state_dict["epoch"]
        self.best_valid_metric = train_state_dict["best_valid_metric"]
        self.batch_num = train_state_dict["batch_num"]
        self.optimizer.set_state_dict(train_state_dict["optimizer"])
        if self.lr_scheduler is not None and "lr_scheduler" in train_state_dict:
            self.lr_scheduler.set_state_dict(train_state_dict["lr_scheduler"])
        self.logger.info(
            "Loaded train state from '{}' with (epoch-{} best_valid_metric-{:.3f})".format(
                train_file, self.epoch, self.best_valid_metric))


    def write_results(self, refs, hyps, results_file):
        with open(results_file, "w", encoding="utf-8") as f:
            for refs_, hyps_ in zip(refs, hyps):
                f.write("{}\n".format("".join(refs_)))
                f.write("{}\n".format("".join(hyps_)))
                f.write("\n")

    @paddle.no_grad()
    def evaluate_generation(self, data_iter, refs_words, save_file=None):
        self.model.eval()
        mm = MetricsManager()
        sents = []
        file = open('./data/word_dict.txt', 'r')
        dict = eval(file.read())
        with open('./data/word.txt', "w", encoding="utf-8") as f:
            for k, v in dict.items():
                f.write("{}\n".format(k))
        import jieba
        jieba.load_userdict("./data/word.txt")
        for id, inputs in enumerate(data_iter):
            # Use greedy_search_infilling or beam_search_infilling to get predictions
            output_ids = beam_search_infilling(
                self.model,
                self.tokenizer,
                inputs)
            refs_batch = refs_words[id * output_ids.shape[0]: (id + 1) * output_ids.shape[0]]
            for refs_, hyps_ids in zip(refs_batch, output_ids):
                hyps_ = "".join(post_process(self.tokenizer.vocab.to_tokens(hyps_ids.tolist())))
                sents.append((list(jieba.cut(hyps_)), list(jieba.cut("".join(refs_.split()).lower()))))


            # if len(sents) > 100: break

        bleu_1, bleu_2 = calc_bleu(sents)
        dist_1, dist_2 = calc_distinct(sents)

        f1 = calc_f1(sents)

        mm.update({"Bleu1": bleu_1, "Bleu2": bleu_2, "dist1": dist_1, "dist2": dist_2, "f1": f1})
        refs, hyps = [], []
        for sent in sents:
            hyps.append(''.join(sent[0]))
            refs.append(''.join(sent[1]))
        if save_file is not None:
            self.write_results(refs, hyps, save_file)
            print("Saved generation results to '{}'".format(save_file))
        return mm