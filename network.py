#!/usr/bin/env python

import os
import json
import logging
import argparse
import torch
import numpy as np
import random

from util.corpus import Corpus
import paddle.nn as nn
import paddle
from paddlenlp.transformers import BertTokenizer
from ernie_gen import ErnieForGeneration
from engine import Trainer
from model import CRS


def model_config():
    """
    model_config
    """
    parser = argparse.ArgumentParser()

    # Data
    data_arg = parser.add_argument_group("Data")
    data_arg.add_argument("--data_dir", type=str, default="./data/")
    data_arg.add_argument("--data_prefix", type=str, default="demo.DuRecDial")
    data_arg.add_argument("--save_dir", type=str, default="./models/")

    # Network
    net_arg = parser.add_argument_group("Model")
    net_arg.add_argument("--model_name_or_path", type=str, default="macbert-base-chinese")

    # Training / Testing
    train_arg = parser.add_argument_group("Training")
    train_arg.add_argument("--lr", type=float, default=1e-4)
    train_arg.add_argument("--adam_epsilon", type=float, default=1e-8)
    train_arg.add_argument("--warmup_proportion", type=float, default=0.1)
    train_arg.add_argument("--num_epochs", type=int, default=10)

    # MISC
    misc_arg = parser.add_argument_group("Misc")
    misc_arg.add_argument("--gpu", type=int, default=0)
    misc_arg.add_argument("--log_steps", type=int, default=1000)
    misc_arg.add_argument("--valid_steps", type=int, default=6000)
    misc_arg.add_argument("--batch_size", type=int, default=2)
    misc_arg.add_argument("--accumulate_batchs_num", type=int, default=4)
    misc_arg.add_argument("--test", type=int, default=0)

    config = parser.parse_args()

    return config


def main():
    """
    main
    """
    config = model_config()
    config.use_gpu = torch.cuda.is_available() and config.gpu >= 0
    device = config.gpu
    if device >= 0:
        torch.cuda.set_device(device)

    tokenizer = BertTokenizer.from_pretrained(config.model_name_or_path, do_lower_case=True)

    # Data definition
    corpus = Corpus(data_dir=config.data_dir, data_prefix=config.data_prefix, tokenizer=tokenizer)

    train_iter = corpus.create_batches(config.batch_size, "train")
    valid_iter = corpus.create_batches(config.batch_size, "valid")
    test_iter = corpus.create_batches(config.batch_size, "test")

    gen_model = ErnieForGeneration.from_pretrained(config.model_name_or_path)
    model = CRS(gen_model)

    max_steps = len(train_iter) * config.num_epochs / config.accumulate_batchs_num

    lr_scheduler = paddle.optimizer.lr.LambdaDecay(
        config.lr,
        lambda current_step, num_warmup_steps=max_steps * config.warmup_proportion,
               num_training_steps=max_steps: float(
            current_step) / float(max(1, num_warmup_steps))
        if current_step < num_warmup_steps else max(
            0.0,
            float(num_training_steps - current_step) / float(
                max(1, num_training_steps - num_warmup_steps))))

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=config.adam_epsilon,
        parameters=model.parameters(),
        grad_clip=nn.ClipGradByGlobalNorm(1.0),
        apply_decay_param_fun=lambda x: x in decay_params)

    # Step1：定义 GradScaler，用于缩放loss比例，避免浮点数溢出
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

    # Step2：在level=’O2‘模式下，将网络参数从FP32转换为FP16
    # model, optimizer = paddle.amp.decorate(models=model, optimizers=optimizer, level='O2', master_weight=None,
    #                                        save_dtype=None)

    # Save directory
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    # Logger definition
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    if config.test == 0:
        fh = logging.FileHandler(os.path.join(config.save_dir, "train.log"))
    else:
        fh = logging.FileHandler(os.path.join(config.save_dir, "test.log"))
    logger.addHandler(fh)

    # Save config
    params_file = os.path.join(config.save_dir, "params.json")
    with open(params_file, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)
    print("Saved params to '{}'".format(params_file))
    logger.info(model)

    # Train
    logger.info("Training starts ...")
    trainer = Trainer(model=model, tokenizer=tokenizer, optimizer=optimizer, train_iter=train_iter,
                      valid_iter=valid_iter, logger=logger, num_epochs=config.num_epochs,
                      save_dir=config.save_dir, log_steps=config.log_steps,
                      valid_steps=config.valid_steps, lr_scheduler=lr_scheduler,
                      scaler=scaler, accumulate_batchs_num=config.accumulate_batchs_num)

    # start training
    if config.test == 0:
        trainer.train()
        logger.info("Training done!")
    else:
        # Test
        logger.info("")
        trainer.load(os.path.join(config.save_dir, "best_{}".format(config.test)))
        # logger.info("Testing starts ...")
        # metrics = trainer.evaluate(test_iter)
        # logger.info(metrics.report_val())
        logger.info("Generation starts ...")
        test_res = torch.load(os.path.join(config.data_dir, "test_res.txt"))
        test_gen_file = os.path.join(config.save_dir, "test.result")
        metrics = trainer.evaluate_generation(test_iter, test_res, save_file=test_gen_file)
        logger.info(metrics.report_val())


if __name__ == '__main__':
    seed = 1011
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")