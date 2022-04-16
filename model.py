import paddle
import paddle.nn as nn
import numpy as np

class PriorNet(nn.Layer):
    r""" 计算先验概率p(z|x)的网络，x为解码器最后一步的输出 """
    def __init__(self, x_size,  # post编码维度
                 latent_size,  # 潜变量维度
                 dims):  # 隐藏层维度
        super(PriorNet, self).__init__()
        assert len(dims) >= 1  # 至少两层感知机

        dims = [x_size] + dims + [latent_size*2]
        dims_input = dims[:-1]
        dims_output = dims[1:]

        self.latent_size = latent_size
        self.mlp = nn.Sequential()
        for idx, (x, y) in enumerate(zip(dims_input[:-1], dims_output[:-1])):
            self.mlp.add_sublayer(f'linear{idx}', nn.Linear(x, y))  # 线性层
            self.mlp.add_sublayer(f'activate{idx}', nn.Tanh())  # 激活层
        self.mlp.add_sublayer('output', nn.Linear(dims_input[-1], dims_output[-1]))

    def forward(self, x):  # [batch, x_size]
        predict = self.mlp(x)  # [batch, latent_size*2]
        mu, logvar = predict.split([self.latent_size]*2, 1)
        std = (0.5 * logvar).exp()
        eps = paddle.randn(std.shape)
        z = eps * std + mu
        return mu, logvar, z

class RecognizeNet(nn.Layer):
    r""" 计算后验概率p(z|x,y)的网络；x，y为解码器最后一步的输出 """
    def __init__(self, x_size,  # post编码维度
                 y_size,  # response编码维度
                 latent_size,  # 潜变量维度
                 dims):  # 隐藏层维度
        super(RecognizeNet, self).__init__()
        assert len(dims) >= 1  # 至少两层感知机

        dims = [x_size+y_size] + dims + [latent_size*2]
        dims_input = dims[:-1]
        dims_output = dims[1:]

        self.latent_size = latent_size
        self.mlp = nn.Sequential()
        for idx, (x, y) in enumerate(zip(dims_input[:-1], dims_output[:-1])):
            self.mlp.add_sublayer(f'linear{idx}', nn.Linear(x, y))  # 线性层
            self.mlp.add_sublayer(f'activate{idx}', nn.Tanh())  # 激活层
        self.mlp.add_sublayer('output', nn.Linear(dims_input[-1], dims_output[-1]))

    def forward(self, x,  # [batch, x_size]
                y):  # [batch, y_size]
        x = paddle.concat([x, y], 1)  # [batch, x_size+y_size]
        predict = self.mlp(x)  # [batch, latent_size*2]
        mu, logvar = predict.split([self.latent_size]*2, 1)
        std = (0.5 * logvar).exp()
        eps = paddle.randn(std.shape)
        z = eps * std + mu
        return mu, logvar, z

class CRS(nn.Layer):
    def __init__(self, gen_model):
        super().__init__()
        self.Ernie_gen = gen_model
        self.hidden_size = self.Ernie_gen.config["self"]["hidden_size"]
        self.z_size = 64
        self.prior_net = PriorNet(self.hidden_size, self.z_size, [self.hidden_size // 2])
        self.recognize_net = RecognizeNet(self.hidden_size, self.hidden_size, self.z_size, [self.hidden_size // 2])
        self.bagofword_output_layer = nn.Sequential(
            nn.Linear(self.z_size + self.hidden_size, self.Ernie_gen.config["self"]["vocab_size"]),
            nn.LogSoftmax(-1))
        self.selectcue = nn.Linear(self.z_size + self.hidden_size, self.hidden_size)
        self.prepareState = nn.LayerList(
            [nn.Linear(self.z_size + self.hidden_size, self.hidden_size) for _ in range(12)])

    def gaussian_kld(self, recog_mu, recog_logvar, prior_mu, prior_logvar):
        kld = 0.5 * paddle.sum(prior_logvar - recog_logvar - 1
                              + recog_logvar.exp() / prior_logvar.exp()
                              + (prior_mu - recog_mu).pow(2) / prior_logvar.exp(), 1)
        return kld

    def forward(self, inputs, attn_id, test=False):

        (src_ids, tgt_ids, tgt_tids, attn_ids, mask_tgt_2_srctgt,
         mask_attn_2_srctgtattn, tgt_labels, raw_tgt_labels,
         cue_ids, goal_ids) = inputs
        tgt_pos = paddle.nonzero(attn_ids == attn_id)
        if len(tgt_labels.shape) == 1:
            tgt_labels = paddle.reshape(tgt_labels, [-1, 1])

        hidden_cue = []
        caches_cue_k, caches_cue_v = [], []
        for cue_ids_ in cue_ids:
            _, __, info = self.Ernie_gen(cue_ids_, encode_only=True)
            tmp_h = info['hiddens'][-1][:, 0, :]
            hidden_cue.append(paddle.reshape(tmp_h, [src_ids.shape[0], -1] + tmp_h.shape[1:]))

            tmp_k, tmp_v = info['caches'][0][-1], info['caches'][1][-1]
            caches_cue_k.append(paddle.reshape(tmp_k, [src_ids.shape[0], -1] + tmp_k.shape[1:]))
            caches_cue_v.append(paddle.reshape(tmp_v, [src_ids.shape[0], -1] + tmp_v.shape[1:]))

        hidden_cue = paddle.concat(hidden_cue, 1)

        src_ids = paddle.concat([goal_ids, src_ids], 1)
        _, __, info = self.Ernie_gen(src_ids, encode_only=True)
        last_hidden = info['hiddens'][-1][:, 0, :]

        prior_mu, prior_logvar, prior_z = self.prior_net(last_hidden)
        if test == False:
            _, __, info = self.Ernie_gen(tgt_ids, tgt_tids, encode_only=True)
            last_hidden_tgt = info['hiddens'][-1][:, 0, :]
            recog_mu, recog_logvar, post_z = self.recognize_net(last_hidden, last_hidden_tgt)
            kl_loss = self.gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar).mean()


            z = paddle.concat([post_z, last_hidden], 1)
            bagOfWord_logit = self.bagofword_output_layer(z)
            bagOfWord_logit = bagOfWord_logit.unsqueeze(1).tile([1, tgt_ids.shape[-1], 1])
            bagOfWord_loss = nn.functional.nll_loss(bagOfWord_logit.transpose([0, 2, 1]), raw_tgt_labels,
                                                    ignore_index=0)
        else:
            z = paddle.concat([prior_z, last_hidden], 1)
            kl_loss = bagOfWord_loss = 0

        scores = paddle.fluid.layers.matmul(hidden_cue, self.selectcue(z).unsqueeze(1),
                                            transpose_y=True) / np.sqrt(self.hidden_size)
        scores = scores.squeeze()
        attn = nn.functional.sigmoid(scores)
        bow_loss = 0

        attn = attn.unsqueeze(-1).unsqueeze(-1)
        new_shape = [caches_cue_k[0].shape[0], -1, caches_cue_k[0].shape[-1]]
        cnt = caches_cue_k[0].shape[1]
        caches_cue_k[0] = paddle.reshape(caches_cue_k[0] * attn[:, :cnt, :, :], new_shape)
        caches_cue_k[1] = paddle.reshape(caches_cue_k[1] * attn[:, cnt:, :, :], new_shape)
        caches_cue_v[0] = paddle.reshape(caches_cue_v[0] * attn[:, :cnt, :, :], new_shape)
        caches_cue_v[1] = paddle.reshape(caches_cue_v[1] * attn[:, cnt:, :, :], new_shape)

        caches_cue_k = paddle.concat(caches_cue_k, 1)
        caches_cue_v = paddle.concat(caches_cue_v, 1)

        decoder_z = [L(z).unsqueeze(1) for L in self.prepareState]

        decoder_z_k = decoder_z[: -1] + [paddle.concat([caches_cue_k, decoder_z[-1]], 1)]
        decoder_z_v = decoder_z[: -1] + [paddle.concat([caches_cue_v, decoder_z[-1]], 1)]

        _, __, info = self.Ernie_gen(
            tgt_ids,
            sent_ids=tgt_tids,
            attn_bias=mask_tgt_2_srctgt,
            past_cache=(decoder_z_k, decoder_z_v),
            encode_only=True)
        cached_k, cached_v = info['caches']

        past_cache_k = [
            paddle.concat([z_, k_], 1) for z_, k_ in zip(decoder_z_k, cached_k)
        ]
        past_cache_v = [
            paddle.concat([z_, v_], 1) for z_, v_ in zip(decoder_z_v, cached_v)
        ]

        _, __, info = self.Ernie_gen(
            attn_ids,
            sent_ids=tgt_tids,
            attn_bias=mask_attn_2_srctgtattn,
            past_cache=(past_cache_k, past_cache_v),
            encode_only=True)
        encoded = info["hiddens"][-1]
        encoded = (encoded).gather_nd(tgt_pos)
        encoded = self.Ernie_gen.act(self.Ernie_gen.mlm(encoded))
        decoder_outputs = self.Ernie_gen.mlm_ln(encoded)

        logits = decoder_outputs.matmul(
            self.Ernie_gen.word_emb.weight, transpose_y=True) + self.Ernie_gen.mlm_bias

        loss = nn.functional.cross_entropy(
            logits,
            tgt_labels,
            reduction="none",
            soft_label=(tgt_labels.shape[-1] != 1))
        nll_loss = loss.mean()

        return (nll_loss, kl_loss, bagOfWord_loss, bow_loss), tgt_labels.shape[0]
