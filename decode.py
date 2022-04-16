#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import re
from collections import namedtuple

import paddle
import paddle.nn as nn
import numpy as np


def gen_bias(encoder_inputs, decoder_inputs, step):
    decoder_bsz, decoder_seqlen = decoder_inputs.shape[:2]
    encoder_bsz, encoder_seqlen = encoder_inputs.shape[:2]
    attn_bias = paddle.reshape(
        paddle.arange(
            0, decoder_seqlen, 1, dtype='float32') + 1, [1, -1, 1])
    decoder_bias = paddle.cast(
        (paddle.matmul(
            attn_bias, 1. / attn_bias, transpose_y=True) >= 1.),
        'float32')  #[1, decoderlen, decoderlen]
    encoder_bias = paddle.unsqueeze(
        paddle.cast(encoder_inputs, 'float32'),
        [1])  #[bsz, 1, encoderlen]
    encoder_bias = paddle.expand(
        encoder_bias, [encoder_bsz, decoder_seqlen,
                       encoder_seqlen])  #[bsz,decoderlen, encoderlen]
    decoder_bias = paddle.expand(
        decoder_bias, [decoder_bsz, decoder_seqlen,
                       decoder_seqlen])  #[bsz, decoderlen, decoderlen]
    if step > 0:
        bias = paddle.concat([
            encoder_bias, paddle.ones([decoder_bsz, decoder_seqlen, step],
                                      'float32'), decoder_bias
        ], -1)
    else:
        bias = paddle.concat([encoder_bias, decoder_bias], -1)
    return bias


BeamSearchState = namedtuple('BeamSearchState',
                             ['log_probs', 'lengths', 'finished'])
BeamSearchOutput = namedtuple('BeamSearchOutput',
                              ['scores', 'predicted_ids', 'beam_parent_ids'])


def log_softmax(x):
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum())


def mask_prob(p, onehot_eos, finished):
    is_finished = paddle.cast(paddle.reshape(finished, [-1, 1]) != 0, 'float32')
    p = is_finished * (1. - paddle.cast(onehot_eos, 'float32')) * -9999. + (
        1. - is_finished) * p
    return p


def hyp_score(log_probs, length, length_penalty):
    lp = paddle.pow((5. + paddle.cast(length, 'float32')) / 6., length_penalty)
    return log_probs / lp

def top_k_top_p_filtering(logits, top_k=0, top_p=0.9, filter_value=-1e9, min_tokens_to_keep=2):
    if top_k > 0:
        logits[logits < paddle.topk(logits, k=top_k)[0][..., -1, None]] = filter_value

    if top_p < 1.0:
        sorted_probs, sorted_indices = paddle.topk(logits, k=logits.shape[-1])
        cumulative_probs = paddle.cumsum(nn.functional.softmax(sorted_probs, -1), axis=-1)
        sorted_indices_to_remove = (cumulative_probs > top_p).astype(np.float32)
        sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        for b in range(sorted_probs.shape[0]):
            sorted_indices_to_remove[b] = paddle.scatter(sorted_indices_to_remove[b], sorted_indices[b],
                                                         sorted_indices_to_remove[b])
        logits[sorted_indices_to_remove.astype(np.bool)] = filter_value
    return logits

def beam_search_step(state, logits, eos_id, beam_width, step, token_ids,
                     length_penalty):
    """logits.shape == [B*W, V]"""
    _, vocab_size = logits.shape

    bsz, beam_width = state.log_probs.shape


    onehot_eos = paddle.cast(
        nn.functional.one_hot(paddle.ones([1], 'int64') * eos_id, vocab_size),
        'int64')  #[1, V]

    probs = nn.functional.log_softmax(logits)  #[B*W, V]

    # probs = enforce_repetition_penalty_(probs, token_ids)
    # banned_batch_tokens = calc_banned_tokens(token_ids, 8, step)
    # for i, banned_tokens in enumerate(banned_batch_tokens):
    #     probs[i, banned_tokens] = -1e9

    probs = mask_prob(probs, onehot_eos, state.finished)  #[B*W, V]
    allprobs = paddle.reshape(state.log_probs, [-1, 1]) + probs  #[B*W, V]

    # sorted_probs, sorted_indices = paddle.topk(allprobs, k=vocab_size)
    # sibling_score = paddle.arange(0, vocab_size).unsqueeze(0) * 2.0
    # sorted_probs -= sibling_score
    # # import pdb
    # # pdb.set_trace()
    # allprobs = paddle.stack([
    #     paddle.index_select(sorted_probs[i], paddle.argsort(sorted_indices[i]))
    #     for i in range(sorted_probs.shape[0])
    # ])

    not_finished = 1 - paddle.reshape(state.finished, [-1, 1])  #[B*W,1]
    not_eos = 1 - onehot_eos
    length_to_add = not_finished * not_eos  #[B*W,V]
    alllen = paddle.reshape(state.lengths, [-1, 1]) + length_to_add

    allprobs = paddle.reshape(allprobs, [-1, beam_width * vocab_size])
    alllen = paddle.reshape(alllen, [-1, beam_width * vocab_size])
    allscore = hyp_score(allprobs, alllen, length_penalty)
    # allscore = top_k_top_p_filtering(allscore/0.1, min_tokens_to_keep=beam_width*2)    ## nucleus_sampling
    if step == 0:
        allscore = paddle.reshape(
            allscore,
            [bsz, beam_width, -1])[:, 0, :]  # first step only consiter beam 0

    # scores = nn.functional.softmax(allscore, -1)
    # scores[allscore >= paddle.topk(allscore, k=beam_width*2)[0][..., -1, None]] += 1e-9
    # idx = paddle.multinomial(scores, num_samples=beam_width*2)
    # scores = [allscore[b][idx[b]] for b in range(bsz)]
    # scores, new_idx = paddle.topk(paddle.stack(scores), k=beam_width)
    # idx = paddle.stack([idx[b][new_idx[b]] for b in range(bsz)])

    # allscore = paddle.reshape(allscore, [-1, vocab_size])
    # next_scores, next_tokens = paddle.topk(
    #     allscore, 2 * beam_width, axis=1)
    #
    # sibling_score = paddle.arange(
    #     1, 2 * beam_width + 1).unsqueeze(0) * 1.0
    #
    # diversed_score = next_scores - sibling_score
    #
    # next_scores = next_scores.reshape([bsz, -1])
    # next_tokens = next_tokens.reshape([bsz, -1])
    #
    # diversed_score = diversed_score.reshape([bsz, -1])
    # diversed_score, diversed_tokens = paddle.topk(diversed_score, beam_width, axis=1)
    # scores = paddle.stack([
    #     paddle.index_select(next_scores[i], diversed_tokens[i])
    #     for i in range(next_scores.shape[0])
    # ])
    # idx = paddle.stack([
    #     paddle.index_select(next_tokens[i], diversed_tokens[i]) + (diversed_tokens[i] // (2 * beam_width)) * vocab_size
    #     for i in range(next_tokens.shape[0])
    # ])

    scores, idx = paddle.topk(allscore, k=beam_width)  #[B, W]
    next_beam_id = idx // vocab_size  #[B, W]
    next_word_id = idx % vocab_size

    token_ids = paddle.concat(
                [
                    paddle.index_select(token_ids,
                                        paddle.reshape(next_beam_id + paddle.arange(0, bsz * beam_width, beam_width).unsqueeze(1), [-1])),
                    paddle.reshape(next_word_id, [-1, 1])
                ],
                axis=-1)

    # import pdb
    # pdb.set_trace()
    gather_idx = paddle.concat(
        [paddle.nonzero(idx != -1)[:, :1], paddle.reshape(idx, [-1, 1])], 1)
    next_probs = paddle.reshape(
        paddle.gather_nd(allprobs, gather_idx), idx.shape)
    next_len = paddle.reshape(paddle.gather_nd(alllen, gather_idx), idx.shape)

    gather_idx = paddle.concat([
        paddle.nonzero(next_beam_id != -1)[:, :1], paddle.reshape(next_beam_id,
                                                                  [-1, 1])
    ], 1)
    next_finished = paddle.reshape(
        paddle.gather_nd(state.finished, gather_idx),
        state.finished.shape)  #[gather new beam state according to new beam id]

    next_finished += paddle.cast(next_word_id == eos_id, 'int64')
    next_finished = paddle.cast(next_finished > 0, 'int64')

    next_state = BeamSearchState(
        log_probs=next_probs, lengths=next_len, finished=next_finished)
    output = BeamSearchOutput(
        scores=scores, predicted_ids=next_word_id, beam_parent_ids=next_beam_id)

    return output, next_state, token_ids

def enforce_repetition_penalty_(lprobs, prev_output_tokens, repetition_penalty=2):
    """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
    score = paddle.index_sample(lprobs, prev_output_tokens)
    score = paddle.where(score < 0, score * repetition_penalty,
                         score / repetition_penalty)
    input_ids = prev_output_tokens + paddle.arange(lprobs.shape[0]).unsqueeze(
        -1) * lprobs.shape[-1]
    outputs = paddle.scatter(lprobs.flatten(),
                             input_ids.flatten(),
                             score.flatten()).reshape(lprobs.shape)
    return outputs

def calc_banned_tokens(prev_input_ids, no_repeat_ngram_size, cur_len):
    # Copied from fairseq for no_repeat_ngram in beam_search"""
    num_hypos = len(prev_input_ids)
    if cur_len < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


@paddle.no_grad()
def beam_search_infilling(model,
                          tokenizer,
                          inputs,
                          max_decode_len=120,
                          beam_width=5,
                          tgt_type_id=1,
                          length_penalty=1.0):

    vocab = tokenizer.vocab
    eos_id = vocab[tokenizer.sep_token]
    sos_id = vocab[tokenizer.cls_token]
    pad_id = vocab[tokenizer.pad_token]
    unk_id = vocab[tokenizer.unk_token]
    vocab_size = len(vocab)
    attn_id = tokenizer.vocab['[ATTN]'] if '[ATTN]' in tokenizer.vocab else tokenizer.vocab['[MASK]']

    (token_ids, tgt_ids, tgt_tids, attn_ids, mask_src_2_src, mask_tgt_2_srctgt,
     mask_attn_2_srctgtattn, mask_cue_2_attn, tgt_labels, raw_tgt_labels,
     cue_ids, goal_ids, golden) = inputs

    hidden_cue = []
    caches_cue_k, caches_cue_v = [], []
    for cue_ids_ in cue_ids:
        _, __, info = model.Ernie_gen(cue_ids_, encode_only=True)
        tmp_h = info['hiddens'][-1][:, 0, :]
        hidden_cue.append(paddle.reshape(tmp_h, [token_ids.shape[0], -1] + tmp_h.shape[1:]))
        tmp_k, tmp_v = info['caches'][0][-1], info['caches'][1][-1]
        caches_cue_k.append(paddle.reshape(tmp_k, [token_ids.shape[0], -1] + tmp_k.shape[1:]))
        caches_cue_v.append(paddle.reshape(tmp_v, [token_ids.shape[0], -1] + tmp_v.shape[1:]))


    hidden_cue = paddle.concat(hidden_cue, 1)

    _, __, info = model.Ernie_gen(paddle.concat([goal_ids, token_ids], 1), encode_only=True)
    last_hidden = info['hiddens'][-1][:, 0, :]
    prior_mu, prior_logvar, z = model.prior_net(last_hidden)

    z = paddle.concat([z, last_hidden], 1)
    scores = paddle.fluid.layers.matmul(hidden_cue, model.selectcue(z).unsqueeze(1),
                                        transpose_y=True) / np.sqrt(model.hidden_size)
    scores = scores.squeeze()
    attn = nn.functional.sigmoid(scores)
    attn = attn.unsqueeze(-1).unsqueeze(-1)
    new_shape = [caches_cue_k[0].shape[0], -1, caches_cue_k[0].shape[-1]]
    cnt = caches_cue_k[0].shape[1]
    caches_cue_k[0] = paddle.reshape(caches_cue_k[0] * attn[:, :cnt, ...], new_shape)
    caches_cue_k[1] = paddle.reshape(caches_cue_k[1] * attn[:, cnt:, ...], new_shape)
    caches_cue_v[0] = paddle.reshape(caches_cue_v[0] * attn[:, :cnt, ...], new_shape)
    caches_cue_v[1] = paddle.reshape(caches_cue_v[1] * attn[:, cnt:, ...], new_shape)

    caches_cue_k = paddle.concat(caches_cue_k, 1)
    caches_cue_v = paddle.concat(caches_cue_v, 1)

    decoder_z = [L(z).unsqueeze(1) for L in model.prepareState]

    cached_k = decoder_z[: -1] + [paddle.concat([caches_cue_k, decoder_z[-1]], 1)]

    cached_v = decoder_z[: -1] + [paddle.concat([caches_cue_v, decoder_z[-1]], 1)]

    d_batch, d_seqlen = token_ids.shape

    state = BeamSearchState(
        log_probs=paddle.zeros([d_batch, beam_width], 'float32'),
        lengths=paddle.zeros([d_batch, beam_width], 'int64'),
        finished=paddle.zeros([d_batch, beam_width], 'int64'))
    outputs = []

    def reorder_(t, parent_id):
        """reorder cache according to parent beam id"""
        gather_idx = paddle.nonzero(
            parent_id != -1)[:, 0] * beam_width + paddle.reshape(parent_id,
                                                                 [-1])
        t = paddle.gather(t, gather_idx)
        return t

    def tile_(t, times):
        _shapes = list(t.shape[1:])
        new_shape = [t.shape[0], times] + list(t.shape[1:])
        ret = paddle.reshape(
            paddle.expand(paddle.unsqueeze(t, [1]), new_shape),
            [-1, ] + _shapes)
        return ret

    cached_k = [tile_(k, beam_width) for k in cached_k]
    cached_v = [tile_(v, beam_width) for v in cached_v]
    past_cache = (cached_k, cached_v)

    z_bias = paddle.concat([paddle.reshape(cue_ids[0], [d_batch, -1]),
                            paddle.reshape(cue_ids[1], [d_batch, -1]),
                            token_ids[:, :1]], 1)
    token_ids = tile_(token_ids, beam_width)
    z_bias = tile_(z_bias, beam_width)

    cls_ids = paddle.ones([d_batch * beam_width], dtype='int64') * sos_id
    attn_ids = paddle.ones(
        [d_batch * beam_width], dtype='int64') * attn_id  # SOS
    ids = paddle.stack([cls_ids, attn_ids], -1)
    for step in range(max_decode_len):
        bias = gen_bias(z_bias != 0, ids, step)
        pos_ids = paddle.to_tensor(
            np.tile(
                np.array(
                    [[step, step + 1]], dtype=np.int64),
                [d_batch * beam_width, 1]))
        _, __, info = model.Ernie_gen(
            ids,
            sent_ids=paddle.ones_like(ids) * tgt_type_id,
            pos_ids=pos_ids,
            attn_bias=bias,
            past_cache=past_cache,
            encode_only=True)
        encoded = info["hiddens"][-1]
        encoded = model.Ernie_gen.act(model.Ernie_gen.mlm(encoded))
        decoder_outputs = model.Ernie_gen.mlm_ln(encoded)
        logits = decoder_outputs.matmul(
            model.Ernie_gen.word_emb.weight, transpose_y=True) + model.Ernie_gen.mlm_bias

        # if step < 15:
        #     bias = gen_bias(token_ids[:, :1] == 0, ids, step)
        #     _, __, info = model.Ernie_gen(
        #         ids,
        #         sent_ids=paddle.ones_like(ids) * tgt_type_id,
        #         pos_ids=pos_ids,
        #         attn_bias=bias,
        #         past_cache=past_cache,
        #         encode_only=True)
        #     encoded = model.Ernie_gen.act(model.Ernie_gen.mlm(info["hiddens"][-1]))
        #     decoder_outputs = model.Ernie_gen.mlm_ln(encoded)
        #     _logits = decoder_outputs.matmul(
        #         model.Ernie_gen.word_emb.weight, transpose_y=True) + model.Ernie_gen.mlm_bias
        #     logits -= 0.5 * _logits

        if logits.shape[-1] > vocab_size:
            logits[:, :, vocab_size:] = -1e9
        logits[:, :, pad_id] = -1e9
        logits[:, :, unk_id] = -1e9
        logits[:, :, attn_id] = -1e9

        output, state, token_ids = beam_search_step(
            state,
            logits[:, 1],
            eos_id=eos_id,
            beam_width=beam_width,
            step=step,
            token_ids=token_ids,
            length_penalty=length_penalty)
        outputs.append(output)

        past_cached_k, past_cached_v = past_cache
        cached_k, cached_v = info['caches']
        cached_k = [
            reorder_(
                paddle.concat([pk, k[:, :1, :]], 1), output.beam_parent_ids)
            for pk, k in zip(past_cached_k, cached_k)
        ]  # concat cached
        cached_v = [
            reorder_(
                paddle.concat([pv, v[:, :1, :]], 1), output.beam_parent_ids)
            for pv, v in zip(past_cached_v, cached_v)
        ]
        past_cache = (cached_k, cached_v)

        pred_ids_flatten = paddle.reshape(output.predicted_ids,
                                          [d_batch * beam_width])
        ids = paddle.stack([pred_ids_flatten, attn_ids], 1)

        if state.finished.numpy().all():
            break

    final_ids = paddle.stack([o.predicted_ids for o in outputs], 0)
    final_parent_ids = paddle.stack([o.beam_parent_ids for o in outputs], 0)
    final_ids = nn.functional.gather_tree(
        final_ids, final_parent_ids)[:, :, 0]  # pick best beam
    final_ids = paddle.transpose(
        paddle.reshape(final_ids, [-1, d_batch * 1]), [1, 0])

    return final_ids.numpy()

en_patten = re.compile(r'^[a-zA-Z0-9]*$')


def post_process(tokens):
    ret = []
    for token in tokens:
        if token.startswith('##'):
            ret.append(token[2:])
        elif token in ['[CLS]', '[PAD]', '[MASK]', '[UNK]']:
            continue
        elif token == "[SEP]":
            break
        else:
            ret.append(token)
    return ret