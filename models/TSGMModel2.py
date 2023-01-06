# This file contains Transformer network
# Most of the code is copied from http://nlp.seas.harvard.edu/2018/04/03/attention.html

# The cfg name correspondance:
# N=num_layers
# d_model=input_encoding_size
# d_ff=rnn_size
# h is always 8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.utils as utils

import copy
import math
import numpy as np

from .TSGMAttModel import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper, AttModel

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, opt):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.opt = opt
        self.concat_type = getattr(opt, 'concat_type', 0)
        self.controller = getattr(opt, 'controller', 0)
        
    def forward(self, src, tgt, src_mask, src_tgt_mask, tgt_mask, rela_len, attr_len, att_len):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_tgt_mask,tgt, tgt_mask, rela_len, attr_len, att_len)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_tgt_mask, tgt, tgt_mask, rela_len, attr_len, att_len):
        memory_att, memory_attr, memory_rela, mask_att, mask_attr, mask_rela = divide_memory(memory, rela_len, attr_len, att_len)
        out = self.decoder(self.tgt_embed(tgt), memory_att, memory_attr, memory_rela, mask_att, mask_attr, mask_rela, tgt_mask)
        return out

def divide_memory(memory, rela_len, attr_len, att_len):
    seq_per_img = memory.shape[0] // len(rela_len)
    max_att_len = max([_ for _ in att_len])
    max_attr_len = max([_ for _ in attr_len])
    max_rela_len = max([_ for _ in rela_len])
    mask_att = torch.zeros([memory.shape[0],max_att_len], requires_grad=False).cuda()
    mask_attr = torch.zeros([memory.shape[0],max_attr_len], requires_grad=False).cuda()
    mask_rela = torch.zeros([memory.shape[0],max_rela_len], requires_grad=False).cuda()

    memory_att = torch.zeros([memory.shape[0], max_att_len, memory.shape[2]]).cuda()
    memory_attr = torch.zeros([memory.shape[0], max_attr_len, memory.shape[2]]).cuda()
    memory_rela = torch.zeros([memory.shape[0], max_rela_len, memory.shape[2]]).cuda()

    for i in range(len(rela_len)):
        mask_att[i*seq_per_img:(i+1)*seq_per_img,:att_len[i]] = 1
        mask_attr[i*seq_per_img:(i+1)*seq_per_img,:attr_len[i]] = 1
        mask_rela[i*seq_per_img:(i+1)*seq_per_img,:rela_len[i]] = 1

        memory_att[i*seq_per_img:(i+1)*seq_per_img,:att_len[i], :] = memory[i*seq_per_img:(i+1)*seq_per_img,:att_len[i],:]
        memory_attr[i*seq_per_img:(i+1)*seq_per_img,:attr_len[i], :] = memory[i*seq_per_img:(i+1)*seq_per_img,att_len[i]:attr_len[i]+att_len[i],:]
        memory_rela[i*seq_per_img:(i+1)*seq_per_img,:rela_len[i], :] = memory[i*seq_per_img:(i+1)*seq_per_img,attr_len[i]+att_len[i]:attr_len[i]+att_len[i]+rela_len[i],:]
    mask_att=mask_att.unsqueeze(1)
    mask_attr=mask_attr.unsqueeze(1)
    mask_rela=mask_rela.unsqueeze(1)
    return memory_att, memory_attr, memory_rela, mask_att, mask_attr, mask_rela

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory_att, memory_attr, memory_rela, mask_att, mask_attr, mask_rela, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory_att, memory_attr, memory_rela, mask_att, mask_attr, mask_rela, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn_att, src_attn_attr, src_attn_rela, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn_att = src_attn_att
        self.src_attn_attr = src_attn_attr
        self.src_attn_rela = src_attn_rela
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 5)
 
    def forward(self, x, memory_att, memory_attr, memory_rela, mask_att, mask_attr, mask_rela, tgt_mask):
        "Follow Figure 1 (right) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x_att = self.sublayer[1](x, lambda x: self.src_attn_att(x, memory_att, memory_att, mask_att))
        x_attr = self.sublayer[2](x, lambda x: self.src_attn_attr(x, memory_attr, memory_attr, mask_attr))
        x_rela = self.sublayer[3](x, lambda x: self.src_attn_rela(x, memory_rela, memory_rela, mask_rela))
        x = mod_controller(x_att, x_attr, x_rela, x)
        return self.sublayer[4](x, self.feed_forward)

def mod_controller(m_att, m_attr, m_rela, query):
    m = torch.stack((m_att, m_attr, m_rela), dim= 3) #m:50*17*512*3, query 50*17*512
    d_k = query.size(-1)
    query = query.unsqueeze(-1)
    scores = torch.matmul(query.transpose(-2, -1), m) / math.sqrt(d_k) #scores:50*17*3
    weights = F.softmax(scores, dim = -1) #scores:50*17*3
    out = torch.matmul(m, weights.transpose(-2, -1))
    out = out.squeeze(-1) #output:50*17*512*3
    return out

def subsequent_mask(size, all_former=0):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    if all_former:
        subsequent_mask = np.ones(attn_shape).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 1
    else:
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    # print(scores.shape)
    # print(mask.shape)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TSGMModel2(AttModel):

    def make_model(self, src_vocab, tgt_vocab, N_enc=6, N_dec=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1, opt={}):
        "Helper: Construct a model from hyperparameters."
        self.opt = opt
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, dropout)
        attn_dec = MultiHeadedAttention(opt.h_dec, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),
            Decoder(DecoderLayer(d_model, c(attn), c(attn_dec), c(attn_dec), c(attn_dec),
                                 c(ff), dropout), N_dec),
            lambda x:x, # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            Generator(d_model, tgt_vocab), opt = self.opt)
        # This was important from their code. 
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, opt):
        super(TSGMModel2, self).__init__(opt)
        self.opt = opt
        # self.config = yaml.load(open(opt.config_file))
        
        self.N_enc = getattr(opt, 'N_enc', opt.num_layers)
        self.N_dec = getattr(opt, 'N_dec', opt.num_layers)
        self.d_model = getattr(opt, 'd_model', opt.input_encoding_size)
        self.d_ff = getattr(opt, 'd_ff', opt.rnn_size)
        self.h = getattr(opt, 'num_att_heads', 8)
        self.dropout = getattr(opt, 'dropout', 0.1)
        self.all_former = getattr(opt, 'all_former', 0)
        self.concat_type = getattr(opt, 'concat_type', 0)
        self.controller = getattr(opt, 'controller', 0)

        delattr(self, 'att_embed')
        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.att_feat_size, self.d_model),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.d_model),) if self.use_bn==2 else ())))
        self.rela_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Embedding(500, self.d_model),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.d_model),) if self.use_bn==2 else ())))
        
        delattr(self, 'embed')
        self.embed = lambda x : x
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x : x
        delattr(self, 'logit')
        del self.ctx2att

        tgt_vocab = self.vocab_size + 1


        self.model = self.make_model(0, tgt_vocab,
            N_enc=self.N_enc,
            N_dec=self.N_dec,
            d_model=self.d_model,
            d_ff=self.d_ff,
            h=self.h,
            dropout=self.dropout, opt = self.opt)

    def logit(self, x): # unsafe way
        return self.model.generator.proj(x)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, att_feats, att_masks, enc_self_masks, rela_seq, rela_len, attr_len, att_len):
        att_feats, seq, att_masks, seq_mask, enc_self_masks = self._prepare_feature_forward(att_feats, att_masks,
                                                                                            enc_self_masks, rela_seq,
                                                                                            rela_len, attr_len, att_len)
        memory = self.model.encode(att_feats, enc_self_masks)

        return memory, att_masks, seq_mask, enc_self_masks

    def _prepare_feature_forward(self, att_feats, att_masks, enc_self_masks, rela_seq, rela_len, attr_len, att_len, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        att_feats = self.att_embed(att_feats)
        rela_feats = self.rela_embed(rela_seq)

        for i in range(len(rela_len)):
            att_feats[i,att_len[i]:att_len[i]+attr_len[i],:] = rela_feats[i,:attr_len[i],:] + rela_feats[i,attr_len[i]:2*attr_len[i],:]
            att_feats[i,att_len[i]+attr_len[i]:att_len[i]+attr_len[i]+rela_len[i], :] = rela_feats[i,2*attr_len[i]:2*attr_len[i]+rela_len[i],:]

        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            seq = seq[:,:-1]
            seq_mask = (seq.data > 0)
            seq_mask[:,0] = 1 # bos

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)

            seq_per_img = seq.shape[0] // att_feats.shape[0]
            if seq_per_img > 1:
                att_feats, att_masks = utils.repeat_tensors(seq_per_img,
                    [att_feats, att_masks]
                )
                enc_self_masks = utils.repeat_tensors(seq_per_img, enc_self_masks)
        else:
            seq_mask = None

        #seq: 17, [0,7961,xxx,] seq_mask: 17, [[1,0,0],[1,1,0]]
        return att_feats, seq, att_masks, seq_mask, enc_self_masks

    def _forward(self, fc_feats, att_feats, seq, att_masks, enc_self_masks, rela_seq, rela_len, attr_len, att_len):
        if seq.ndim == 3:  # B * seq_per_img * seq_len
            seq = seq.reshape(-1, seq.shape[2])
        att_feats, seq, att_masks, seq_mask, enc_self_masks = self._prepare_feature_forward(att_feats, att_masks, enc_self_masks, rela_seq, rela_len, attr_len, att_len, seq)

        out = self.model(att_feats, seq, enc_self_masks, att_masks, seq_mask, rela_len, attr_len, att_len)
        outputs = self.model.generator(out)
        return outputs

    def core(self, it, memory, state, mask, rela_len, attr_len, att_len):
        """
        state = [ys.unsqueeze(0)]
        """
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.decode(memory, mask,
                               ys, 
                               subsequent_mask(ys.size(1), self.all_former)
                                        .to(memory.device), rela_len, attr_len, att_len)

        return out[:, -1], [ys.unsqueeze(0)]

    def _sample(self, fc_feats, att_feats, att_masks, enc_self_masks, rela_seq, rela_len, attr_len, att_len, opt={}):

        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        sample_n = int(opt.get('sample_n', 1))
        group_size = opt.get('group_size', 1)
        output_logsoftmax = opt.get('output_logsoftmax', 1)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, att_masks, enc_self_masks, rela_seq, rela_len, attr_len, att_len,  opt)
        if group_size > 1:
            return self._diverse_sample(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size * sample_n)
        att_feats, att_masks, seq_mask, enc_self_masks = self._prepare_feature(att_feats, att_masks,enc_self_masks, rela_seq,rela_len, attr_len, att_len)

        if sample_n > 1:
           att_feats, att_masks = utils.repeat_tensors(sample_n,
                [att_feats, att_masks]
            )

        trigrams = []  # will be a list of batch_size dictionaries

        seq = fc_feats.new_zeros((batch_size * sample_n, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, self.seq_length, self.vocab_size + 1)
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.new_zeros(batch_size * sample_n, dtype=torch.long)

            logprobs, state = self.get_logprobs_state(it, att_feats, att_masks, state, rela_len, attr_len, att_len,
                                                      output_logsoftmax=output_logsoftmax)

            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            if remove_bad_endings and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                prev_bad = np.isin(seq[:, t - 1].data.cpu().numpy(), self.bad_endings_ix)
                # Make it impossible to generate bad_endings
                tmp[torch.from_numpy(prev_bad.astype('uint8')), 0] = float('-inf')
                logprobs = logprobs + tmp

            # Mess with trigrams
            # Copy from https://github.com/lukemelas/image-paragraph-captioning
            if block_trigrams and t >= 3:
                # Store trigram generated at last step
                prev_two_batch = seq[:, t - 3:t - 1]
                for i in range(batch_size):  # = seq.size(0)
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current = seq[i][t - 1]
                    if t == 3:  # initialize
                        trigrams.append({prev_two: [current]})  # {LongTensor: list containing 1 int}
                    elif t > 3:
                        if prev_two in trigrams[i]:  # add to list
                            trigrams[i][prev_two].append(current)
                        else:  # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = seq[:, t - 2:t]
                mask = torch.zeros(logprobs.size(), requires_grad=False).cuda()  # batch_size x vocab_size
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i, j] += 1
                # Apply mask to log probs
                # logprobs = logprobs - (mask * 1e9)
                alpha = 2.0  # = 4
                logprobs = logprobs + (mask * -0.693 * alpha)  # ln(1/2) * alpha (alpha -> infty works best)

            # sample the next word
            if t == self.seq_length:  # skip if we achieve maximum length
                break
            it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature)

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:, t] = it
            seqLogprobs[:, t] = logprobs
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs

    def _sample_beam(self, fc_feats, att_feats, att_masks, enc_self_masks, rela_seq, rela_len, attr_len, att_len, opt={}):
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        sample_n = opt.get('sample_n', 10)
        # when sample_n == beam_size then each beam is a sample.
        assert sample_n == 1 or sample_n == beam_size // group_size, 'when beam search, sample_n == 1 or beam search'
        batch_size = fc_feats.size(0)

        att_feats, att_masks, seq_mask, enc_self_masks = self._prepare_feature(att_feats, att_masks, enc_self_masks,
                                                                               rela_seq, rela_len, attr_len, att_len)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = fc_feats.new_zeros((batch_size * sample_n, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, self.seq_length, self.vocab_size + 1)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]

        state = self.init_hidden(batch_size)

        # first step, feed bos
        it = fc_feats.new_zeros([batch_size], dtype=torch.long)


        logprobs, state = self.get_logprobs_state(it, att_feats, att_masks, state, rela_len, attr_len, att_len)
        att_feats, att_masks = utils.repeat_tensors(beam_size,
                                                    [att_feats, att_masks]
                                                    )

        # logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)
        # p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(beam_size,
        #                                                                           [p_fc_feats, p_att_feats,
        #                                                                            pp_att_feats, p_att_masks]
        #                                                                           )
        self.done_beams = self.beam_search(state, logprobs, rela_len, attr_len, att_len, att_feats, att_masks, opt=opt)
        for k in range(batch_size):
            if sample_n == beam_size:
                for _n in range(sample_n):
                    seq_len = self.done_beams[k][_n]['seq'].shape[0]
                    seq[k * sample_n + _n, :seq_len] = self.done_beams[k][_n]['seq']
                    seqLogprobs[k * sample_n + _n, :seq_len] = self.done_beams[k][_n]['logps']
            else:
                seq_len = self.done_beams[k][0]['seq'].shape[0]
                seq[k, :seq_len] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
                seqLogprobs[k, :seq_len] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq, seqLogprobs


    def get_logprobs_state(self, it, att_feats, att_masks, state, rela_len, attr_len, att_len, output_logsoftmax=1):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state = self.core(xt, att_feats, state, att_masks, rela_len, attr_len, att_len)
        if output_logsoftmax:
            logprobs = F.log_softmax(self.logit(output), dim=1)
        else:
            logprobs = self.logit(output)

        return logprobs, state
