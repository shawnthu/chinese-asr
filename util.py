from gpd import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
from Levenshtein import editops, distance

from warnings import warn
from operator import itemgetter
from collections import namedtuple
from time import time
import os
from functools import partial
import random
from itertools import accumulate
from typing import List, Tuple


# --------------------- functions ---------------------
def preprocess_text(text_list):
    """
    1. 全都大写
    :param text_list: list of string
    :return:
    """
    return [txt.upper() for txt in text_list]


def round_down(x, ndigits=0):
    """
    :param x: float
    :param ndigits: int, def 0
    :return:
    """
    return round(int(x * (10 ** ndigits)) / (10 ** ndigits), ndigits)


def tile_batch(t, k, batch_first=False):
    """
    :param t: [b, l, d] or [b, l] if batch_first, else [l, b, d] or [l, b]
    :param k: int
    :return: [l, b * k, d] or [l, b * k]
    """
    if batch_first:
        # [b, ...] -> [b, k, ...] -> [b * k, ...]
        shape = t.shape
        t = t.unsqueeze(dim=1).expand(-1, k, *shape[1:]).contiguous().view(-1, *shape[1:])
    else:
        # [l, b, ...] -> [l, b, k, ...] -> [l, b * k, ...]
        shape = t.shape
        t = t.unsqueeze(dim=2).expand(-1, -1, k, *shape[2:]).contiguous().view(-1, shape[1] * k, *shape[2:])

    return t


def to_device(inp, device):
    if isinstance(inp, (list, tuple)):
        inp = [ele.to(device) for ele in inp]
    elif isinstance(inp, torch.Tensor):
        inp = inp.to(device)
    else:
        warn("input must be torch.Tensor or sequence of torch.Tensor")
    return inp


def init_lstm(m):
    for name, param in m.named_parameters(prefix='', recurse=True):
        # if name.startswith('bias'):
        if 'bias' in name:
            nn.init.zeros_(param)
            # 添加forget gate bias，在pytorch中，门的顺序是i, f, c, o，因此1/4 ~ 1/2初始化为1／2
            assert param.dim() == 1
            size = param.size(0)
            nn.init.constant_(param[size // 4: size // 2], .5)

        # elif name.startswith('weight_hh'):
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param)
        # elif name.startswith('weight_ih'):
        elif 'weight_ih' in name:
            nn.init.xavier_normal_(param)
        else:
            warn('unknown parameter name in lstm, do not contain bias or weight')
            nn.init.xavier_normal_(param)


def init_rnn(m, mode):
    """
    rnn.0/1/2.weight_ih/hh_l0
    rnn.0/1/2.weight_ih/hh_l0_reverse
    rnn.0/1/2.bias_ih/hh_l0
    rnn.0/1/2.bias_ih/hh_l0_reverse
    """
    for name, param in m.named_parameters(prefix='', recurse=True):
        if 'bias' in name:
            nn.init.zeros_(param)
            if mode == 'LSTM':
                # 添加forget gate bias，在pytorch中，门的顺序是i, f, c, o，因此1/4 ~ 1/2初始化为1/2
                assert param.dim() == 1
                size = param.size(0)
                nn.init.constant_(param[size // 4: size // 2], .5)

        elif 'weight_hh' in name:
            nn.init.orthogonal_(param)

        elif 'weight_ih' in name:
            nn.init.xavier_normal_(param)

        else:
            warn('unknown parameter name, do not contain bias or weight')
            nn.init.xavier_normal_(param)


def get_mask(lens, max_len=None):
    """
    Get a mask, where 1 for none-padding and 0 for padding \n
    :param lens: [b], torch.int
    :param max_len: scalar, int
    :return: [l, b] torch.float
    """
    bsz = lens.size(0)
    mask = torch.arange(max_len or lens.max(), dtype=lens.dtype,
                        device=lens.device).expand(bsz, -1) < lens.view(-1, 1)  # [b, l]
    mask = mask.t().float()  # [l, b]
    return mask


def get_mask_for_softmax(lens, max_len=None):
    """
    Get a mask, where -inf value for padding and 0 for non-padding \n
    :param lens: [b]
    :param max_len: scalar, int
    :return: mask [l, b]
    """
    mask = torch.arange(max_len or lens.max().item(), dtype=lens.dtype,
                        device=lens.device).expand(lens.size(0), -1) >= lens.view(-1, 1)
    mask = mask.t().float()  # .to(torch.float32) .double() <==> .to(torch.double)
    mask.masked_fill_(mask == 1., -np.inf)  # [l, b]
    return mask


def pad(x, l, ks, stride):
    """
    :param x: [b, d, l]
    :param l: int, max valid length
    :param ks: int
    :param stride: int
    :return:
    """
    right_pad = (l - ks) % stride
    if right_pad > 0:
        right_pad = stride - right_pad
        right_pad = max(l + right_pad - x.size(-1), 0)
        x = F.pad(x, (0, right_pad))
    return x


def pad2d(x, h_w, ks, stride):
    """
    :param x: [b, d, h, w]
    :param h_w: tuple of int, max valid length
    :param ks: int or tuple
    :param stride: int or tuple
    :return:
    """
    ks = (ks, ks) if isinstance(ks, int) else ks
    stride = (stride, stride) if isinstance(stride, int) else stride

    right_pad = (h_w[1] - ks[1]) % stride[1]
    if right_pad > 0:
        right_pad = stride[1] - right_pad
        right_pad = max(h_w[1] + right_pad - x.size(-1), 0)

    bottom_pad = (h_w[0] - ks[0]) % stride[0]
    if bottom_pad > 0:
        bottom_pad = stride[0] - bottom_pad
        bottom_pad = max(h_w[0] + bottom_pad - x.size(-2), 0)

    x = F.pad(x, (0, right_pad, 0, bottom_pad))
    return x


def get_wer_python(pred, ref, normalize=True):
    """
    计算a (len(a)=m) 和b (len(b)=n) 编辑距离一般思想：
    1. 构造一个矩阵d，d[i, j]表示a[1: i]到b[1: j]的距离，
       1.1 如果a[i] = b[j]，d[i, j] = d[i-1, j-1]
       1.2 否则，d[i][j]等于以下三个中的最小值
         1.2.1 删除a[i]，d[i-1, j] + 1（也就是额外多一个删除操作）
         1.2.2 删除b[j]，d[i, j-1] + 1（也就是额外多一个增加操作）
         1.2.3 d[i-1, j-1] + 1（也就是额外多一个替换操作）
       计算量：m×n×3
    2. 实际操作：
    i\j 0 1 2 3
      0 0 1 2 3 -> 对应到程序中的dist数组
      1 1 * * *
      2 2 * * *
      3 3 * * *
    每次计算d[i, j]，只需要d[i, j-1], d[i-1, j], d[i-1, j-1]，其实只需要维护一个列表l和
    old变量即可，l[j-1] = d[i, j-1], l[j] = d[i-1, j], old = d[i-1, j-1]，然后更新l[j]
    即可
    :param pred: list, length n
    :param ref: list, length m
    :param normalize: default True
    :return: 从ref到pred的距离，返回一个四元组(all, insert, delete, substitute)
    """
    # 空间复杂度n
    m, n = len(ref), len(pred)
    assert m > 0, "empty reference not allowed!"
    if n == 0:
        return 1. if normalize else m
    # row wise
    # 初始化，ref为空时，ref到pred从空到完整的距离
    dist = np.arange(n + 1, dtype='int')
    # 0, 1, 2, 3, 4, 5, ...n
    for i in range(1, m + 1):  # ref
        pre = i  # ref[1: i] -> pred[1: j-1]的编辑距离，insert error
        # 计算ref[1: i] -> pred[1: j]的编辑距离
        for j in range(1, n + 1):  # pred
            if pred[j - 1] == ref[i - 1]:  # 最后一个字符相等
                cur = dist[j - 1]  # 斜对角
            else:
                cur = min(pre, dist[j], dist[j - 1]) + 1  # 分别对应insert、delete、substitute
            dist[j - 1] = pre
            pre = cur
        dist[-1] = cur
    if normalize:
        out = dist[-1] / (1. * len(ref))
    else:
        out = dist[-1]
    return out


def get_wer(pred, ref, normalize=True, return_tuple=False):
    """
    :param pred: str
    :param ref: str
    :param normalize: boolean
    :return: tuple (all, insert, delete, replace)
    """
    n = len(ref) * 1.

    if not return_tuple:
        r = distance(pred, ref)
        if normalize:
            return r / n
        else:
            return r

    r = editops(pred, ref)
    # ret = (0, ) * 4
    d = {'insert': 0, 'delete': 0, 'replace': 0}
    for ele in r:
        d[ele[0]] += 1
    r = (sum(d.values()), d['insert'], d['delete'], d['replace'])
    if normalize:
        return tuple((ele / n for ele in r))
    else:
        return r


def label_smoothing(logits, targets, ls_value=gpd['label_smooth']):
    """
    assign ls_value / (k - 1) to non-target logit
    :param logits: [b, k]
    :param targets: [b]
    :param ls_value: a small float value, [0, 1)
    :return: scalar loss
    """
    b, k = logits.size()
    lse = torch.logsumexp(logits, 1)  # [b], log sum exp
    target_logit = logits.gather(1, targets[:, None])[:, 0]  # [b]
    other_logit_sum = logits.sum(1) - target_logit  # [b]
    loss = (1 - ls_value) * target_logit + \
           (ls_value / (k - 1)) * other_logit_sum - lse
    return -loss


def label_smoothing_old(logits, targets, ls_value=gpd['label_smooth']):
    """
    assign ls_value / k to non-target logit
    :param logits: [b, k]
    :param targets: [b]
    :param ls_value: a small float value, [0, 1)
    :return: scalar loss
    """
    b, k = logits.size()
    logp = F.log_softmax(logits, dim=1)
    epsion = ls_value / k
    loss = epsion * logp.sum(dim=1) + (1. - ls_value) * logp.gather(dim=1, index=targets[:, None])[:, 0]
    loss = -loss
    return loss


def rand_disp_list(*lst, n):
    rand_disp_idx = np.random.permutation(len(lst[0]))[:n]
    disp_s = ''
    for i, idx in enumerate(rand_disp_idx, 1):
        for l in lst:
            disp_s += f"\t{i:>2d} {l[idx]}\n"
    print(disp_s, end='')


def log_img(logger, img, flag, iteration):
    """
    :param logger: instance of Logger
    :param img: list: tensor [l, b] or [l, b, n]
    :param flag: train, dev or test
    :param iteration: int
    :return: None
    """
    assert flag in ('train', 'dev', 'test')

    for ele in img:
        if ele.size == 0:  # array.size <=> tensor.nelement()
            print(f"[WARN] Do not log alignment image at iteration {iteration}, empty image!")
            return None

    if img[0].ndim == 2:  # array.ndim <=> tensor.dim()
        logger.image(f'{flag}_alignment', img, iteration)
    elif img[0].ndim == 3:
        for h in range(1, 1 + img[0].shape[-1]):
            logger.image(f'{flag}_alignment_head_{h}',
                         [ele[..., h - 1] for ele in img], iteration)
    else:
        raise Exception('unknown alignment dim, must be [l, b] or [l, b, n]')


def parse_batch_alignment(alignment, audio_len, text_len, max_nb=2, idx=None):
    # alignment: list of tensor [l, b] or [l, b, n], maybe different batch size
    # audio_len: tensor [b]
    # text_len: tensor [b]
    # max_nb: int, def 2
    # idx: list of int, def None

    alignment = torch.stack(alignment, dim=0)  # [l_y, l_x, b], text-audio-batch
    # alignment = nn.utils.rnn.pad_sequence(alignment, True)  # [l_y, l_x, b]
    # or [l_y, l_x, b, n]
    bsz = alignment.size(2)
    if idx is None:
        idx = random.sample(range(bsz), max_nb)  # list of int, length max_nb
    alignment = [
        a[:t_l, :a_l] for a, a_l, t_l in
        zip(alignment[:, :, idx].unbind(2), audio_len[idx], text_len[idx])
    ]  # list: [l_y, l_x] or [l_y, l_x, n]
    alignment = [(ele * 255).to(torch.uint8).cpu().numpy() for ele in alignment]  # 越白值越大
    # list: [l_y, l_x] or [l_y, l_x, n]
    # list: int
    return alignment, idx


def parse_multi_batch_alignment(alignment, audio_len, text_len, max_nb=2):
    # alignment: list of list of tensor
    # audio_len: list of tensor
    # text_len: list of tensor
    assert isinstance(alignment[0], list)

    bszs = [ele.size(0) for ele in audio_len]
    # print(f"bszs: {bszs}")
    cum_bsz = list(accumulate(bszs))
    batch_idx_and_idx = sample_batch_idx(bszs, max_nb)[1]  # list of tuple(batch_idx, [idx])

    full_alignment = []
    full_idx = []
    for batch_idx, idx in batch_idx_and_idx:
        alignment_, _ = parse_batch_alignment(
            alignment[batch_idx], audio_len[batch_idx], text_len[batch_idx], idx=idx)
        full_alignment.extend(alignment_)
        base_idx = cum_bsz[batch_idx - 1] if batch_idx > 0 else batch_idx
        full_idx.extend([base_idx + i for i in idx])

    return full_alignment, full_idx


def sample_batch_idx(bszs, nb):
    cum_len = np.array(list(accumulate(bszs)))
    idx = random.sample(range(cum_len[-1]), nb)

    batch_idx = []
    for i in idx:
        first_zero_ind = np.where(i // cum_len == 0)[0][0]
        if first_zero_ind == 0:
            batch_idx.append((first_zero_ind, i))
        else:
            batch_idx.append((first_zero_ind, i - cum_len[first_zero_ind - 1]))

    batch_idx_dict = dict()
    for ele in batch_idx:
        if ele[0] not in batch_idx_dict:
            batch_idx_dict[ele[0]] = [ele[1]]
        else:
            batch_idx_dict[ele[0]].append(ele[1])

    return idx, list(batch_idx_dict.items())


def parse_alignemt_old(alignments, text, max_nb=2):
    """
    :param alignment: list of list of [l, b, n] or [l, b] tensor, may be different batch_size
    :param text: list of list of text
    :return: numpy [b, l_y, l_x]
    """
    # 先反转
    if alignments[0].dim() == 3:
        heads = alignments[0].size(-1)
        alignments = [ele.view(ele.size(0), -1) for ele in alignments]  # convert to list of [l, b * n]
    else:
        heads = 1

    alignments = [ele.t() for ele in alignments]  # [b, l_x]
    pad = nn.utils.rnn.pad_sequence(alignments, batch_first=True)  # [b, l_y, l_x]
    bsz = pad.size(0) // heads
    idx = torch.randperm(bsz)[:max_nb]  # [m]
    if heads > 1:
        idx = idx[:, None] * heads + torch.arange(heads).view(-1)
    pad = (pad[idx] * 255).to(torch.uint8).cpu().numpy()  # 越白值越大
    if heads > 1:
        b, l_y, l_x = pad.shape()
        pad = pad.reshape(b // heads, heads, l_y, l_x)
    return pad, idx


def tensor2words(outs, lens, int2word):
    if outs:
        outs = torch.stack(outs, dim=1)  # [b, l]
        outs = [seq[:l] for seq, l in zip(outs.unbind(0), lens.unbind(0))]
        outs = [[int2word[i] for i in sample.tolist()] for sample in outs]
    else:
        outs = []
    return outs  # list of list of word


def set_opt_lr(opt, lr, ratio=None):
    if lr:
        assert ratio is None
    else:
        assert ratio is not None

    for p in opt.param_groups:
        if ratio:
            p['lr'] *= ratio
        else:
            p['lr'] = lr


def get_grad_norm(parameters):
    grad_norm = 0
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    for p in parameters:
        param_norm = (p.grad.data ** 2).sum()
        grad_norm += param_norm.item()
    grad_norm = grad_norm ** .5
    return grad_norm


def compute_self_attention(q, k, v, lens, heads, proj_weight=None):
    """
    :param q: [b, l, d]
    :param k: [b, l, d]
    :param v: [b, l, d]
    :param lens: [b] or None
    :param heads: int
    :param proj_weight: default None
    :return: tuple, (attention, alignment):
    if heads == 1: ([b, l, d], [b, l, l])
    if heads > 1: ([b, l, d], [b, n, l, l])
    """
    bsz, l, d = q.size()

    if heads > 1:
        assert (d // heads) * heads == d, "heads must be divided by query dim"

    # 1. get the mask and compute alignment
    if heads > 1:
        q = q.view(bsz, l, heads, -1).transpose(1, 2)  # [b, n, l, d // n]
        k = k.view(bsz, l, heads, -1).transpose(1, 2)  # [b, n, l, d // n]
        alignment = q.matmul(k.transpose(2, 3))  # [b, n, l_q, l_k]
    else:
        alignment = q.bmm(k.transpose(1, 2))  # [b, l_q, l_k]

    if lens is not None:
        mask_key = get_mask_for_softmax(lens, l).t()  # [b, l_k]

        if heads > 1:
            alignment = alignment + mask_key[:, None, None]
        else:
            alignment = alignment + mask_key[:, None]
    alignment = F.softmax(alignment, -1)  # [b, l_q, l_k] or [b, n, l_q, l_k]

    # 2. compute the self attention
    if heads > 1:
        v = v.view(bsz, l, heads, -1).transpose(1, 2)  # [b, n, l_k, d // n]
        attn = alignment.matmul(v)  # [b, n, l_q, d // n]
        attn = attn.transpose(1, 2).contiguous().view(bsz, l, -1)
        if proj_weight is not None:
            attn = F.linear(attn, proj_weight)
    else:
        attn = alignment.bmm(v)  # [b, l_q, d]

    # 3. post mask
    if lens is not None:
        mask_query = get_mask(lens, l).t()  # [b, l_q]
        attn = attn * mask_query[..., None]

    return attn, alignment


def compute_self_local_attention(q, k, v, lens, ws, heads, proj_weight=None):
    """
    :param q: [b, l, d]
    :param k: [b, l, d]
    :param v: [b, l, d]
    :param lens: None or [b]
    :param ws: int
    :param heads: int
    :param proj_weight: default None
    :return: tuple, (attention, alignment)
    if heads == 1, ([b, l, d], [b, l, l])
    if heads > 1, ([b, l, d], [b, n, l, l])
    """
    device = q.device
    bsz, l, d = q.size()

    if heads > 1:
        assert (d // heads) * heads == d, "heads must be divided by query dim"

    if lens is not None:
        mvl = lens.max().item()  # max valid length

    else:
        mvl = l

    if mvl <= ws:
        # 如果最大都小于ws，则just pure self attention
        return compute_self_attention(q, k, v, lens, heads, proj_weight)

    n = ws // 2

    # 1. get the start index
    """
    假设有效长度为l，window size是ws，n = ws // 2，则每个query对应的key的start index是：
    1. l >= ws
    0 <= torch.arange(l) - n <= l - ws
    2. l < ws
    0 <= torch.arange(l) - n <= max(l - ws, 0)，其实都是0
    """

    # 2. get the whole index
    if lens is not None:
        # 每个sample的valid length都不同，因此l - ws也都不同！！！
        batch_idx = []
        for i in range(bsz):
            l_ = lens[i].item()
            l_upper = max(0, l_ - ws)
            idx = torch.arange(l_) - n  # always cpu
            idx.masked_fill_(idx < 0, 0)
            idx.masked_fill_(idx > l_upper, l_upper)

            idx = idx[:, None] + torch.arange(ws, dtype=idx.dtype)  # 注意！会导致后面需要mask
            idx = idx.view(-1)  # [l * ws]

            batch_idx.append(idx)

        batch_idx = nn.utils.rnn.pad_sequence(batch_idx, True).to(device)  # [b, mvl * ws]

    else:
        l_upper = max(0, mvl - ws)
        idx = torch.arange(mvl) - n
        idx.masked_fill_(idx < 0, 0)  # 小于零的全部置零
        idx.masked_fill_(idx > l_upper, l_upper)  # 大于mvl-ws的全部置于mvl-ws
        idx = idx[:, None] + torch.arange(ws, dtype=idx.dtype)  # [mvl, ws]
        idx = idx.view(-1)  # [mvl * ws]
        batch_idx = idx.to(device)

    # 3. get the new key
    if lens is not None:
        k = k.gather(1, batch_idx.unsqueeze(-1).expand(-1, -1, d))  # [b, mvl * ws, d]
        k = k.view(bsz, mvl, ws, -1)  # [b, mvl, ws, d]
    else:
        k = k[:, batch_idx].view(bsz, mvl, ws, -1)  # [b, mvl, ws, d]
    new_k = k.new_zeros((bsz, l, ws, d))  # [b, l, ws, d], no grad
    new_k[:, :mvl] = k  # if k grad, then new_k grad !!!
    k = new_k  # [b, l, ws, d]
    del new_k

    # 4. get the alignment
    if lens is not None:
        # [b, ws]
        mask_key = get_mask_for_softmax(lens.min(lens.new_full([], ws)), ws).t()

    if heads > 1:
        # [b, l, d] -> [b, l, n, d // n] -> [b, n, l, d // n]
        q = q.view(bsz, l, heads, d // heads).transpose(1, 2)
        # [b, l, ws, d] -> [b, l, ws, n, d // n] -> [b, n, l, ws, d // n]
        k = k.view(bsz, l, ws, heads, d // heads).permute(0, 3, 1, 2, 4)
        alignment = k.matmul(q[..., None]).squeeze(-1)  # [b, n, l, ws]
    else:
        alignment = k.matmul(q[..., None]).squeeze(-1)  # [b, l, ws]

    if lens is not None:
        alignment = alignment + mask_key[:, None, None] if heads > 1 else mask_key[:, None]
    alignment = F.softmax(alignment, -1)

    # 5. get the value
    if lens is not None:
        v = v.gather(1, batch_idx.unsqueeze(-1).expand(-1, -1, d)).view(bsz, mvl, ws, -1)  # [b, mvl, ws, d]
    else:
        v = v[:, batch_idx].view(bsz, mvl, ws, -1)  # [b, mvl, ws, d]
    new_v = v.new_zeros(bsz, l, ws, d)
    new_v[:, :mvl] = v
    v = new_v  # [b, l, ws, d]
    del new_v

    # 6. get the attention
    if heads > 1:
        # [b, l, ws, d] -> [b, l, ws, n, d // n] -> [b, n, l, ws, d // n]
        v = v.view(bsz, l, ws, heads, d // heads).permute(0, 3, 1, 2, 4)
        attn = alignment.unsqueeze(3).matmul(v).squeeze(3)  # [b, n, l, d // n]
        attn = attn.transpose(1, 2).contiguous().view(bsz, l, -1)  # [b, l, d]
    else:
        attn = alignment.unsqueeze(2).matmul(v).squeeze(2)  # [b, l, d]

    # 7. project
    if proj_weight is not  None:
        attn = F.linear(attn, proj_weight)

    # 8. post mask
    if lens is not None:
        mask = get_mask(lens, l).t()  # [b, l]
        attn = mask[..., None] * attn

    return attn, alignment


def get_shape(x):
    if isinstance(x, (tuple, list)):
        assert hasattr(x[0], 'shape')
        return type(x)([ele.shape for ele in x])
    else:
        assert hasattr(x[0], 'shape')
        return x.shape


def get_steps(x, base_steps):
    """
    when x < 0, means round(-x * base_steps)
    :param x: positive int or negative number (including int and float)
    :param base_steps: int
    :return: int
    """
    assert isinstance(base_steps, int), "base steps must be int"
    if x < 0:
        x = round(-x * base_steps)
    return x


def get_conv_length(l, ks_stride):
    """
    :param l: int
    :param ks_stride: list,
    :return:
    """
    assert isinstance(ks_stride, list), f"{type(ks_stride)} is not list"
    assert isinstance(ks_stride[0], tuple), f"{type(ks_stride[0])} is not tuple"
    for ks, stride in ks_stride:
        l = 1 + (l - ks + stride - 1) // stride
    return l


def reduce_lr_on_plateau(wer, tv, dec_rate_threshold, factor, patience, min_lr):
    if wer >= tv.best_wer * (1 - dec_rate_threshold):
        tv.num_no_imprv += 1
        if tv.num_no_imprv == patience:  # # 认为此时模型已经达到plateau, reduce lr
            warn(f"After {patience} times, "
                 f"eval wer do not improve! Reduce lr with factor {factor}!")
            tv.lr *= factor
            if tv.lr < min_lr:
                tv.lr = min_lr
            tv.num_no_imprv = 0
    else:
        tv.num_no_imprv = 0
        tv.best_wer = wer
        return True

    return False


def view_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    opt_state_dict = ckpt['optimizer_state_dict']  # dict
    """
    'state': [dict]
        param_address -> int: [dict]
            'step'
            'exp_avg'
            'exp_avg_sq'
    'param_groups': [list]
        'lr'
        'betas'
        'eps'
        'weight_decay'
        'amsgrad'
        'params' -> [list]' 这里有重复！！！为什么？？？
    """
    # print(f"keys:\n{opt_state_dict.keys()}\n\tstate keys: {opt_state_dict['state'].keys()}")
    state = opt_state_dict['state']
    param_groups = opt_state_dict['param_groups']
    print(f'length of param_groups: {len(param_groups)}')

    p_add = state.keys()
    step = [state[add]['step'] for add in p_add]
    print(f"length of step: {len(step)}, length of first params: {len(param_groups[0]['params'])}")
    print(f"lr: {param_groups[0]['lr']}")

    args = ckpt['args']
    print(vars(args))

    state_param_shape = {k: state[k]['exp_avg'].shape for k in state}
    all_param = param_groups[0]['params']
    print(len(state_param_shape), len(all_param), len(set(all_param)))


def get_rnn(mode, input_size, hidden_size, num_layers=1, bias=True, batch_first=False,
            droupout=0., bidirectional=False):
    assert mode in ('LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU')
    if mode == 'LSTM':
        rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                      bias=bias, batch_first=batch_first, dropout=droupout, bidirectional=bidirectional)
    elif mode == 'GRU':
        rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                      bias=bias, batch_first=batch_first, dropout=droupout, bidirectional=bidirectional)
    elif mode == 'RNN_TANH':
        rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                     nonlinearity='tanh', bias=bias, batch_first=batch_first, dropout=droupout,
                     bidirectional=bidirectional)
    elif mode == 'RNN_RELU':
        rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                     nonlinearity='relu', bias=bias, batch_first=batch_first, dropout=droupout,
                     bidirectional=bidirectional)
    else:
        raise ValueError("Unrecognized RNN mode: " + mode)

    return rnn


def get_sin_pos_embedding(lens, dim):
    # return [l, b, d], dtype: float32
    d = torch.arange(dim)
    d = 10000 ** ((d // 2 * 2).float() / dim)  # [d]
    # embed = torch.empty(l, dim, dtype=torch.float32)  # [l, d]
    embed = torch.empty(lens.max(), lens.size(0), dim, dtype=torch.float32)  # [l, b, d]
    # pos = torch.arange(l).float()  # [l]
    pos = [torch.arange(l).float() for l in lens]
    pos = nn.utils.rnn.pad_sequence(pos)  # [l, b]
    # sin
    embed[..., ::2] = torch.sin(pos[..., None] / d[None, None, ::2])
    embed[..., 1::2] = torch.cos(pos[..., None] / d[None, None, 1::2])
    embed = embed.to(lens.device)
    # mask
    mask = get_mask(lens)  # [l, b]
    embed = mask[..., None] * embed
    return embed


# --------------------- classes ---------------------
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()

        self.weight_ih = nn.Parameter(torch.randn(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(torch.randn(hidden_size, 4 * hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(4 * hidden_size))
        else:
            self.bias = 0

        self.hidden_size = hidden_size

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        gates = torch.mm(input, self.weight_ih) + torch.mm(hx, self.weight_hh) + self.bias
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class LSTM(nn.Module):
# class LSTM(torch.jit.ScriptModule):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False,
                 droupout=0., bidirectional=False):
        super().__init__()

        directions = 2 if bidirectional else 1
        self._all_params = []
        for i in range(num_layers):
            for j in range(directions):
                weight_ih = nn.Parameter(
                    torch.randn(input_size if i == 0 else hidden_size * directions, 4 * hidden_size))
                weight_hh = nn.Parameter(torch.randn(hidden_size, 4 * hidden_size))
                if bias:
                    bias = nn.Parameter(torch.zeros(4 * hidden_size))
                else:
                    bias = 0

                suffix = ' _reverse' if j == 1 else ''
                setattr(self, f"weight_ih_l{i}{suffix}", weight_ih)
                setattr(self, f"weight_hh_l{i}{suffix}", weight_hh)
                if bias:
                    setattr(self, f"bias_l{i}{suffix}", bias)

                self._all_params.append((weight_ih, weight_hh, bias))

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.directions = directions
        self.reset_parameters()

    def reset_parameters(self):
        pass

    # @staticmethod
    # @torch.jit.script_method
    def update(self, x, h, weight_hh):
        gate = x + h[0].mm(weight_hh)
        i, f, c, o = gate.chunk(4, dim=1)
        c = torch.sigmoid(f) * h[1] + torch.sigmoid(i) * torch.tanh(c)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c

    # @torch.jit.script_method
    def forward(self, x, h0):
        # x: [l, b, d]
        l, bsz, _ = x.shape

        # if h0 is None:
        #     h0 = x.new_zeros((self.num_layers * self.directions, bsz, self.hidden_size))
        #     h0 = (h0, h0)

        state = torch.jit.annotate(List[Tuple[torch.Tensor]], [])
        for layer in range(self.num_layers):
            y = torch.jit.annotate(List[torch.Tensor], [])
            for direction in range(self.directions):
                weight_ih, weight_hh, bias = self._all_params[layer * self.directions + direction]
                # pre-compute hidden state due to input
                h_input = x.matmul(weight_ih) + bias  # [l, b, d]
                h_input = h_input.unbind()
                # get the initial state
                h = (h0[0][layer * self.directions + direction],
                     h0[1][layer * self.directions + direction])

                for i in range(l):
                    h = self.update(h_input[i] if direction == 0 else h_input[l - 1 - i], h, weight_hh)
                    y.append(h[0])

                state.append(h)

            x = torch.stack(y) if self.directions == 1 else \
                torch.cat((torch.stack(y[:l]), torch.stack(y[:(l - 1):-1])), -1)
            # if self.directions == 1:
            #     x = torch.stack(y)
            # else:

        state = (torch.stack([ele[0] for ele in state]),
                 torch.stack([ele[1] for ele in state]))

        return x, state


class ConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        # 也可以把这两个conv合起来，用一个conv计算
        self.conv_x = nn.Conv1d(in_channels, out_channels * 4, kernel_size, 1)
        self.conv_h = nn.Conv1d(out_channels, out_channels * 4, kernel_size, 1)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # self.stride = stride if isinstance(stride, tuple) else (stride, stride)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.conv_x.weight)
        nn.init.zeros_(self.conv_x.bias)

        nn.init.xavier_normal_(self.conv_h.weight)
        nn.init.zeros_(self.conv_h.bias)

    def forward(self, x, lens=None, h_c=None):
        """
        variable width
        :param x: [b, c, h, w]
        :param lens: [b], def None
        :return:
        """
        # get the basic info
        bsz, _, height, width = x.size()

        # pad the height dimension
        top_pad = (self.kernel_size - 1) // 2
        bottom_pad = self.kernel_size - 1 - top_pad
        # left, right; top, bottom; front, back
        x = F.pad(x, (0, 0, top_pad, bottom_pad))

        # prepare the initial state
        if h_c is None:
            h = x.new_zeros((bsz, self.out_channels, height))  # [b, c', h]
            h_c = (h, h)

        # case 1: input with variable width, e.g. with padding
        if lens is not None:
            y = []
            correct_h_c = (x.new_empty(bsz, self.out_channels, height),
                           x.new_empty(bsz, self.out_channels, height))
            for i in range(width):
                inp = x[..., i]  # [b, c, h]
                h = F.pad(h_c[0], (top_pad, bottom_pad))  # [b, c', h] -> pad
                gates = self.conv_x(inp) + self.conv_h(h)  # gate: i, f, c, o
                i_gate, f_gate, c_gate, o_gate = gates.chunk(4, dim=1)  # [b, c' * 4, h]
                i_gate = torch.sigmoid(i_gate)
                f_gate = torch.sigmoid(f_gate)
                o_gate = torch.sigmoid(o_gate)
                c_gate = torch.tanh(c_gate)

                c = i_gate * c_gate + f_gate * h_c[1]
                h = o_gate * torch.tanh(c)

                mask = (i + 1) == lens
                correct_h_c[0][mask] = h[mask]
                correct_h_c[1][mask] = c[mask]

                h_c = (h, c)
                y.append(h)

            h_c = correct_h_c

            y = torch.stack(y, dim=3)  # [b, c', h, w]
            mask = get_mask(lens, width).t()  # [b, w]
            y = y * mask[:, None, None]

        # case 2: input without padding
        else:
            y = []
            for i in range(width):
                inp = x[..., i]  # [b, c, h]
                h = F.pad(h_c[0], (top_pad, bottom_pad))  # [b, c', h] -> pad
                gates = self.conv_x(inp) + self.conv_h(h)  # gate: i, f, c, o
                i_gate, f_gate, c_gate, o_gate = gates.chunk(4, dim=1)  # [b, c' * 4, h]
                i_gate = torch.sigmoid(i_gate)
                f_gate = torch.sigmoid(f_gate)
                o_gate = torch.sigmoid(o_gate)
                c_gate = torch.tanh(c_gate)

                c = i_gate * c_gate + f_gate * h_c[1]
                h = o_gate * torch.tanh(c)

                h_c = (h, c)
                y.append(h)

            y = torch.stack(y, dim=3)

        return y, lens, h_c


class BConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.fw = ConvLSTM(in_channels, out_channels, kernel_size)
        self.bw = ConvLSTM(in_channels, out_channels, kernel_size)

    @staticmethod
    def reverse_tensor(t, lens=None):
        if lens is None:
            return t[..., torch.arange(t.size(-1) - 1, -1, -1)]
        else:
            out = torch.zeros_like(t)
            for i in range(lens.size(0)):
                l = lens[i].item()
                out[i, :, :, :l] = t[i, :, :, torch.arange(l - 1, -1, -1)]
        return out

    def forward(self, x, lens=None, h_c=None):
        """
        variable widht
        :param x: [b, c, h, w]
        :param lens: [b], def None
        :param h_c: ([b, 2, c, h], [b, 2, c, h]), def None
        :return:
        """

        # 1. forward
        y_fw, _, h_c_fw = self.fw(x, lens, h_c if h_c is None else (h_c[0][:, 0], h_c[1][:, 0]))

        # 2. backward
        # reverse x
        x = BConvLSTM.reverse_tensor(x, lens)  # [b, c, h, w]
        y_bw, _, h_c_bw = self.bw(x, lens, h_c if h_c is None else (h_c[0][:, 1], h_c[1][:, 1]))
        # reverse result
        y_bw = BConvLSTM.reverse_tensor(y_bw, lens)

        # [b, 2, c', h, w], [b], ([b, 2, c', h], [b, 2, c', h])
        return torch.stack([y_fw, y_bw], dim=1), lens, (torch.stack([h_c_fw[0], h_c_bw[0]], dim=1),
                                                        torch.stack([h_c_fw[1], h_c_bw[1]], dim=1))


# TODO
class LocalRNN(nn.Module):
    def __init__(self, mode, input_size, hidden_size, num_layers=1, bias=True, batch_first=False,
                 droupout=0., bidirectional=False, res=False, skip_step=1, ksize=None):
        super().__init__()

        num_directions = 2 if bidirectional else 1
        # output: [L, b, num_directions * d]
        # state: [num_layers * num_directions, b, d]
        self.rnn = [get_rnn(mode, input_size if i == 0 else num_directions * hidden_size,
                            hidden_size, 1, bias=bias, batch_first=batch_first,
                            droupout=droupout, bidirectional=bidirectional)
                    for i in range(num_layers)]
        self.rnn = nn.ModuleList(self.rnn)

        self.num_directions = num_directions
        self.mode = mode
        self.layers = num_layers
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.res = res
        self.skip_step = (skip_step,) * num_layers if isinstance(skip_step, int) else skip_step
        self.ksize = (ksize,) * num_layers if not isinstance(ksize, (list, tuple)) else ksize

        self.reset_parameters()

    def reset_parameters(self):
        init_rnn(self.rnn, self.mode)

    def forward(self, x, state=None):
        """
        :param x: PackedSequence or Tensor [l, b, d] or [b, l, d]
        :param state: None, Tensor or tuple
        :return:
        """
        is_pack = isinstance(x, nn.utils.rnn.PackedSequence)

        # bsz = x.batch_sizes[0].item() if is_pack else \
        #     (x.size(0) if self.batch_first else x.size(1))

        # if state is None:
        #     # output: num_layers * num_directions
        #     # view: (num_layers, num_directions, batch_size, hidden_size)
        #     state = x.data.new_zeros(
        #         (self.layers * self.num_directions, bsz, self.hidden_size), requires_grad=False) \
        #         if is_pack else x.new_zeros(
        #         (self.layers * self.num_directions, bsz, self.hidden_size), requires_grad=False)
        #
        #     if self.mode == 'LSTM':
        #         state = (state, state)

        states = []
        for i, m in enumerate(self.rnn):
            if self.mode != 'LSTM':
                layer_state = state if state is None else state[i * self.num_directions: (i + 1) * self.num_directions]
            else:
                layer_state = state if state is None else \
                    (state[0][i * self.num_directions: (i + 1) * self.num_directions],
                     state[1][i * self.num_directions: (i + 1) * self.num_directions])

            # print(f"x type: {type(x)}\ntype state: {type(layer_state)} len(state): {len(layer_state)}\n"
            #       f"state shape: {layer_state[0].shape}, {layer_state[1].shape}\n"
            #       f"type(m): {type(m)}")
            # # ksize
            # ksize = self.ksize[i]
            # if ksize is not None:
            #     if not is_pack:
            #         # [l, b, d]
            #         # m = l + k - 1 // k
            #         # -> [l, b, d]
            #         l = x.size(1) if self.batch_first else x.size(0)
            #         remainder = l % ksize
            #
            #     else:
            #         # pack

            y, cur_state = m(x, layer_state)

            states.append(cur_state)

            # res
            if self.res and (i > 0):
                if is_pack:
                    assert (x.batch_sizes == y.batch_sizes).all()
                    x = nn.utils.rnn.PackedSequence(data=x.data + y.data, batch_sizes=x.batch_sizes)
                else:
                    x = x + y
            else:
                x = y

            # skip step
            step = self.skip_step[i]
            if step > 1:
                """
                0 * * * *
                1 * * *
                2 * *
                batch_sizes: [3, 3, 2]
                lens: [4, 3, 2]
                
                if skip step 为 2
                0 * *
                1 * *
                2 *
                batch_sizes: [3, 2]
                lens: [2, 2, 1]
                """
                if is_pack:
                    pad, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=False)  # [l, b, d]
                    pad = pad[::step]  # 如果最后一个skip step不够，则保留第一个
                    lens = (lens + step - 1) // step
                    x = nn.utils.rnn.pack_padded_sequence(pad, lens)
                else:
                    x = x[:, ::step] if self.batch_first else x[::step]

        if self.mode == 'LSTM':
            h = torch.cat(tuple(map(itemgetter(0), states)), dim=0)
            c = torch.cat(tuple(map(itemgetter(1), states)), dim=0)
            state = (h, c)
        else:
            state = torch.cat(states, dim=0)
        return x, state


class RNN_RES(nn.Module):
    def __init__(self, mode, input_size, hidden_size, num_layers=1, bias=True, batch_first=False,
                 droupout=0., bidirectional=False, res=True, nin=False, skip_step=0):
        super().__init__()

        num_directions = 2 if bidirectional else 1
        # output: [L, b, num_directions * d]
        # state: [num_layers * num_directions, b, d]
        self.rnn = [get_rnn(mode, input_size if i == 0 else num_directions * hidden_size,
                            hidden_size, 1, bias=bias, batch_first=batch_first,
                            droupout=droupout, bidirectional=bidirectional)
                    for i in range(num_layers)]
        self.rnn = nn.ModuleList(self.rnn)

        # TODO
        if nin:
            # 目前的实现与需求不一致
            # 在两层rnn之间加入kernel size=1的1D卷积 -> BN1D -> ReLU
            nin_layers = []
            for _ in range(num_layers - 1):
                nin_layers.extend([
                    nn.Linear(hidden_size * num_directions, hidden_size * num_directions),  # [l, b, d]
                    nn.BatchNorm1d(hidden_size * num_directions),
                    nn.ReLU()
                ])
            self.nin = nn.ModuleList(nin_layers)
        else:
            self.nin = None

        self.layers = num_layers
        self.num_directions = num_directions
        self.res = res
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.mode = mode
        self.skip_step = skip_step

        self.reset_parameters()

    def reset_parameters(self):
        init_rnn(self.rnn, self.mode)

        if self.nin is not None:
            for name, p in self.nin.named_parameters():
                if 'bias' in name:
                    nn.init.zeros_(p)
                elif 'weight' in name:
                    if p.dim() == 1:
                        nn.init.ones_(p)  # bn
                    elif p.dim() == 2:
                        nn.init.xavier_normal_(p)  # linear map
                    else:
                        raise Exception('weight dim more than 2')
                else:
                    raise Exception('unknown parameter name')

    def set_from_another_rnn(self, other):
        # TODO
        # weight_ih/hh_l0/1/2/...
        # bias_ih_hh_l0/1/2/...
        # reverse: append _reverse
        # ------------------------------
        # rnn.0/1/2.weight/bias_ih/hh_l0
        other_dict_params = dict(other.named_parameters())
        for i, rnn in enumerate(self.rnn):
            for name, p in rnn.named_parameters():
                lst = name.split('_')
                if lst[-1] == 'reverse':
                    lst[-2] = f"l{i}"
                else:
                    lst[-1] = f"l{i}"
                name = '_'.join(lst)
                p.data.copy_(other_dict_params[name])

    def forward(self, x, state=None):
        """
        :param x: PackedSequence or Tensor [l, b, d] or [b, l, d]
        :param state: None, Tensor or tuple
        :return:
        """
        is_pack = isinstance(x, nn.utils.rnn.PackedSequence)
        if is_pack:
            bsz = x.batch_sizes[0].item()
        else:
            bsz = x.size(0) if self.batch_first else x.size(1)

        if state is None:
            # output: num_layers * num_directions
            # view: (num_layers, num_directions, batch_size, hidden_size)
            if not is_pack:
                state = x.new_zeros((self.layers * self.num_directions, bsz, self.hidden_size),
                                    requires_grad=False)
            else:
                state = x.data.new_zeros((self.layers * self.num_directions, bsz, self.hidden_size),
                                         requires_grad=False)

            if self.mode == 'LSTM':
                state = (state, state)

        states = []
        for i, m in enumerate(self.rnn):
            if self.mode != 'LSTM':
                layer_state = state[i * self.num_directions: (i + 1) * self.num_directions]
            else:
                layer_state = (state[0][i * self.num_directions: (i + 1) * self.num_directions],
                               state[1][i * self.num_directions: (i + 1) * self.num_directions])

            # print(f"x type: {type(x)}\ntype state: {type(layer_state)} len(state): {len(layer_state)}\n"
            #       f"state shape: {layer_state[0].shape}, {layer_state[1].shape}\n"
            #       f"type(m): {type(m)}")
            y, cur_state = m(x, layer_state)

            # TODO
            # nin, input: y
            if (self.nin is not None) and (i < self.layers - 1):
                nin_linear_map = self.nin[i * 3]
                nin_bn = self.nin[i * 3 + 1]
                nin_act = self.nin[i * 3 + 2]
                if is_pack:
                    y = nin_linear_map(y.data)
                    # convert pack to padding: [b, d, l]
                    y, lens = nn.utils.rnn.pad_packed_sequence(
                        nn.utils.rnn.PackedSequence(y, x.batch_sizes),
                        True,
                    )  # [b, l, d]
                    y = y.transpose(1, 2)  # [b, d, l]
                    y = nin_bn(y)
                    y = nin_act(y)
                    # convert to pack
                    y = y.transpose(1, 2)  # [b, l, d]
                    y = nn.utils.rnn.pack_padded_sequence(y, lens, batch_first=True)  # pack
                else:
                    raise Exception('Currently do not support non-pack input!')

            states.append(cur_state)
            if self.res and (i > 0):
                if is_pack:
                    assert (x.batch_sizes == y.batch_sizes).all()
                    x = nn.utils.rnn.PackedSequence(data=x.data + y.data, batch_sizes=x.batch_sizes)
                else:
                    x = x + y
            else:
                x = y

            # skip step
            if self.skip_step > 0 and i < (len(self.rnn) - 1):
                """
                0 * * * *
                1 * * *
                2 * *
                batch_sizes: [3, 3, 2, 1]
                if skip step 为 2
                0 * *
                1 * *
                2 *
                batch_sizes: [3, 2]
                """
                if is_pack:
                    pad, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=False)  # [l, b, d]
                    pad = pad[::self.skip_step]
                    lens = lens // self.skip_step
                    lens[lens == 0] = 1  # 至少留1个，因此在lens为0处设为1
                    x = nn.utils.rnn.pack_padded_sequence(pad, lens)
                else:
                    if self.batch_first:
                        x = x[:, ::self.skip_step]
                    else:
                        x = x[::self.skip_step]

        if self.mode == 'LSTM':
            h = torch.cat(tuple(map(itemgetter(0), states)), dim=0)
            c = torch.cat(tuple(map(itemgetter(1), states)), dim=0)
            state = (h, c)
        else:
            state = torch.cat(states, dim=0)
        return x, state


class Conv1D(nn.Module):
    def __init__(self, in_channels, out_channels, ks, stride, act, norm, skip_connect):
        super().__init__()
        # 长度不够时，直接丢弃，比如输出长度100，stride=2，输出长度是(100-1)//2 + 1=99
        self.conv = nn.Conv1d(in_channels, out_channels, ks, stride)

        self.ks = ks
        self.stride = stride

        assert act in ('GLU', 'RELU', 'SIGMOID', 'TANH', 'NONE'), f"{act} is not GLU, RELU, SIGMOID, TANH or NONE"
        if act == 'GLU':
            self.act = partial(F.glu, dim=1)
        elif act == 'RELU':
            self.act = F.relu
        elif act == 'SIGMOID':
            self.act = torch.sigmoid
        elif act == 'TANH':
            self.act = torch.tanh
        else:
            self.act = None

        assert norm in ('NONE', 'BN', 'LN', 'IN'), "norm must be BN, LN or IN"
        if norm == 'BN':
            # input: [b, d, l]
            self.norm = nn.BatchNorm1d(out_channels, affine=True, track_running_stats=True)
            print(f'[BN]: affine={True}, track_running_states={True}')
        elif norm == 'LN':
            # input: [b, d, l] -> [b, l, d]
            self.norm = nn.LayerNorm(out_channels, elementwise_affine=True)
            print(f'[LN]: affine={True}')
        elif norm == 'IN':
            # input: [b, d, l]
            self.norm = nn.InstanceNorm1d(out_channels, affine=True, track_running_stats=False)
            print(f'[IN]: affine={True}, track_running_states={False}')
        elif norm == 'NONE':
            self.norm = None
        else:
            raise Exception("unknown norm type!")
        self.norm_type = norm

        self.skip_connect = skip_connect

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

        if self.norm:
            nn.init.ones_(self.norm.weight)
            nn.init.zeros_(self.norm.bias)

    def forward(self, x, lens=None):
        """
        order: conv -> norm -> activation -> skip_connect
        conv的原则是自动补全，不丢弃输入信息
        :param x: [b, d, l], channel first，如果有padding，则padding部分必须是0
        :param lens: None or [b] torch.int
        :param act: activation function
        :return: Conv1DOutput
        """

        bsz, _, l = x.size()

        # 1. pad
        x = pad(x, lens.max().item() if lens is not None else l, self.ks, self.stride)

        # 2. convolution
        y = self.conv(x)  # [b, d', l']

        # 3. norm
        if self.norm:
            if self.norm_type == 'ln':
                y = y.transpose(1, 2)  # [b, d', l'] -> [b, l', d']
            y = self.norm(y)
            if self.norm_type == 'ln':
                y = y.transpose(1, 2)

        # 4. activation
        if self.act:
            y = self.act(y)  # [b, d' / 2, l'] if activation is glu (gated linear unit)

        # 5. skip connect
        if self.skip_connect:
            assert x.size(1) == y.size(1)
            to_be_add = x[..., (self.ks - 1)::self.stride]
            y = y + to_be_add

        # 6. perform mask output
        if lens is not None:
            y_lens = (lens - self.ks + self.stride - 1) // self.stride + 1
            mask = get_mask(y_lens, y.size(-1)).t()  # [b, l']
            y = y * mask[:, None]  # [b, d', l']
        else:
            y_lens = None

        return ConvOutput(y, y_lens)


class MaskWidthCNN2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x, lens=None):
        """
        variable width
        :param x: [b, c, h, w]
        :param lens: [b], default None
        :return: [b, c', h', w']
        """

        y = self.conv(x)  # [b, c', h', w']

        if lens is not None:
            w_ks = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[1]
            w_stride = self.stride if isinstance(self.stride, int) else self.stride[1]
            w_pad = self.padding if isinstance(self.padding, int) else self.padding[1]
            mask = get_mask((lens + 2 * w_pad - w_ks) // w_stride + 1, y.size(-1)).t()  # [b, w']
            y = y * mask[:, None, None]

        return y


# class MaskWidthBN2D(nn.Module):
#     def __init__(self):



class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, ks, stride, act, norm, skip_connect):
        """
        :param in_channels: int
        :param out_channels: int
        :param ks: int or tuple
        :param stride: int or tuple
        """
        super().__init__()
        # 长度不够时，直接丢弃，比如输出长度100，stride=2，输出长度是(100-1)//2 + 1=99
        if norm != 'NONE':
            self.conv = nn.Conv2d(in_channels, out_channels, ks, stride, bias=False)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, ks, stride, bias=True)

        assert act in ('GLU', 'RELU', 'SIGMOID', 'TANH', 'NONE'), f"{act} is not GLU, RELU, SIGMOID, TANH or None"
        if act == 'GLU':
            self.act = partial(F.glu, dim=1)
        elif act == 'RELU':
            self.act = F.relu
        elif act == 'SIGMOID':
            self.act = torch.sigmoid
        elif act == 'TANH':
            self.act = torch.tanh
        else:
            self.act = None

        assert norm in ('NONE', 'BN', 'LN', 'IN'), "norm must be BN, LN, IN or NONE"
        if norm == 'BN':
            # input: [b, c, h, w]
            self.norm = nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True)
            print(f'[BN]: affine={True}, track_running_states={True}')
        elif norm == 'LN':
            # input: [b, c, h, w] -> [b, h, w, c]
            self.norm = nn.LayerNorm(out_channels, elementwise_affine=True)
            print(f'[LN]: affine={True}')
        elif norm == 'IN':
            # input: [b, c, h, w]
            self.norm = nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=False)
            print(f'[IN]: affine={True}, track_running_states={False}')
        elif norm == 'NONE':
            self.norm = None
        else:
            raise Exception("unknown norm type")
        self.norm_type = norm

        self.skip_connect = skip_connect

        ks = (ks, ks) if isinstance(ks, int) else ks
        stride = (stride, stride) if isinstance(stride, int) else stride
        self.ks = ks
        self.stride = stride

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        if self.norm:
            nn.init.ones_(self.norm.weight)
            nn.init.zeros_(self.norm.bias)

    def forward(self, x, lens=None, h_pad=None):
        """
        该conv自动pad，不丢弃输入信息，并且只有width维是可变长的
        :param x: [b, d, h, w]，对语音输入来说，就是[b, c, d, l]，其中d是frequency维，如果有padding，则padding部分必须是0
        :param lens: None or [b]（因为频率维同样长度，因此不需要padding，因此都是有效的长度）
        :return: ConvOutput
        """

        bsz, _, h, w = x.size()

        # 1. pad
        if h_pad is not None:
            h_pad = (h_pad, h_pad) if isinstance(h_pad, int) else h_pad
            x = F.pad(x, (0, 0, h_pad[0], h_pad[1]))
        x = pad2d(x, (h, lens.max().item()) if lens is not None else (h, w),
                  self.ks, self.stride)

        # 2. convolution
        y = self.conv(x)  # [b, d', h', w']

        # 3. norm
        if self.norm:
            # pad下的norm
            y = self.norm(y)

        # 4. activation
        if self.act:
            y = self.act(y)  # [b, d' / 2, h', w'] if activation is glu

        # 5. skip connect
        if self.skip_connect:
            # 这种skip connect有待商榷！
            to_be_add = x[..., (self.ks[0] - 1)::self.stride[0], (self.ks[1] - 1)::self.stride[1]]  # [b, d, h, w]
            y = y + to_be_add

        # 6. perform mask output
        if lens is not None:
            y_lens = (lens - self.ks[1] + self.stride[1] - 1) // self.stride[1] + 1
            mask = get_mask(y_lens, y.size(-1)).t()  # [b, w']
            y = y * mask[:, None, None]  # [b, d', h', w']
        else:
            y_lens = None

        return ConvOutput(y, y_lens)


class Duration(object):
    def __init__(self):
        self.cum_dur = 0.

    def tic(self):
        self.ts = time()

    def toc(self):
        te = time()
        duration = te - self.ts
        self.cum_dur += duration
        self.ts = te
        return duration


class Checkpoint(object):
    def __init__(self):
        """
        ckpt文件命名规则：iter-xxx_metric-xxx.ckpt，
        具体地，iter-xxx_wer-xxx.ckpt
        """
        pass

    def best_checkpoint(self, ckpt_dir, mode='min'):
        """
        :param ckpt_dir: absolute or relative path
        :return: latest checkpoint file name
        """
        assert mode in ('max', 'min')
        ckpts = os.listdir(ckpt_dir)
        fn = eval(mode)
        best_ckpt = None
        if ckpts:
            best_ckpt = fn(ckpts, key=lambda s: float(s.split('-')[-1][:-5]))
        return best_ckpt

    def latest_checkpoint(self, ckpt_dir):
        """
        :param ckpt_dir: absolute or relative path
        :return: latest checkpoint file name
        """
        ckpts = os.listdir(ckpt_dir)
        latest_ckpt = None
        if ckpts:
            latest_ckpt = max(ckpts, key=lambda s: int(s.split('_', 1)[0][5:]))
        return latest_ckpt


class RNNCellBase(nn.Module):
    def __init__(self, mode, input_size, hidden_size, bias=True, layers=1):
        super().__init__()

        assert mode in ('LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU')
        if mode == 'LSTM':
            self.cell = nn.ModuleList([nn.LSTMCell(input_size if i == 0 else hidden_size, hidden_size, bias)
                                       for i in range(layers)])
        elif mode == 'GRU':
            self.cell = nn.ModuleList([nn.GRUCell(input_size if i == 0 else hidden_size, hidden_size, bias)
                                       for i in range(layers)])
        elif mode == 'RNN_TANH':
            self.cell = nn.ModuleList([nn.RNNCell(input_size if i == 0 else hidden_size, hidden_size, bias, 'tanh')
                                       for i in range(layers)])
        else:
            self.cell = nn.ModuleList([nn.RNNCell(input_size if i == 0 else hidden_size, hidden_size, bias, 'relu')
                                       for i in range(layers)])

        self.mode = mode
        self.layers = layers

        self.reset_parameters()

    def reset_parameters(self):
        init_rnn(self.cell, self.mode)

    def forward(self, x, state=None):
        if state is not None:
            assert isinstance(state, list)
        else:
            state = [None] * self.layers

        states = []
        for i in range(self.layers):
            y = self.cell[i](x, state[i])
            x = y[0] if self.mode == 'LSTM' else y
            states.append(y)
        return states


class FFN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, act=F.relu):
        # feed forward network
        super().__init__()

        self.weight_1 = nn.Parameter(torch.randn(hidden_size, input_size))
        self.weight_2 = nn.Parameter(torch.randn(output_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size + output_size))

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.act = act

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight_1)
        nn.init.xavier_normal_(self.weight_2)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        :param x: [b, *, d]
        :return: [b, *, d']
        """
        y = self.act(F.linear(x, self.weight_1, self.bias[:self.hidden_size]))
        y = F.linear(y, self.weight_2, self.bias[self.hidden_size:])
        return y


class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, proj=False):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        self.bias = nn.Parameter(torch.zeros(3 * hidden_size))

        if proj:
            self.proj_weight = nn.Parameter(torch.randn(hidden_size, hidden_size))
        else:
            self.proj_weight = None

        self.hidden_size = hidden_size
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)
        if self.proj_weight is not None:
            nn.init.xavier_normal_(self.proj_weight)

    def forward(self, q, lens=None, heads=1):
        """
        :param q: [b, l, d]
        :param lens: default None
        :param heads: default int
        :return: [b, l, d]
        """
        head_size = self.hidden_size // heads

        scaling = head_size ** -0.5

        q = q * scaling

        q, k, v = F.linear(q, self.weight, self.bias).chunk(3, -1)

        attn, alignment = compute_self_attention(q, k, v, lens, heads, self.proj_weight)

        return attn, lens


class SelfLocalAttention(nn.Module):
    def __init__(self, input_size, hidden_size, proj=False):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        self.bias = nn.Parameter(torch.zeros(3 * hidden_size))

        if proj:
            self.proj_weight = nn.Parameter(torch.randn(hidden_size, hidden_size))
        else:
            self.proj_weight = None

        self.hidden_size = hidden_size
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)
        if self.proj_weight is not None:
            nn.init.xavier_normal_(self.proj_weight)

    def forward(self, ws, q, lens=None, heads=1):
        """
        :param ws: slide window size, int or None
        :param q: [b, l, d]
        :param lens: [b] or None
        :param heads: int
        :return: [b, l, d]
        """
        head_size = self.hidden_size // heads

        scaling = head_size ** -0.5

        q = q * scaling

        q, k, v = F.linear(q, self.weight, self.bias).chunk(3, -1)

        attn, alignment = compute_self_local_attention(q, k, v, lens, ws, heads, self.proj_weight)

        return attn, lens


class SelfAttentionBlock(nn.Module):
    def __init__(self, input_size, hidden_size, proj, ffn_size):
        # basic block for self attention encoder
        super().__init__()

        self.mha = SelfAttention(input_size, hidden_size, proj)
        self.ffn = FFN(hidden_size, ffn_size, hidden_size)

        # layer normalization
        self.ln_1 = nn.LayerNorm(hidden_size, elementwise_affine=True)
        self.ln_2 = nn.LayerNorm(hidden_size, elementwise_affine=True)

        self.reset_parameters()

    def reset_parameters(self):
        # mha and ffn has been initialized
        nn.init.ones_(self.ln_1.weight)
        nn.init.zeros_(self.ln_1.bias)

        nn.init.ones_(self.ln_2.weight)
        nn.init.zeros_(self.ln_2.bias)

    def forward(self, x, lens=None, heads=1):
        """
        self_attention -> layer_normalization -> ffn -> skip_connect -> layer_normalization
        :param x: [b, l, d]
        :param lens: default None
        :param heads: int
        :return: [b, l, d]
        """

        # 1. sublayer 1: self-attention + res + ln
        y, lens = self.mha(x, lens, heads)  # [b, l, d]
        if x.size(-1) == y.size(-1):
            y = x + y
        x = self.ln_1(y)  # [b, l, hidden_size]

        # 2. sublayer 2: ffn + res + ln
        y = self.ffn(x)  # [b, l, hidden_size]
        y = x + y
        y = self.ln_2(y)

        return y, lens


class SelfLocalAttentionBlock(nn.Module):
    def __init__(self, input_size, hidden_size, proj, ffn_size):
        # basic block for self local attention encoder
        super().__init__()

        self.sla = SelfLocalAttention(input_size, hidden_size, proj)
        self.ffn = FFN(hidden_size, ffn_size, hidden_size)

        # layer normalization
        self.ln_1 = nn.LayerNorm(hidden_size, elementwise_affine=True)
        self.ln_2 = nn.LayerNorm(hidden_size, elementwise_affine=True)

        self.reset_parameters()

    def reset_parameters(self):
        # mha and ffn has been initialized
        nn.init.ones_(self.ln_1.weight)
        nn.init.zeros_(self.ln_1.bias)

        nn.init.ones_(self.ln_2.weight)
        nn.init.zeros_(self.ln_2.bias)

    def forward(self, ws, x, lens=None, heads=1):
        """
        self_attention -> layer_normalization -> ffn -> skip_connect -> layer_normalization
        :param x: [b, l, d]
        :param lens: default None
        :param heads: int
        :return: [b, l, d]
        """

        # 1. sublayer 1
        y, lens = self.sla(ws, x, lens, heads)  # [b, l, d]
        if x.size(-1) == y.size(-1):
            y = x + y
        x = self.ln_1(y)  # [b, l, hidden_size]

        # 2. sublayer 2
        y = self.ffn(x)  # [b, l, hidden_size]
        y = x + y
        y = self.ln_2(y)

        return y, lens


# borrowed from https://github.com/pytorch/fairseq/blob/master/fairseq/modules/multihead_attention.py
class MultiheadAttention_fair(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None,
                 dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        if self.qkv_same_dim:
            self.in_proj_weight = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        else:
            self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))
            self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))

        if bias:
            self.in_proj_bias = nn.Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)
            nn.init.xavier_uniform_(self.q_proj_weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = None
        else:
            saved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(key)

        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)

            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.shape == torch.Size([]):
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if self.onnx_trace:
                attn_weights = torch.where(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    torch.Tensor([float("-Inf")]),
                    attn_weights.float()
                ).type_as(attn_weights)
            else:
                attn_weights = attn_weights.float().masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf'),
                ).type_as(attn_weights)  # FP16 support: cast to float and back
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if (self.onnx_trace and attn.size(1) == 1):
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_q(self, query):
        if self.qkv_same_dim:
            return self._in_proj(query, end=self.embed_dim)
        else:
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[:self.embed_dim]
            return F.linear(query, self.q_proj_weight, bias)

    def in_proj_k(self, key):
        if self.qkv_same_dim:
            return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)
        else:
            weight = self.k_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[self.embed_dim:2 * self.embed_dim]
            return F.linear(key, weight, bias)

    def in_proj_v(self, value):
        if self.qkv_same_dim:
            return self._in_proj(value, start=2 * self.embed_dim)
        else:
            weight = self.v_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[2 * self.embed_dim:]
            return F.linear(value, weight, bias)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)


# adabound borrowed from https://github.com/Luolc/AdaBound
class AdaBound(optim.Optimizer):
    """Implements AdaBound algorithm.
    It has been proposed in `Adaptive Gradient Methods with Dynamic Bound of Learning Rate`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Adam learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        final_lr (float, optional): final (SGD) learning rate (default: 0.1)
        gamma (float, optional): convergence speed of the bound functions (default: 1e-3)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsbound (boolean, optional): whether to use the AMSBound variant of this algorithm
    .. Adaptive Gradient Methods with Dynamic Bound of Learning Rate:
        https://openreview.net/forum?id=Bkg3g2R9FX
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3,
                 eps=1e-8, weight_decay=0, amsbound=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= final_lr:
            raise ValueError("Invalid final learning rate: {}".format(final_lr))
        if not 0.0 <= gamma < 1.0:
            raise ValueError("Invalid gamma parameter: {}".format(gamma))
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,
                        weight_decay=weight_decay, amsbound=amsbound)
        super(AdaBound, self).__init__(params, defaults)

        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))

    def __setstate__(self, state):
        super(AdaBound, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                amsbound = group['amsbound']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsbound:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsbound:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsbound:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Applies bounds on actual learning rate
                # lr_scheduler cannot affect final_lr, this is a workaround to apply lr decay
                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)

                p.data.add_(-step_size)

        return loss


class AdaBoundW(optim.Optimizer):
    """Implements AdaBound algorithm with Decoupled Weight Decay (arxiv.org/abs/1711.05101)
    It has been proposed in `Adaptive Gradient Methods with Dynamic Bound of Learning Rate`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Adam learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        final_lr (float, optional): final (SGD) learning rate (default: 0.1)
        gamma (float, optional): convergence speed of the bound functions (default: 1e-3)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsbound (boolean, optional): whether to use the AMSBound variant of this algorithm
    .. Adaptive Gradient Methods with Dynamic Bound of Learning Rate:
        https://openreview.net/forum?id=Bkg3g2R9FX
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3,
                 eps=1e-8, weight_decay=0, amsbound=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= final_lr:
            raise ValueError("Invalid final learning rate: {}".format(final_lr))
        if not 0.0 <= gamma < 1.0:
            raise ValueError("Invalid gamma parameter: {}".format(gamma))
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,
                        weight_decay=weight_decay, amsbound=amsbound)
        super(AdaBoundW, self).__init__(params, defaults)

        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))

    def __setstate__(self, state):
        super(AdaBoundW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                amsbound = group['amsbound']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsbound:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsbound:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsbound:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Applies bounds on actual learning rate
                # lr_scheduler cannot affect final_lr, this is a workaround to apply lr decay
                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)

                if group['weight_decay'] != 0:
                    decayed_weights = torch.mul(p.data, group['weight_decay'])
                    p.data.add_(-step_size)
                    p.data.sub_(decayed_weights)
                else:
                    p.data.add_(-step_size)

        return loss


class TrainVar(object):
    def __init__(self, step, loss, best_wer, lr, duration, num_no_imprv):
        self.step = step
        self.loss = loss
        self.best_wer = best_wer
        self.lr = lr
        self.duration = duration
        self.num_no_imprv = num_no_imprv


# TODO
class MWER(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


# EMA有很多用法：
# 1. 记录loss值
# 2. average 权重值，据说eval、infer时使用ema后的权重，model会更好，在有的时候
# 3. Training时候，计算gradients的ma，如果某个时刻gradients norm 的 “方差大于” ma，则discard，保证train的稳定
class EMA(object):
    def __init__(self, alpha):
        self.alpha = alpha
        self.shadow = {}

    def register(self, var_name, value=0):
        assert var_name not in self.shadow, f"{var_name} has been registerd"
        self.shadow[var_name] = value

    def update(self, var_name, value):
        # all weight sum: 1 - alpha ** n
        self.shadow[var_name] = (1 - self.alpha) * value + self.alpha * self.shadow[var_name]

    @staticmethod
    def get_ema_from_sequence(alpha, seq):
        out = [seq[0]]
        for v in seq:
            out.append((1 - alpha) * v + alpha * out[-1])
        return out[1:]


# --------------------- others ---------------------
TrainOutput = namedtuple('TrainOutput', ('loss', 'max_audio_len', 'max_text_len', 'grad_norm', 'alignment',
                                         'audio_feat_len', 'text_len'))
EvalOutput = namedtuple('EvalOutput', ('pred_text', 'score', 'text', 'wer', 'n', 'alignment',
                                       'audio_feat_len', 'text_len'))
EncoderOutput = namedtuple('EncoderOutput', ('out', 'out_lens', 'state'))
DecoderOutput = namedtuple('DecoderOutput', ('logit', 'attn_hidden_state', 'alignment', 'cell_state'))
ConvOutput = namedtuple('ConvOutput', ('out', 'out_lens'))
TrainInput = namedtuple('TrainInput', ('audio', 'audio_len', 'text_src', 'text_tgt', 'text_len'))
EvalInput = namedtuple('EvalInput', ('audio', 'audio_len', 'text'))
InferInput = namedtuple('InferInput', ('audio', 'audio_len'))


# test
def test_util():
    # pass
    # device = torch.device('cuda')
    # lens = torch.IntTensor([5, 2, 1, 3]).to(device)
    # max_len = 6
    # mask = get_mask(lens, max_len)
    # # mask = get_mask_for_softmax(lens, max_len)
    # print(mask)
    # print(mask.device)

    # from time import sleep
    # dur = Duration()
    # dur.tic()
    # sleep(1.234)
    # print(dur.toc(), dur.cum_dur)
    # sleep(4.8)
    # print(dur.toc(), dur.cum_dur)
    # sleep(3)
    # print(dur.toc(), dur.cum_dur)

    # ckpt_dir = '/home/share/anji/asr/code/SR_with_Pytorch/aishell-1/ckpt/enc_5'
    # ckpt = Checkpoint()
    # print(ckpt.latest_checkpoint(ckpt_dir))
    # print(ckpt.best_checkpoint(ckpt_dir, 'min'))

    # t = torch.randn(4, 3, 5)
    # # print(t)
    # t = tile_batch(t, 2, batch_first=False)
    # print(t.shape)

    # from Levenshtein import editops, distance
    # from timeit import timeit
    # src = 'shawn is a good man'
    # tgt = 'u r a very good person'
    #
    # # print(get_wer(src, tgt, False))
    # # print(distance(src, tgt))
    #
    # print(timeit("get_wer(src, tgt, False)",
    #              f"from __main__ import get_wer; src='{src}'; tgt='{tgt}'",
    #              number=10000))
    # # 3.7594034
    #
    # print(timeit("distance(src, tgt)",
    #              f"from Levenshtein import distance; src='{src}'; tgt='{tgt}'",
    #              number=10000))
    # # 0.007233
    # # 结论：我自己写的跟包没法比！ %>_<%

    # src = 'shawn is a good man'
    # tgt = 'u r a very good person'
    # # print(get_wer_python(src, tgt))
    # # print(get_wer(src, tgt, True, True))
    # print(timeit("get_wer_python(src, tgt, True)",
    #              f"from __main__ import get_wer_python; src='{src}'; tgt='{tgt}'",
    #              number=100000))  # 34.7
    # print(timeit("get_wer(src, tgt, True, True)",
    #              f"from __main__ import get_wer; src='{src}'; tgt='{tgt}'",
    #              number=100000))  # 0.427
    # print(timeit("distance(src, tgt)",
    #              f"from __main__ import distance; src='{src}'; tgt='{tgt}'",
    #              number=100000))  # 0.06

    def test_rnn_res():
        l, b, d = (11, 7, 2)
        # x = torch.randn(l, b, d)
        seq = [torch.randn(i, d) for i in [11, 10, 8, 7, 6, 5, 3]]
        x = nn.utils.rnn.pack_sequence(seq, True)

        torch.manual_seed(123)
        rnn = RNN_RES('LSTM', d, 4, 5, bidirectional=True, res=True)
        y, state = rnn(x)

        if isinstance(y, nn.utils.rnn.PackedSequence):
            y, _ = nn.utils.rnn.pad_packed_sequence(y)
        print(y.shape, get_shape(state))

        # torch.manual_seed(123)
        # rnn = LSTM_RES(d, 4, 5, bidirectional=True)
        # y2, state2 = rnn(x)
        #
        # assert (y == y2).all()
        # assert (state[0] == state2[0]).all()
        # assert (state[1] == state2[1]).all()

        # rnn = nn.RNNBase('RNN_TANH', 1, 3, 2, bidirectional=True)
        # for name, param in rnn.named_parameters():
        #     print(name)

    def test_localrnn():
        m = LocalRNN('GRU', 7, 11, 5, bidirectional=True, res=False, skip_step=[1, 1, 1, 1, 2])
        for p in m.parameters():
            nn.init.constant_(p, 1.231)

        x = torch.randn(13, 4, 7)  # [l, b, d]
        # 4, 2, 2, 2
        y, state = m(x)
        print(y.shape, state.shape)

        m2 = nn.GRU(input_size=7, hidden_size=11, num_layers=5, bidirectional=True)
        for p in m2.parameters():
            nn.init.constant_(p, 1.231)
        y2, state2 = m2(x)
        y2 = y2[::2]

        print((y - y2).abs().sum(), (state - state2).abs().sum())

    # test pad2d
    # x = torch.randn(1, 1, 38, 19)
    # y = pad2d(x, (38, 19), 7, 3)
    # print(y.shape)

    # test conv2d
    # conv2d = Conv2D(1, 18, (3, 3), (2, 2))
    # h, w = 32, 23
    # x = torch.randn(1, 1, h, w)
    # y = conv2d(x, act=nn.GLU(dim=1))
    # print(y.out.shape, y.out_lens, y.input_pad.shape)

    # test pad
    # x = torch.randn(2, 1, 20)
    # l = 20
    # ks, stride = 3, 3
    # y = pad(x, l, ks, stride)
    # print(y.shape)

    # get mask
    # x = get_mask_for_softmax(torch.tensor([3, 2, 1], dtype=torch.int,
    #                                       device=torch.device('cuda')))
    # print(x.device)

    # # test attention
    # m = SelfAttention(3, 4, 4, True)
    # x = torch.randn(4, 8, 3)  # [b, l, d]
    # lens = torch.IntTensor([4, 8, 6, 2])
    # # make sure input pad with zero
    # for i, l in enumerate(lens.tolist()):
    #     x[i, l:] = 0
    #
    # y = m(x, lens)
    # print(y[0].shape, y[1])  # [b, l, d]
    # print(y[0][:, :, 0])

    # test mask

    # d = 512
    # q, k, v = torch.randn(3, 13, d * 3).chunk(3, 2)
    #
    # ws = 12
    # # lens = None
    # lens = torch.tensor([2, 12, 13])
    # lens = torch.IntTensor([5, 10, 12])
    # lens = torch.tensor([6, 2, 3])
    # lens = torch.tensor([ws] * 3)

    # mask = LocalSelfAttention.get_mask_3d(x, lens, ws)
    # print(mask)

    # idx = LocalSelfAttention.get_single_mask(9, 5)
    # print(idx)

    # y = LocalSelfAttention.get_local_mask(q, k, v, lens, ws, 4)
    # print(y)

    # sa = SelfAttention(d, 2 * d, 8, True)
    # print(f"y shape: {y.shape}")

    # attn1, alignment1 = compute_self_attention(q[:2], k[:2], v[:2], lens[:2], 64)
    # attn2, alignment2 = compute_self_local_attention(q, k, v, lens, ws, 64)
    # attn2 = attn2[:2]
    #
    # print('attention:\n', f"shape: {attn1.shape}",
    #       (attn1 == attn2).all().item(), (attn1 - attn2).abs().max().item())
    # idx = torch.randperm(attn1.nelement())[:5]
    # print(attn1.view(-1)[idx])
    # print(attn2.view(-1)[idx])

    # print('alignment:\n', (alignment1 == alignment2).all().item(), (alignment1 - alignment2).abs().sum().item())
    # idx = torch.randperm(alignment1.nelement())[:5]
    # print(alignment1.view(-1)[idx])
    # print(alignment2.view(-1)[idx])

    # x, y = compute_self_local_attention(q, k, v, lens, ws, 1)
    # print(x.shape, y.shape)

    def test_init_rnn():
        rnn = RNN_RES('gru', 2, 3, 1, bidirectional=True)

        # for k, v in rnn.named_parameters():
        #     if 'bias' in k:
        #         print(k, v)

        l = 13
        b = 24

        # x = torch.randn(l, b, 2)
        # y, state = rnn(x)
        # print(y.shape, get_shape(state))

        x = [torch.randn(l, 2) for l in [12, 11, 10]]
        x = nn.utils.rnn.pack_sequence(x)
        y, state = rnn(x)
        print(get_shape(state))

    def test_conv1d():
        d = 2
        m = Conv1D(d, 4, 5, 2, 'glu', 'in', True)
        # for k, v in m.named_parameters():
        #     print(k, v.shape, v)

        b = 2
        l = 13
        x = torch.randn(b, d, l)
        y = m(x, torch.tensor([8, 10]))
        print(y[0].shape, y[1])
        print(y[0][:, 0])

    def test_conv2d():
        c = 2
        m = Conv2D(c, 4, 5, 2, 'glu', 'in', True)
        # for k, v in m.named_parameters():
        #     print(k, v.shape, v)

        b = 2
        h = 14
        w = 15
        x = torch.randn(b, c, h, w)
        y = m(x, torch.tensor([13, 8]))
        print(y[0].shape, y[1])
        print(y[0][:, 0])

    def test_self_local_attention():
        q = torch.randn(1, 4, 128)
        k = torch.randn(1, 4, 128)
        v = torch.randn(1, 4, 128)
        ws = 3
        lens = None

        # plain compute
        alignment = [
            q[0, 0: 1].mm(k[0, :ws, :].t()),
            q[0, 1: 2].mm(k[0, :ws, :].t()),
            q[0, 2: 3].mm(k[0, 1: 1 + ws, :].t()),
            q[0, 3: 4].mm(k[0, 1: 1 + ws, :].t()),
        ]
        alignment = torch.cat(alignment, 0)  # [4, 3]
        alignment = F.softmax(alignment, -1)
        attn = [
            alignment[0][0] * v[0, 0] + alignment[0][1] * v[0, 1] + alignment[0][2] * v[0, 2],
            alignment[1][0] * v[0, 0] + alignment[1][1] * v[0, 1] + alignment[1][2] * v[0, 2],
            alignment[2][0] * v[0, 1] + alignment[2][1] * v[0, 2] + alignment[2][2] * v[0, 3],
            alignment[3][0] * v[0, 1] + alignment[3][1] * v[0, 2] + alignment[3][2] * v[0, 3],
        ]
        attn = torch.stack(attn, 0)

        #
        attn2, _ = compute_self_local_attention(q, k, v, lens, ws, 1)

        print((attn == attn2).all())
        print(f"error: {(attn - attn2).abs().sum()}")

    def test_rnncellbase():
        cell = RNNCellBase('LSTM', 2, 3, True, 12)
        # for k, v in cell.named_parameters():
            # print(k, v.shape)
            # if 'bias' in k:
            #     print(v)

        x = torch.randn(15, 2)
        states = cell(x, [(torch.randn(15, 6).chunk(2, 1)) for _ in range(12)])
        for ele in states:
            print(get_shape(ele))

    def test_label_smoothing():
        b = 1434
        k = 3000
        l = torch.randn(b, k)
        targets = torch.randint(k, size=(b,))
        v = .0
        ls = label_smoothing(l, targets, v)
        # ls = label_smoothing_old(l, targets, v)

        def raw_label_smoothing(logits, targets, ls):
            logp = F.log_softmax(logits, 1)  # [b, k]
            target_logp = logp.gather(1, targets[:, None])[:, 0]  # [b]
            other_logp_sum = logp.sum(1) - target_logp

            loss = (1 - ls) * target_logp + ls / (logits.size(1)-1) * other_logp_sum
            return -loss

        def raw_label_smoothing_old(logits, targets, ls):
            k = logits.size(1)
            logp = F.log_softmax(logits, 1)  # [b, k]
            target_logp = logp.gather(1, targets[:, None])[:, 0]  # [b]
            other_logp_sum = logp.sum(1) - target_logp

            loss = (1 - ls * (k-1) / k) * target_logp + ls / k * other_logp_sum
            return -loss

        ls2 = raw_label_smoothing(l, targets, v)
        # ls2 = raw_label_smoothing_old(l, targets, v)
        ce = nn.CrossEntropyLoss(reduction='none')(l, targets)
        print((ls == ls2).all())
        print((ls - ls2).abs().sum())

        print((ls == ce).all())
        print((ls - ce).abs().sum())

    def test_get_steps():
        print(get_steps(-100000, 939))

    def test_get_conv_l():
        print(get_conv_length(23, [(3, 2)] * 3))

    def test_maskwidthcnn2d():
        m = MaskWidthCNN2D(1, 1, 3, 2)

        x = torch.randn(3, 1, 13, 17)
        lens = torch.tensor([17, 9, 13])  # 8, 3, 6

        y = m(x, lens)
        print(f"y shape: {y.shape}")
        print(f"y:\n{y[:, 0, 0, :]}")

        m2 = Conv2D(1, 1, 3, 2, 'NONE', 'NONE', False)
        m2.conv.weight = m.conv.weight
        m2.conv.bias = m.conv.bias

        y2, _ = m2(x, lens)

        print((y == y2).all())

    def test_convlstm():
        torch.manual_seed(123)
        m = ConvLSTM(1, 1, 3)

        x = torch.randn(3, 1, 4, 9)  # [b, c, h, w]
        # lens = torch.tensor([9, 5, 7])
        lens = torch.tensor([9, 9, 9])
        y, y_lens, h_c = m(x, lens)

        # print(f"y shape: {y.shape}, y_lens: {y_lens}, h_c shape: {get_shape(h_c)}")
        # print(f"y:\n{y[:, 0, 0]}")

        torch.manual_seed(123)
        m2 = ConvLSTM(1, 1, 3)
        y2, _, h_c2 = m2(x)

        print((y == y2).all())
        print((h_c[0] == h_c2[0]).all())
        print((h_c[1] == h_c2[1]).all())

        # test gradient
        y.backward(torch.ones_like(y))
        g = [ele.grad for ele in m.parameters()]

        y2.backward(torch.ones_like(y2))
        g2 = [ele.grad for ele in m2.parameters()]

        # print(g)
        # print(g2)
        print(all([(ele1 == ele2).all().item() for ele1, ele2 in zip(g, g2)]))

    def test_bconvlstm():
        m = BConvLSTM(1, 1, 3)
        x = torch.randn(3, 1, 4, 9)  # [b, c, h, w]
        lens = torch.tensor([9, 9, 9])

        y, y_lens, h_c = m(x, lens)

        print(f"y shape: {y.shape}, y_lens: {y_lens}, h_c shape: {get_shape(h_c)}")
        # print(f"y:\n{y[:, 0, 0]}")

    def test_sample_batch_idx():
        x = sample_batch_idx([1, 2, 3], nb=2)
        print(x)

    def test_alignment():
        def gen_sth():
            l_x = 6
            l_y = 4
            n = 5
            audio_len = torch.tensor([5, 3, 4])
            alignment = [torch.randn(l_x, 3, n) for _ in range(l_y)]
            alignment = [F.softmax(ele + get_mask_for_softmax(audio_len, l_x)[..., None], 0) for ele in alignment]
            text_len = torch.tensor([3, 1, 2])
            return alignment, audio_len, text_len

        # alignment, audio_len, text_len = gen_sth()
        # ali, idx = parse_batch_alignment(alignment, audio_len, text_len)
        # print(f"idx: {idx}")
        # for ele in ali:
        #     print(f"shape: {ele.shape}")

        # multi batch
        alignment = []
        audio_len = []
        text_len = []

        for _ in range(10):
            x, y, z = gen_sth()
            alignment.append(x)
            audio_len.append(y)
            text_len.append(z)

        ali, idx = parse_multi_batch_alignment(alignment, audio_len, text_len, 10)
        print(f"idx: {idx}")
        for ele in ali:
            print(f"shape: {ele.shape}")

    def test_round_down():
        print(round_down(99.99, 2))
        print(round_down(99.99, 1))
        print(round_down(99.994, 2))
        print(round_down(99.996, 2))
        print(round_down(0.9999, 2))

    def test_view_ckpt():
        view_ckpt('../all_data/ckpt/base_adam/step-162560_wer-1.4166.ckpt')

    def test_lstm():
        m = nn.LSTM(input_size=128, hidden_size=256, bias=False, num_layers=5, bidirectional=True)
        for p in m.parameters():
            nn.init.constant_(p, 1.2345)

        m2 = LSTM(128, 256, bias=False, num_layers=5, bidirectional=True)
        for p in m2.parameters():
            nn.init.constant_(p, 1.2345)

        m3 = RNN_RES('LSTM', 128, 256, 5, False, bidirectional=True)
        for p in m3.parameters():
            nn.init.constant_(p, 1.2345)

        x = torch.randn(13, 4, 128)

        # y, state = m(x)
        # y2, state2 = m2(x)

        # # test forward: success!
        # print('y error:', (y - y2).abs().sum())
        # print('h error:', (state[0] - state2[0]).abs().sum())
        # print('c error:', (state[1] - state2[1]).abs().sum())
        #
        # # test forward: success!
        # y.sum().backward()
        # y2.sum().backward()
        #
        # for p, p2 in zip(m.parameters(), m2.parameters()):
        #     print('grad error:', (p.grad.t() - p2.grad).abs().sum())
        #     # print(p.shape, p2.shape)

        # cmp speed
        m3.to('cuda')
        x = x.to('cuda')
        h0 = x.new_zeros((5 * 2, x.size(1), 256))
        h0 = (h0, h0)

        ts = time()
        for _ in range(500):
            # _ = m(x, h0)  # cpu: 2.493s [50]; gpu: 1.572s [500]
            _ = m3(x, h0)  # cpu: 1.566s [50]; gpu: 12.248s [500]
            # fastrnns: cpu: 2.324s [50]; gpu: 7.353s [500]
        print(f'run 500 times, time cost {time()-ts:.3f}s')


    # test_localrnn()
    # test_rnn_res()
    # test_view_ckpt()
    # test_round_down()
    # test_alignment()
    # test_sample_batch_idx()
    # test_bconvlstm()
    # test_convlstm()
    # test_maskwidthcnn2d()
    # test_get_steps()
    # test_get_conv_l()
    # test_init_rnn()
    # test_conv1d()
    # test_conv2d()
    # test_self_local_attention()
    # test_rnncellbase()
    # test_label_smoothing()
    test_lstm()

    # embed = get_sin_pos_embedding(torch.tensor([3, 2, 4]).to('cuda'), 5)
    # print(embed[:, 0])
    # print(embed[:, 1])
    # print(embed[:, 2])

if __name__ == '__main__':
    test_util()
