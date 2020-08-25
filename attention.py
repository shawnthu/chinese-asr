from gpd import *
from encoder import *

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
2D Attention:
考虑encoder特征是[b, c, h, w]，状态是[b, c, h]

decoder: ConvRNN
query: [b, c]
key: []
value:
"""


class BauAttn(nn.Module):
    attn_size = gpd['attn_size']

    def __init__(self, enc_size):
        super().__init__()

        attn_size = self.attn_size
        hidden_size = gpd['decoder_hidden_size']

        self.W_enc = nn.Parameter(data=torch.randn(enc_size, attn_size))
        self.b_attn = nn.Parameter(data=torch.zeros(attn_size))
        self.W_hidden = nn.Parameter(data=torch.randn(hidden_size, attn_size))
        self.v = nn.Parameter(data=torch.randn(attn_size))

        self.map_enc = None
        if gpd['map_enc']:
            self.map_enc = nn.Linear(enc_size, attn_size, False)

        self.context_size = enc_size if not gpd['map_enc'] else attn_size

        self.linear_map = None
        if gpd['heads'] > 1:
            assert attn_size % gpd['heads'] == 0, \
                f"attention size {attn_size} must be divided by num heads {gpd['heads']}"
            assert self.context_size % gpd['heads'] == 0, \
                f"context size {self.context_size} must be divided by num heads {gpd['heads']}"
            if gpd['linear_map']:
                self.linear_map = nn.Parameter(
                    data=torch.randn(self.context_size, self.context_size))
                print('[INFO] If multi heads, linear map after concat!')

        self.reset_parameters()

    def reset_parameters(self):
        # weights
        nn.init.xavier_normal_(self.W_enc)
        nn.init.xavier_normal_(self.W_hidden)
        nn.init.normal_(self.v, 0., .1)
        if self.linear_map is not None:
            nn.init.xavier_normal_(self.linear_map)

        if self.map_enc is not None:
            nn.init.xavier_normal_(self.map_enc.weight)

        # biases
        nn.init.zeros_(self.b_attn)

    def compute_key_value(self, enc_outputs):
        """
        compute the attention keys using encoder feature
        :param enc_outputs: [b, b, d], padding in general
        :return: [l, b, d]
        """
        if self.map_enc is not None:
            values = self.map_enc(enc_outputs)
        else:
            values = enc_outputs
        keys = torch.matmul(enc_outputs, self.W_enc) + self.b_attn
        return keys, values

    def forward(self, enc_outputs, mask, hidden_state, keys=None, values=None):
        # enc_outputs: [l, b, d]，一般是padded
        # lens: [b]
        # hidden_state: [b, D]，一般是padded
        # mask: [l, b]
        # keys: [l, b, d]，一般是padded
        l, b, _ = enc_outputs.size()

        if keys is None:
            keys, values = self.compute_key_value(enc_outputs)

        if gpd['heads'] == 1:
            alignment = (torch.tanh(keys + torch.mm(hidden_state, self.W_hidden)) * self.v).sum(dim=2)  # [l, b]
            # print('alignment sum:', alignment.sum().item(), alignment.shape)
            alignment = F.softmax(mask + alignment, dim=0)  # [l, b]
            context = (alignment[..., None] * values).sum(dim=0)  # [b, d]

        else:
            alignment = torch.tanh(keys + torch.mm(hidden_state, self.W_hidden)) * self.v  # [l, b, d]
            alignment = alignment.view(l, b, gpd['heads'], -1)  # [l, b, n, d // n]
            alignment = alignment.sum(3)  # [l, b, n]
            alignment = F.softmax(mask[..., None].expand(-1, -1, gpd['heads']) + alignment, dim=0)  # [l, b, n]
            # reshape encoder outputs
            values = values.view(l, b, gpd['heads'], -1)  # [l, b, n, d // n]
            context = (alignment[..., None] * values).sum(dim=0)  # [b, n, d // n]
            # concat context
            context = context.view(b, -1)  # [b, d]
            if self.linear_map is not None:
                # linear map
                context = context.mm(self.linear_map)  # [b, d]

        return context, alignment


def test_attn():
    l = 100
    b = 4
    enc_size = 512
    gpd['heads'] = 4
    gpd['map_enc'] = True
    gpd['decoder_hidden_size'] = 256

    # prepare model
    m = BauAttn(enc_size)
    for p in m.parameters():
        nn.init.ones_(p)

    # prepare data
    enc_outputs = torch.ones(l, b, enc_size)
    # lens = torch.tensor([90, 100, 88, 91])
    lens = torch.tensor([100] * b)
    hidden_state = torch.ones(b, gpd['decoder_hidden_size'])
    mask = get_mask_for_softmax(lens)

    context, alignment = m(enc_outputs, mask, hidden_state, None, None)
    print(context.sum())  # 2048.0002
    # print(f"context shape: {context.shape}\nalignment shape: {alignment.shape}")



if __name__ == '__main__':
    test_attn()