from gpd import *
from util import *
from encoder import *
from attention import *

import torch.nn as nn


# for rnn encoder
class RNNDecoder(nn.Module):
    vocab_size = gpd['max_num_words']
    real_vcb_sz = vocab_size + 4
    hidden_size = gpd['decoder_hidden_size']
    embed_dim = gpd['embed_dim']
    layers = gpd['decoder_num_layers']
    dec_type = gpd['decoder_type']

    def __init__(self, attn_mechanism):
        # input: attention mechanism
        super().__init__()

        context_size = attn_mechanism.context_size

        if gpd['attn_type'] == 'L':
            self.input_size = self.embed_dim + gpd['attn_hidden_size'] if gpd['input_feeding'] \
                else self.embed_dim
        else:
            self.input_size = self.embed_dim + context_size

        self.embedding = nn.Embedding(self.real_vcb_sz, self.embed_dim, padding_idx=gpd['pad'])

        self.cell = RNNCellBase(self.dec_type, self.input_size, self.hidden_size, True, self.layers)
        if gpd['dec_init_cell_state_as_param']:
            num_state = 2 if self.dec_type == 'LSTM' else 1
            chunks = self.layers * num_state
            self.dec_init_cell_state = nn.ParameterList(
                [nn.Parameter(torch.empty(self.hidden_size)) for _ in range(chunks)]
            )
        else:
            self.dec_init_cell_state = None

        if gpd['attn_type'] == 'L':
            # Luo提出：tanh(w[h; c])
            self.attn_hidden_weight = nn.Parameter(
                torch.randn(self.hidden_size + context_size, gpd['attn_hidden_size']))
            self.proj_linear = nn.Linear(gpd['attn_hidden_size'], self.real_vcb_sz, bias=False)
        else:
            # Bau式attention
            self.attn_hidden_weight = None
            # self.proj_linear = nn.Linear(context_size, self.real_vcb_sz)  # only-context
            self.proj_linear = nn.Linear(self.hidden_size + context_size, self.real_vcb_sz)  # only-context-version-2

        self.attn_mechanism = attn_mechanism
        self.reset_parameters()

    def get_initial_state(self, bsz, enc_state):
        # 优先使用encoder state
        if enc_state is not None:
            return [enc_state] * self.layers

        # 其次如果decoder初始状态可训，否则返回None
        if self.dec_init_cell_state is not None:
            if self.dec_type != 'LSTM':
                return [ele.expand(bsz, -1) for ele in self.dec_init_cell_state]
            else:
                out = []
                for i in range(self.layers):
                    out.append(
                        (self.dec_init_cell_state[2 * i].expand(bsz, -1),
                         self.dec_init_cell_state[2 * i + 1].expand(bsz, -1)))
                return out

        return None

    def reset_parameters(self):
        # 1. embedding
        nn.init.normal_(self.embedding.weight, 0., .1)
        # nn.init.xavier_normal_(self.embedding.weight)

        # 2. cell has been initialized

        # 3. if initial cell state trainable
        if self.dec_init_cell_state is not None:
            for ele in self.dec_init_cell_state:
                nn.init.zeros_(ele)

        # 4. attention hidden weight
        if self.attn_hidden_weight:
            nn.init.xavier_normal_(self.attn_hidden_weight)

        # 5. linear map for softmax
        nn.init.xavier_normal_(self.proj_linear.weight)

    def forward(self, enc_outputs, mask, keys, values, token, cell_state=None, attn_hidden_state=None,
                compute_logit=True):
        # enc_outputs: [l, b, d], paddde in general
        # mask: [l, b]
        # keys: [l, b, d']
        # tokes: [b]，一般是padded
        # attn_hidden_state: [b, d]
        # return: logit [b, v], h_c

        # 1. prepare cell input and cell state
        x = self.embedding(token)  # [b, d]
        if gpd['input_feeding']:
            if attn_hidden_state is None:
                if gpd['attn_type'] == 'L':
                    attn_hidden_state = x.new_zeros(x.size(0), gpd['attn_hidden_size'])
                else:
                    attn_hidden_state = x.new_zeros(x.size(0), self.attn_mechanism.context_size)
            x = torch.cat((x, attn_hidden_state), dim=1)

        # 2. cell computation
        cell_state = self.cell(x, cell_state)
        last_layer_h = cell_state[-1][0] if self.dec_type == 'LSTM' else cell_state[-1]

        # 3. calculate context
        context, alignment = self.attn_mechanism(enc_outputs, mask, last_layer_h, keys, values)  # [b, d]
        # print('context sum:', context.sum().item(), context.shape)

        # 4. calculate attentional hidden state
        if self.attn_hidden_weight:
            attn_hidden_state = torch.tanh(
                torch.cat((last_layer_h, context), dim=1).mm(self.attn_hidden_weight))  # [b, d]
        else:
            # attn_hidden_state = torch.cat((last_layer_h, context), dim=1)
            attn_hidden_state = context

        logit = None
        if compute_logit:
            if gpd['attn_type'] == 'B':
                # print('h:', last_layer_h.mean().item(), last_layer_h.std().item(), last_layer_h.sum().item())
                logit = self.proj_linear(torch.cat([last_layer_h, attn_hidden_state], -1))
            else:
                logit = self.proj_linear(attn_hidden_state)

        return DecoderOutput(logit, attn_hidden_state, alignment, cell_state)
