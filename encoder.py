from gpd import *
from util import *

import torch.nn as nn

from functools import partial


class RNNEncoder(nn.Module):
    # input_size = gpd['n_mels'] * (3 if gpd['downsample'] else 1) * \
    #              (3 if gpd['delta_delta'] else 1)
    # num_directions = 2 if gpd['encoder_bidirectional'] else 1
    # hidden_size = gpd['encoder_hidden_size']
    # num_layers = gpd['encoder_num_layers']
    # enc_size = hidden_size * num_directions

    def __init__(self,
                 enc_type=gpd['encoder_type'],
                 input_size=gpd['n_mels'] * (3 if gpd['downsample'] else 1) * (3 if gpd['delta_delta'] else 1),
                 bidirectional=gpd['encoder_bidirectional'],
                 hidden_size=gpd['encoder_hidden_size'],
                 num_layers=gpd['encoder_num_layers'],
                 res=gpd['residual'], skip_step=gpd['skip_step']
                 ):
        super().__init__()

        self.rnn = RNN_RES(enc_type, input_size, hidden_size, num_layers,
                           bias=True, batch_first=False, droupout=0., bidirectional=bidirectional,
                           res=res, skip_step=skip_step
                           )

        num_directions = 2 if bidirectional else 1
        self.enc_size = hidden_size * num_directions
        self.num_directions = num_directions

    def forward(self, x, lens):
        """
        :param x: list of tensor, length equals to batch_size
        :param lens: [b]
        :return: (enc_outputs, out_lens, cell_state),  [l, b, d], [b], tuple
        """

        device = x[0].device
        bsz = len(x)

        # lens = torch.IntTensor([ele.size(0) for ele in x], device=x.device)
        idx = lens.argsort(descending=True)
        seq = [x[i] for i in idx]
        recover_idx = torch.empty(bsz, dtype=torch.long, device=device)
        recover_idx[idx] = torch.arange(bsz, dtype=torch.long, device=device)

        # print('seq:', get_shape(seq), [(ele.sum(), ele.mean(), ele.std()) for ele in seq])
        pack = nn.utils.rnn.pack_sequence(seq)  # 'PackedSequence' object has no attribute 'device'
        # print('pack:', pack.data.shape, pack.data.sum(), pack.data.mean(), pack.data.std())

        # output: [l, b, d * directions]
        # h, c: [layers * directions, b, d]
        state = None
        output, state = self.rnn(pack, state)  # output: packed
        # torch.save(list(self.named_parameters()), '/data/fw.pt')
        # print(output.data.shape, output.data.sum(), output.data.mean(), output.data.std())

        pad_output, out_lens = \
            nn.utils.rnn.pad_packed_sequence(output, batch_first=False, padding_value=0.)  # [l, b, d]

        pad_output = pad_output[:, recover_idx]
        if isinstance(state, (tuple, list)):
            # just keep last layer of state
            state = tuple([
                ele[(-self.num_directions):,
                recover_idx].transpose(0, 1).contiguous().view(bsz, -1) for ele in state
            ])

        else:
            # just keep last layer of state
            # [layers * dirs, b, d] -> [dirs, b, d] -> [b, dirs, d] -> [b, dirs * d]
            state = state[(-self.num_directions):, recover_idx].transpose(0, 1).contiguous().view(bsz, -1)

        # return EncoderOutput(pad_output, out_lens.to(lens.device, lens.dtype), state)  # !!!!就是这一步的问题！！！
        # print('enc:', pad_output.sum(), pad_output.mean(), pad_output.std())
        return EncoderOutput(pad_output, lens, state)
        # return EncoderOutput(pad_output, out_lens.to(lens.device, lens.dtype), None)  # state is None


class CNN1DRNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn1d = CNN1DEncoder(oc=256, ks=3, stride=2, norm='BN', act='RELU', skip_connect=False, layers=2)
        # self.rnn = RNNEncoder(enc_type='LSTM', input_size=self.cnn1d.enc_size)
        self.rnn = RNNEncoder(enc_type='GRU', input_size=self.cnn1d.enc_size)

        self.enc_size = self.rnn.enc_size

    def forward(self, x, lens=None):
        x, lens, _ = self.cnn1d(x, lens)  # [l, b, d]
        x = x.transpose(0, 1)
        out = self.rnn(x, lens)
        return out


class CNN1DEncoder(nn.Module):
    # input_size = gpd['n_mels'] * (3 if gpd['downsample'] else 1) * \
    #              (3 if gpd['delta_delta'] else 1)
    # enc_size = gpd['encoder_hidden_size'] // 2 if gpd['act'] == 'GLU' else gpd['encoder_hidden_size']
    # norm = gpd['norm']
    # act = gpd['act']
    # skip_connect = gpd['residual']
    # layers = gpd['encoder_num_layers']

    def __init__(self,
                 input_size=gpd['n_mels'] * (3 if gpd['downsample'] else 1) * (3 if gpd['delta_delta'] else 1),
                 oc=gpd['encoder_hidden_size'], ks=gpd['ks'], stride=gpd['stride'], norm=gpd['norm'],
                 act=gpd['act'], skip_connect=gpd['residual'], layers=gpd['encoder_num_layers']):
        super().__init__()

        oc_k_s = list(zip([oc] * layers if isinstance(oc, int) else oc,
                          [ks] * layers if isinstance(ks, int) else ks,
                          [stride] * layers if isinstance(stride, int) else stride,
                          )
                      )
        self.convs = nn.ModuleList([Conv1D(input_size, *oc_k_s[0], act, norm, False)])
        for i in range(1, len(oc_k_s)):
            self.convs.append(
                Conv1D(oc_k_s[i-1][0] if act != 'GLU' else oc_k_s[i - 1][0] // 2,
                       *oc_k_s[i], act, norm, skip_connect))

        self.enc_size = oc_k_s[-1][0] // 2 if act == 'GLU' else oc_k_s[-1][0]

    def forward(self, x, lens=None):
        """
        conv -> norm -> activation -> skip_connect
        :param x: [b, d, l]，可能是padding，保证padding部分为0
        :param lens: None or [b], torch.int
        :return: (y, y_lens), [l, b, d], [b]
        """

        for conv in self.convs:
            x, lens = conv(x, lens)

        # [b, d, l] -> [l, b, d]
        x = x.permute(2, 0, 1)

        return EncoderOutput(x, lens, None)


class CNN2DEncoder(nn.Module):
    input_size = (3 if gpd['downsample'] else 1) * (3 if gpd['delta_delta'] else 1)
    # [b, c, h, w] -> [w, b, c * h]
    act = gpd['act']
    norm = gpd['norm']
    skip_connect = gpd['residual']
    layers = gpd['encoder_num_layers']

    def __init__(self):
        super().__init__()

        oc = gpd['encoder_hidden_size']
        ks = gpd['ks']
        stride = gpd['stride']
        oc_k_s = list(zip([oc] * self.layers if not isinstance(oc, list) else oc,
                          [ks] * self.layers if not isinstance(ks, list) else ks,
                          [stride] * self.layers if not isinstance(stride, list) else stride,
                          )
                      )

        # [b, c, h, w] -> [b, c * h, w]
        self.enc_size = (gpd['encoder_hidden_size'] // 2 if gpd['act'] == 'GLU' else gpd['encoder_hidden_size']) * \
                        get_conv_length(gpd['n_mels'], [(ele[1][0], ele[2][0]) for ele in oc_k_s])

        self.convs = nn.ModuleList([Conv2D(self.input_size, *oc_k_s[0], self.act, self.norm, False)])
        for i in range(1, len(oc_k_s)):
            self.convs.append(Conv2D(oc_k_s[i-1][0] if gpd['act'] != 'GLU' else oc_k_s[i - 1][0] // 2,
                                     *oc_k_s[i],
                                     self.act, self.norm, self.skip_connect,
                                     )
                              )

    def forward(self, x, lens=None):
        """
        :param x: [b, d, h, w]，可能是padding，保证padding部分为0
        :param lens: None or [b], torch.int
        :return: (y, y_lens), [l, b, d], [b]
        """

        for conv in self.convs:
            x, lens = conv(x, lens)
        # [b, d, h, w] -> [b, d * h, w] -> [l, b, d]
        x = x.view(x.size(0), -1, x.size(-1)).permute(2, 0, 1)
        return EncoderOutput(x, lens, None)


class SelfAttentionEncoder(nn.Module):
    # input_size = gpd['n_mels'] * (3 if gpd['downsample'] else 1) * \
    #              (3 if gpd['delta_delta'] else 1)
    # hidden_size = gpd['encoder_hidden_size']
    # heads = gpd['self_attn_heads']
    # proj = gpd['mha_proj']
    # ffn_size = gpd['ffn_size']
    # layers = gpd['encoder_num_layers']
    # enc_size = gpd['encoder_hidden_size']

    def __init__(self,
                 input_size=gpd['n_mels'] * (3 if gpd['downsample'] else 1) * (3 if gpd['delta_delta'] else 1),
                 hidden_size=gpd['encoder_hidden_size'],
                 heads=gpd['self_attn_heads'], proj=gpd['mha_proj'], ffn_size=gpd['ffn_size'],
                 layers=gpd['encoder_num_layers'],
                 ):
        super().__init__()

        blocks = nn.ModuleList([])
        for i in range(layers):
            blocks.append(
                SelfAttentionBlock(
                    input_size if i == 0 else hidden_size, hidden_size, proj, ffn_size)
            )

        self.heads = heads
        self.blocks = blocks
        self.enc_size = hidden_size

    def forward(self, x, lens=None, pos=False):
        """
        :param x: [b, l, d]
        :param lens: None or [b]
        :return: y [l, b, d']
        """
        if pos:
            pos_emb = get_sin_pos_embedding(x.shape[:2].to(x.device), x.shape[-1])
            x = x + pos_emb
        for i in range(self.layers):
            x, lens = self.blocks[i](x, lens, self.heads)
        x = x.transpose(0, 1)  # [l, b, d']
        return EncoderOutput(x, lens, None)


class CNN1DSelfAttnEncoder(nn.Module):
    # 不要看gpd！！！！
    def __init__(self):
        super().__init__()

        # [l, b, d]
        self.cnn1d = CNN1DEncoder(oc=256, ks=3, stride=2, norm='BN', act='RELU', skip_connect=False, layers=2)
        self.sa = SelfAttentionEncoder(input_size=self.cnn1d.enc_size, hidden_size=256, heads=4, proj=True,
                                       ffn_size=512, layers=4)

    def forward(self, x, lens=None):
        x, lens, _ = self.cnn1d(x, lens)  # [l, b, d]
        x = x.transpose(0, 1)
        out = self.sa(x, lens, True)
        return out


class SelfLocalAttentionEncoder(nn.Module):
    input_size = gpd['n_mels'] * (3 if gpd['downsample'] else 1) * \
                 (3 if gpd['delta_delta'] else 1)
    hidden_size = gpd['encoder_hidden_size']
    heads = gpd['self_attn_heads']
    proj = gpd['mha_proj']
    ffn_size = gpd['ffn_size']
    layers = gpd['encoder_num_layers']
    enc_size = gpd['encoder_hidden_size']

    def __init__(self):
        super().__init__()

        blocks = nn.ModuleList([])
        for i in range(self.layers):
            blocks.append(
                SelfLocalAttentionBlock(
                    self.input_size if i == 0 else self.hidden_size,
                    self.hidden_size, self.proj, self.ffn_size)
            )

        self.blocks = blocks

    def forward(self, ws, x, lens=None):
        """
        :param ws: int
        :param x: [b, l, d]
        :param lens: default None
        :return: y [l, b, d]
        """
        for i in range(self.layers):
            x, lens = self.blocks[i](ws, x, lens, self.heads)
        x = x.transpose(0, 1)  # [l, b, d]
        return EncoderOutput(x, lens, None)


class CRNNEncoder(nn.Module):
    input_size = (3 if gpd['downsample'] else 1) * (3 if gpd['delta_delta'] else 1)
    height = gpd['n_mels']
    num_directions = 1

    def __init__(self, out_channels, rnn_hidden_size, rnn_layers):
        super().__init__()

        print('[INFO] Encoder: CRNN')

        self.heads = nn.ModuleList([Conv2D(self.input_size, out_channels, 3, (1, 2), 'NONE', 'BN', False),
                                    Conv2D(out_channels, out_channels, 3, (1, 2), 'NONE', 'BN', False)])

        rnn_input_size = out_channels * self.height

        self.rnn = RNN_RES('GRU', rnn_input_size, rnn_hidden_size, rnn_layers,
                           bias=True, batch_first=False,
                           bidirectional=False, droupout=0., res=False, nin=True)

        self.conv_lstm = nn.ModuleList(
            [ConvLSTM(out_channels, out_channels, 3) for _ in range(3)]
        )

        # self.enc_size = rnn_hidden_size
        self.enc_size = out_channels * self.height

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x, lens):
        # 先两层卷积，再来rnn

        for m in self.heads:
            x, lens = m(x, lens, 1)  # [b, c, h, w] h_pad=1

        # convlstm
        for m in self.conv_lstm:
            x, lens, _ = m(x, lens)

        # [b, c, w, h] ->  [b, d, l] -> [l, b, d]
        x = x.view(x.size(0), -1, x.size(-1)).contiguous().permute(2, 0, 1)

        # x = x.view(x.size(0), -1, x.size(-1))  # [b, d, l]
        # # convert to list
        # x = [t[:, :l].t() for t, l in zip(x.unbind(0), lens)]  # list of [l, d]

        # # copy from above code
        # device = x[0].device
        # bsz = len(x)
        #
        # idx = lens.argsort(descending=True)
        # seq = [x[i] for i in idx]
        # recover_idx = torch.empty(bsz, dtype=torch.long, device=device)
        # recover_idx[idx] = torch.arange(bsz, dtype=torch.long, device=device)
        #
        # pack = nn.utils.rnn.pack_sequence(seq)  # 'PackedSequence' object has no attribute 'device'
        #
        # # output: [l, b, d * directions]
        # # h, c: [layers * directions, b, d]
        # state = None
        # output, state = self.rnn(pack, state)  # output: packed
        #
        # pad_output, out_lens = \
        #     nn.utils.rnn.pad_packed_sequence(output, batch_first=False, padding_value=0.)  # [l, b, d]
        #
        # pad_output = pad_output[:, recover_idx]
        # if isinstance(state, (tuple, list)):
        #     # just keep last layer of state
        #     state = tuple([
        #         ele[(-self.num_directions):,
        #         recover_idx].transpose(0, 1).contiguous().view(bsz, -1) for ele in state
        #     ])
        #
        # else:
        #     # just keep last layer of state
        #     # [layers * dirs, b, d] -> [dirs, b, d] -> [b, dirs, d] -> [b, dirs * d]
        #     state = state[(-self.num_directions):, recover_idx].transpose(0, 1).contiguous().view(bsz, -1)

        # return EncoderOutput(pad_output, out_lens.to(lens.device, lens.dtype), state)
        return EncoderOutput(x, lens, None)


class DCNNEncoder(nn.Module):
    # implementation of Very deep convolutional networks for end-to-end speech recognition
    def __init__(self, in_channels, out_channels, n_head=2, n_middle=4):
        super().__init__()

        # head layers
        self.heads = [Conv2D(in_channels, out_channels, 3, (1, 2), 'NONE', 'BN')]
        for _ in range(n_head - 1):
            self.heads.append(Conv2D(out_channels, out_channels, 3, (1, 2), 'NONE', 'BN'))

        # middle layers
        self.middle = [ResConvLSTM(out_channels, out_channels) for _ in range(n_middle)]

        # tail layers
        self.nin = NIN(out_channels, out_channels, 3)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x, lens=None):
        # x: [b, c, h, w]
        # lens: [b], default None

        # 1. first two cnn2d layer, kernel_size=(3, 3), stride = 2 to reduce time
        for m in self.heads:
            x, lens = m(x, lens)

        # 2. ResConvLSTM * 4
        for m in self.middle:
            x, lens = m(x, lens)

        # 3. NIN module
        return


class ResCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        # x -> cnn2d -> bn -> relu -> cnn2d -> bn -> y + x -> relu
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)  # keep the same h, w
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)  # keep the input/output same shape
        self.bn2 = nn.BatchNorm2d(out_channels)

        if out_channels != in_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, 1, 1)
        else:
            self.downsample = None

        self.reset_parameters()

    def reset_parameters(self):
        # weight
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)

        # bias
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.bn1.bias)
        nn.init.zeros_(self.bn2.bias)

        # bn weight
        nn.init.ones_(self.bn1.weight)
        nn.init.ones_(self.bn2.weight)

        # downsample
        if self.downsample is not None:
            nn.init.xavier_normal_(self.downsample.weight)
            nn.init.zeros_(self.downsample.bias)

    def forward(self, x, lens=None):
        """
        only w can be variable length
        :param x: [b, c, h, w]
        :param lens: [b], def None
        :return:
        """
        y = self.conv1(x)  # [b, c', h, w]

        if lens is not None:
            mask = get_mask(lens, x.size(-1)).t()  # [b, w]
            w_mask = lambda t: t * mask[:, None, None]
        else:
            w_mask = lambda t: t

        y = w_mask(y)
        y = self.bn1(y)  # BN under padding
        y = F.relu(y)
        y = w_mask(y)  # mask before conv, enforce the padding to be 0

        y = self.conv2(y)
        y = w_mask(y)  # BN under padding
        y = self.bn2(y)

        if self.downsample is not None:
            x = self.downsample(x)
        y = x + y
        y = F.relu(y)

        y = w_mask(y)
        return y, lens


class ResConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        # 这里的normalization还待商榷！
        super().__init__()

        self.conv_lstm1 = BConvLSTM(in_channels, out_channels, kernel_size)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv_lstm2 = BConvLSTM(out_channels * 2, out_channels, kernel_size)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if out_channels != in_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, 1, 1)
        else:
            self.downsample = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.bn1.weight)
        nn.init.zeros_(self.bn1.bias)

        nn.init.ones_(self.bn2.weight)
        nn.init.zeros_(self.bn2.bias)

        if self.downsample is not None:
            nn.init.xavier_normal_(self.downsample.weight)
            nn.init.zeros_(self.downsample.bias)

    def forward(self, x, lens=None):
        """
        variable width
        :param x: [b, c, h, w]
        :param lens: [b], def None
        :return:
        """

        y, _, _ = self.conv_lstm1(x, lens)  # [b, 2 * c', h, w]
        y = self.bn1(y)
        y = F.relu(y)

        if lens is not None:
            mask = get_mask(lens, x.size(-1)).t()  # [b, w]
            w_mask = lambda t: t * mask[:, None, None]
        else:
            w_mask = lambda t: t

        y = w_mask(y)

        y, _, _ = self.conv_lstm2(y, lens)  # [b, 2 * c', h, w]
        y = self.bn2(y)

        # skip connect
        if self.downsample is not None:
            x = self.downsample(x)  # [b, c', h, w]
        y = x + y
        y = F.relu(y)

        y = w_mask(y)

        return y, lens


class NIN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.conv_lstm1 = BConvLSTM(in_channels, out_channels, kernel_size)
        self.conv1 = Conv2D(out_channels * 2, out_channels, 1, 1, 'NONE', 'BN', False)

        self.conv_lstm2 = BConvLSTM(out_channels, out_channels, kernel_size)
        self.conv1 = Conv2D(out_channels * 2, out_channels, 1, 1, 'NONE', 'BN', False)

        self.conv_lstm = BConvLSTM(out_channels, out_channels, kernel_size)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x, lens=None):
        """
        :param x: [b, c, h, w]
        :param lens: [b], def None
        :return:
        """
        # L -> C(1 * 1) -> BN -> Relu
        # -> L -> C(1 * 1) -> BN -> Relu
        # -> L

        x, lens, _ = self.conv_lstm1(x, lens)  # [b, 2, c', h, w]
        x = x.view(x.size(0), -1, x.size(3), x.size(4))  # [b, 2 * c', h, w]
        x, lens = self.conv1(x, lens)
        x = F.relu(x)

        x, lens, _ = self.conv_lstm2(x, lens)
        x = x.view(x.size(0), -1, x.size(3), x.siez(4))
        x, lens = self.conv2(x, lens)
        x = F.relu(x)

        x, lens, state = self.conv_lstm(x, lens)
        x = x.view(x.size(0), -1, x.size(4))  # -> [b, 2, c', h , w] -> [w, 2 * c' * h, b]
        state = (state[0].view(state[0].size(0), -1, state[0].size(-1)),
                 state[1].view(state[1].size(0), -1, state[1].size(-1))
                 ) # [b, 2, c, h] -> [b, 2 * c, h]
        return EncoderOutput(x, lens, state)


def test_encoder():
    # test rnn encoder
    # gpd['encoder_type'] = 'lstm'
    # d = RNNEncoder.input_size
    # lens = torch.IntTensor([6, 2, 3])
    # x = [torch.randn(l, d) for l in lens]
    #
    # torch.manual_seed(123)
    # m = RNNEncoder()
    # y = m(x, lens)
    # print(y.out.shape, y.out_lens, y.state[0].shape, y.state[1].shape)
    #
    # torch.manual_seed(123)
    # m = RNNEncoder_old()
    # y2 = m(x, lens)
    #
    # assert ((y.out - y2.out).abs().sum() < 1e-7).all()
    # assert (y.state[0] - y2.state[0]).abs().sum() < 1e-7

    # test cnn1d and cnn2d encoder
    # x = torch.randn(4, CNN2DEncoder.input_size, 28)  # [b, d, l]
    # lens = None
    # lens = torch.IntTensor([21, 21, 13, 17])  # [7, 7, 5, 6]
    # print('x shape:', x.shape)
    #
    # gpd['out_channels'] = 16
    # gpd['ks'] = 3
    # gpd['stride'] = 3
    # gpd['act'] = 'glu'
    # gpd['encoder_num_layers'] = 1
    # gpd['residual'] = True
    #
    # m = CNN2DEncoder()
    # x = x.unsqueeze(dim=1)  # [b, 1, d, l] -> [b, d, h, w]

    # m = CNN1DEncoder()

    # y = m(x, lens)
    # print(y[0].shape, y[1])
    #
    # print(y[0][:, :, 0])
    # assert (y[0][7:, 0, :] == 0).all()
    # assert (y[0][7:, 1, :] == 0).all()
    # assert (y[0][5:, 2, :] == 0).all()
    # assert (y[0][6:, 3, :] == 0).all()
    # test cnn1d encoder

    def test_rnn():
        gpd['n_mels'] = 80
        gpd['encoder_bidirectional'] = True
        gpd['encoder_hidden_size'] = 256
        gpd['encoder_num_layers'] = 4
        gpd['residual'] = True
        gpd['encoder_type'] = 'LSTM'

        m = RNNEncoder()
        for p in m.parameters():
            nn.init.ones_(p)

        lens = [10, 8, 23, 14]
        x = [torch.ones(l, m.input_size) for l in lens]
        y = m(x, torch.tensor(lens))
        # print(y[0].shape, y[1], get_shape(y[2]))
        print(y[0].sum(), y[2][0].sum(), y[2][1].sum())  # 110345.5000, 2048, 28160


    def test_cnn1d():
        gpd['n_mels'] = 80
        gpd['out_channels'] = 64
        gpd['ks'] = 5
        gpd['stride'] = 3
        gpd['encoder_num_layers'] = 1
        gpd['norm'] = 'bn'
        gpd['act'] = 'glu'
        gpd['residual'] = True

        m = CNN1DEncoder()

        b = 2
        l = 13
        x = torch.randn(b, m.input_size, l)
        lens = torch.tensor([8, 12])
        y = m(x, lens)
        print(y[0].shape, y[1])

    def test_cnn2d():
        gpd['n_mels'] = 80
        gpd['out_channels'] = 64
        gpd['ks'] = 5
        gpd['stride'] = 3
        gpd['encoder_num_layers'] = 2
        gpd['norm'] = 'bn'
        gpd['act'] = 'glu'
        gpd['residual'] = True

        m = CNN2DEncoder()

        b = 2
        h = 24  # 1 + (24 - 5 + 2) // 3 = 8
        w = 15  # 1 + (15 - 5 + 2) // 3 = 5
        x = torch.randn(b, m.input_size, h, w)
        lens = torch.tensor([8, 12])  # 2, 4
        y = m(x, lens)  # [b, c, h, w]
        print(y[0].shape, y[1])

    def test_self_attention():
        gpd['encoder_hidden_size'] = 256
        gpd['encoder_num_layers'] = 4
        gpd['ffn_size'] = 1024
        gpd['heads'] = 1

        m = SelfAttentionEncoder()

        b = 2
        l = 15
        x = torch.randn(b, l, m.input_size)
        # lens = None
        lens = torch.tensor([8, 12])
        y = m(x, lens)
        print(y[0][..., 0])
        print(y[0].shape, y[1])

    def test_self_local_attention():
        gpd['encoder_hidden_size'] = 256
        gpd['encoder_num_layers'] = 4
        gpd['ffn_size'] = 1024
        gpd['heads'] = 1

        m = SelfLocalAttentionEncoder()

        ws = 4
        b = 2
        l = 15
        x = torch.randn(b, l, m.input_size)
        lens = None
        # lens = torch.tensor([8, 12])
        y = m(ws, x, lens)
        print(y[0][..., 0])
        print(y[0].shape, y[1])

    def test_rescnn():
        x = torch.randn(3, 13, 44, 15)
        lens = torch.tensor([15, 6, 12])
        m = ResCNN(13, 25)

        y = m(x, lens)
        print(y.shape)
        print(y[:, 0, 0])

    def test_cnn1d_sa():
        m = CNN1DRNNEncoder()
        print(gpd['encoder_hidden_size'])
        print(m.enc_size)


    # test_cnn1d()
    # test_cnn2d()
    # test_self_attention()
    # test_self_local_attention()
    # test_rescnn()
    # test_rnn()
    test_cnn1d_sa()

if __name__ == '__main__':
    test_encoder()