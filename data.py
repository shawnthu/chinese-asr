from gpd import *
from util import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler, Dataset, DataLoader
# import torchaudio

import numpy as np
from scipy.signal import convolve
from soundfile import read as wav_read

import random
from time import time
from collections import defaultdict
from operator import itemgetter
import pickle


def create_fb_matrix(n_stft, f_min, f_max, n_mels):
    # type: (int, float, float, int) -> Tensor
    """ Create a frequency bin conversion matrix.

    Inputs:
        n_stft (int): number of filter banks from spectrogram
        f_min (float): minimum frequency
        f_max (float): maximum frequency
        n_mels (int): number of mel bins

    Outputs:
        Tensor: triangular filter banks (fb matrix)
    """
    def _hertz_to_mel(f):
        # type: (float) -> Tensor
        return 2595. * torch.log10(torch.tensor(1.) + (f / 700.))

    def _mel_to_hertz(mel):
        # type: (Tensor) -> Tensor
        return 700. * (10**(mel / 2595.) - 1.)

    # get stft freq bins
    stft_freqs = torch.linspace(f_min, f_max, n_stft)
    # calculate mel freq bins
    m_min = 0. if f_min == 0 else _hertz_to_mel(f_min)
    m_max = _hertz_to_mel(f_max)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hertz(m_pts)
    # calculate the difference between each mel point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels + 1)
    slopes = f_pts.unsqueeze(0) - stft_freqs.unsqueeze(1)  # (n_stft, n_mels + 2)
    # create overlapping triangles
    z = torch.tensor(0.)
    down_slopes = (-1. * slopes[:, :-2]) / f_diff[:-1]  # (n_stft, n_mels)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_stft, n_mels)
    fb = torch.max(z, torch.min(down_slopes, up_slopes))
    return fb


def mel_scale(spec_f, f_min, f_max, n_mels, fb=None):
    # type: (Tensor, float, float, int, Optional[Tensor]) -> Tuple[Tensor, Tensor]
    """ This turns a normal STFT into a mel frequency STFT, using a conversion
    matrix.  This uses triangular filter banks.

    Inputs:
        spec_f (Tensor): normal STFT
        f_min (float): minimum frequency
        f_max (float): maximum frequency
        n_mels (int): number of mel bins
        fb (Optional[Tensor]): triangular filter banks (fb matrix)

    Outputs:
        Tuple[Tensor, Tensor]: triangular filter banks (fb matrix) and mel frequency STFT
    """
    if fb is None:
        fb = create_fb_matrix(spec_f.size(2), f_min, f_max, n_mels).to(spec_f.device)
    else:
        # need to ensure same device for dot product
        fb = fb.to(spec_f.device)
    spec_m = torch.matmul(spec_f, fb)  # (c, l, n_fft) dot (n_fft, n_mels) -> (c, l, n_mels)
    return fb, spec_m


class MelScale(object):
    """This turns a normal STFT into a mel frequency STFT, using a conversion
       matrix.  This uses triangular filter banks.

    Args:
        n_mels (int): number of mel bins
        sr (int): sample rate of audio signal
        f_max (float, optional): maximum frequency. default: `sr` // 2
        f_min (float): minimum frequency. default: 0
        n_stft (int, optional): number of filter banks from stft. Calculated from first input
            if `None` is given.  See `n_fft` in `Spectrogram`.
    """
    def __init__(self, n_mels=128, sr=16000, f_max=None, f_min=0., n_stft=None):
        self.n_mels = n_mels
        self.sr = sr
        self.f_max = f_max if f_max is not None else sr // 2
        self.f_min = f_min
        self.fb = create_fb_matrix(
            n_stft, self.f_min, self.f_max, self.n_mels) if n_stft is not None else n_stft

    def __call__(self, spec_f):
        self.fb, spec_m = mel_scale(spec_f, self.f_min, self.f_max, self.n_mels, self.fb)
        return spec_m


def fast_read(path):
    # rate, data = wavfile.read(path)
    data, rate = wav_read(path, dtype='float32')
    # if rate != gpd['sample_rate']:
    #     f = tempfile.NamedTemporaryFile(suffix='.wav')
    #     os.system(f"sox {path} -r {gpd['sample_rate']} {f.name}")
    #     data, rate = wav_read(f.name, dtype='float32')
    #     # os.remove(f.name)
    # if rate != gpd['sample_rate'] or data.dtype is not np.dtype(f"int{gpd['bit_depth']}"):
    #     print(f'[WARN] rate={rate}, dtype={data.dtype}, path={path}')
    if rate != gpd['sample_rate']:
        print(f'[WARN] rate={rate}, dtype={data.dtype}, path={path}')
    return data


def read_from_hdf5(hdf5_dst, i):
    return hdf5_dst[i]


# borrowed from https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_audio.py
def add_delta_deltas(filterbanks):
    """Compute time first and second-order derivative channels.
      Args:
        filterbanks: float32 tensor with shape [batch_size, 1, len, num_bins]  # in pytorch, NCHW
        name: scope name
      Returns:
        float32 tensor with shape [batch_size, len, num_bins, 3]
      """
    delta_filter = np.array([2, 1, 0, -1, -2])
    delta_delta_filter = convolve(delta_filter, delta_filter, "full")

    delta_filter_stack = np.array(
        [[0] * 4 + [1] + [0] * 4, [0] * 2 + list(delta_filter) + [0] * 2,
         list(delta_delta_filter)],
        dtype=np.float32).T[:, None, None, :]

    # [h, w, in, out] for tf
    delta_filter_stack /= np.sqrt(
        np.sum(delta_filter_stack ** 2, axis=0, keepdims=True))

    # [out, in, h, w] for pytorch
    delta_filter_stack = np.transpose(delta_filter_stack, axes=(3, 2, 0, 1))
    # [9, 1, 1, 3] => [3, 1, 9, 1]

    # NCHW
    h, w = delta_filter_stack.shape[2:]
    wl = (w - 1) // 2
    hl = (h - 1) // 2
    filterbanks = F.pad(filterbanks,
                        pad=(wl, w - 1 - wl, hl, h - 1 -hl),
                        mode='constant', value=0.)
    filterbanks = F.conv2d(filterbanks,
                           torch.from_numpy(delta_filter_stack),
                           bias=None, stride=1, padding=0)

    return filterbanks


def get_log_mel(training, file_path, ms, window, data_aug=False):
# def get_log_mel(hdf5_dst, idx, ms, window, data_aug=False):
    audio = fast_read(file_path)  # int16, 1D in general
    # audio = read_from_hdf5(hdf5_dst, idx)
    # if audio.dtype is np.dtype(f"int{gpd['bit_depth']}"):
    #     audio = audio.astype('float32') / 2 ** (gpd['bit_depth'] - 1)

    # audio = audio.astype('float32') / 2 ** (gpd['bit_depth'] - 1)

    # if audio.dtype is not np.dtype('float32'):
    #     audio = audio.astype('float32') / 2 ** (gpd['bit_depth'] - 1)

    # data augmentation
    if data_aug:
        aug_prob = gpd['aug_prob']

        # volume perturbation
        if random.random() < aug_prob:
            audio = gain_db(audio,
                            random.uniform(gpd['volume_gain_min'], gpd['volume_gain_max'])
                            )
        # speed perturbation
        if random.random() < aug_prob:
            audio = change_speed(audio,
                                 random.uniform(gpd['speed_rate_min'], gpd['speed_rate_max'])
                                 )
        # shift perturbation
        if random.random() < aug_prob:
            audio = shift(audio,
                          random.uniform(gpd['shift_ms_min'], gpd['shift_ms_max']),
                          gpd['sample_rate'])

    if gpd['dither'] > 0. and training:
        audio += np.random.normal(0., gpd['dither'], size=audio.shape)
    if gpd['preemphasis'] > 0.:
        audio = audio[1:] - gpd['preemphasis'] * audio[:-1]

    audio = torch.tensor(audio).view(1, -1)  # [1, l], l必须大于等于512，否则RuntimeError
    spec_f = torch.stft(audio, n_fft=512,
                        hop_length=int(gpd['sample_rate'] * gpd['window_step']),
                        win_length=int(gpd['sample_rate'] * gpd['window_len']),
                        window=window,
                        center=False, pad_mode='reflect', normalized=False, onesided=True)  # [1, 257, l, 2]
    # try:
    #     spec_f = torch.stft(audio, n_fft=512,
    #                         hop_length=int(gpd['sample_rate'] * gpd['window_step']),
    #                         win_length=int(gpd['sample_rate'] * gpd['window_len']),
    #                         window=window,
    #                         center=False, pad_mode='reflect', normalized=False, onesided=True)  # [1, 257, l, 2]
    # except:
    #     print('wav_path:', file_path, audio.shape)
    #     sys.exit()
    # print('audio shape:', audio.shape, 'spec_f shape:', spec_f.shape)
    spec_f.transpose_(1, 2)  # [1, l, 257, 2]
    feature = spec_f.pow(2).sum(-1)  # [1, l, 257]
    feature = ms(feature)  # [1, l, 80]
    feature.masked_fill_(feature == 0., torch.finfo(torch.float32).eps)
    feature = torch.log(feature[0])  # [l, 80]

    if gpd['delta_delta']:
        # [1, 1, l, 80]
        feature = add_delta_deltas(feature[None, None])  # [1, 3, l, 80]
        feature = feature.squeeze(0)  # [3, l, d]

    if gpd['downsample']:
        if feature.ndimension() == 2:
            # [l, d]
            feature = feature[:(3 * (feature.size(0) // 3))]  # [l, d]
            if gpd['encoder_type'] != 'CNN2D':
                # [l, d]
                feature = feature.view(feature.size(0) // 3, -1)  # [l // 3, 3 * d]
            else:
                # [l, 3, d] 将长度l维放第一个维度
                feature = feature.view(feature.size(0) // 3, 3, -1)  # [l // 3, 3, d]

        elif feature.ndimension() == 3:
            # [3, l, d]
            feature = feature[:, :(3 * (feature.size(1) // 3))]
            if gpd['encoder_type'] != 'CNN2D':
                # output shape [l, d]
                feature = feature.view(feature.size(0), feature.size(1) // 3, -1)  # [3, l // 3, 3 * d]
                feature = feature.transpose(0, 1)  # [l // 3, 3, 3 * d]
                feature = feature.contiguous().view(feature.size(0), -1)  # [l // 3, 9 * d]
            else:
                # output shape [l, 9, d]
                feature = feature.view(feature.size(0), feature.size(1) // 3, 3, -1)  # [3, l // 3, 3, d]
                feature = feature.transpose(0, 1).contiguous().view(-1, 9, gpd['n_mels'])  # [l // 3, 9, d]
        else:
            raise Exception("audio feature dims must be 2 or 3")
    else:
        if feature.dim() == 2:
            # [l, d]
            if gpd['encoder_type'] == 'CNN2D':
                feature = feature.unsqueeze(1)  # [l, 1, d]
        elif feature.dim() == 3:
            # [3, l, d]
            if gpd['encoder_type'] == 'CNN2D':
                feature = feature.transpose(0, 1)  # [l, 3, d]
            else:
                feature = feature.transpose(0, 1).contiguous().view(feature.size(1), -1)  # [l, 3 * d]
        else:
            raise Exception("audio feature dims must be 2 or 3")

    # 特征的四种shape
    # 1. [l, d]
    # 2. delta [3, l, d]
    # 3. downsample: [l // 3, 3, d]
    # 4. delta and downsample: [3, l // 3, 3, d]

    # encoder input shape
    # 1. rnn, cnn1d [l, d]
    # 2. cnn2d, [l, c, d]

    return feature


def gain_db(sample, gain):
    """Apply gain in decibels to samples.
    Note that this is an in-place transformation.

    :param gain: Gain in decibels to apply to samples.
    :type gain: float|1darray
    """
    dtype = sample.dtype
    sample *= 10. ** (gain / 20.)
    return sample.astype(dtype)
    # sample[:] = sample * 10. ** (gain / 20.)
    # return 0


def change_speed(sample, speed_rate):
    """Change the audio speed by linear interpolation.
    Note that this is an in-place transformation.

    :param speed_rate: Rate of speed change:
                       speed_rate > 1.0, speed up the audio;
                       speed_rate = 1.0, unchanged;
                       speed_rate < 1.0, slow down the audio;
                       speed_rate <= 0.0, not allowed, raise ValueError.
    :type speed_rate: float
    :raises ValueError: If speed_rate <= 0.0.
    """
    dtype = sample.dtype
    if speed_rate <= 0:
        raise ValueError("speed_rate should be greater than zero.")
    old_length = sample.shape[0]
    new_length = int(old_length / speed_rate)
    old_indices = np.arange(old_length)
    new_indices = np.linspace(start=0, stop=old_length, num=new_length)
    sample = np.interp(new_indices, old_indices, sample)
    return sample.astype(dtype)


def shift(sample, shift_ms, sample_rate=16000):
    """Shift the audio in time. If `shift_ms` is positive, shift with time
    advance; if negative, shift with time delay. Silence are padded to
    keep the duration unchanged.
    Note that this is an in-place transformation.
    :param shift_ms: Shift time in millseconds. If positive, shift with
                     time advance; if negative; shift with time delay.
    :type shift_ms: float
    :raises ValueError: If shift_ms is longer than audio duration.
    """
    # if abs(shift_ms) / 1000 * sample_rate > sample.shape[0]:
    #     raise ValueError("Absolute value of shift_ms should be smaller "
    #                      "than audio duration.")
    dtype = sample.dtype
    shift_sample = int(shift_ms * sample_rate / 1000)
    if shift_sample > 0:
        # time advance
        sample[:-shift_sample] = sample[shift_sample:]
        sample[-shift_sample:] = 0
    elif shift_sample < 0:
        # time delay
        sample[-shift_sample:] = sample[:shift_sample]
        sample[:-shift_sample] = 0
    return sample.astype(dtype)


class TrainSampler(Sampler):
    def __init__(self, text_list, bsz):
        # 先shuffle，设一个shuffle updates，每隔这么多updates排序一下
        self.text_list = text_list
        self.buffer_size = gpd['shuffle_updates'] * bsz

    def __len__(self):
        return len(self.text_list)

    def __iter__(self):
        # 该sampler用于train
        indices = torch.randperm(len(self.text_list)).tolist()
        # indices = list(range(len(self.text_list)))

        start_idx = 0
        while start_idx < len(self):
            cand_indices = indices[start_idx: start_idx + self.buffer_size]
            cand_indices = sorted(cand_indices, key=lambda i: len(self.text_list[i]), reverse=True)
            for idx in cand_indices:
                # print('idx:', idx)
                yield idx
            start_idx += self.buffer_size


# AudioBase用于提供基本的数据
class AudioBase(object):
    def __init__(self):
        with open('dict.pkl', 'rb') as f:
            self.word2int, self.int2word = pickle.load(f)

        # self.ms = torchaudio.transforms.MelScale(n_mels=gpd['n_mels'], sr=gpd['sample_rate'],
        #                                          f_max=7600, f_min=80, n_stft=257)
        self.ms = MelScale(n_mels=gpd['n_mels'], sr=gpd['sample_rate'],
                           f_max=7600, f_min=80, n_stft=257)

        winfunc = torch.hann_window
        self.window = winfunc(int(gpd['window_len'] * gpd['sample_rate']))


def get_wav_path_text_list_from_manifest(loc):
    lines = open(loc, 'r').readlines()
    lines = [line.split(',') for line in lines]
    return [line[0] for line in lines], [line[1] for line in lines]


# 一个Dataset和DataLoader解决，区分train、eval、infer三种模式
class AudioDst(Dataset):
    def __init__(self, audio_base, mode='train', dev_or_test='dev', path_list=None, text_list=None):
        assert mode in ('train', 'eval', 'infer'), "mode must be train, eval or infer"
        if dev_or_test is not None:
            assert dev_or_test in ('dev', 'test'), "dev_or_test must be dev or test"

        if path_list is not None:
            self.path_list = path_list
        else:
            if mode == 'train':
                self.path_list = audio_base.train_wav_path_list
                # self.indices = audio_base.train_indices
            elif mode == 'eval':
                if dev_or_test == 'dev':
                    self.path_list = audio_base.dev_wav_path_list
                    # self.indices = audio_base.dev_indices
                elif dev_or_test == 'test':
                    self.path_list = audio_base.test_wav_path_list
                    # self.indices = audio_base.test_indices
            elif mode == 'infer':
                assert audio_base.infer_wav_path_list is not None, \
                    "you must provide wav path list in infer mode in AudioBase"
                self.path_list = audio_base.infer_wav_path_list
                # self.indices = audio_base.infer_indices

        if text_list is not None:
            assert path_list is not None
            self.text_list = text_list
        else:
            if path_list is not None:
                assert mode == 'infer'
                self.text_list = None
            else:
                if mode == 'train':
                    self.text_list = audio_base.train_text_list
                elif mode == 'eval':
                    if dev_or_test == 'dev':
                        self.text_list = audio_base.dev_text_list
                    elif dev_or_test == 'test':
                        self.text_list = audio_base.test_text_list

        self.word2int = audio_base.word2int
        self.data_aug = False
        if (mode == 'train') and (gpd['aug_prob'] > 0):
            self.data_aug = True
        self.audio_base = audio_base
        self.mode = mode

    def __len__(self):
        return len(self.path_list)
        # return len(self.indices)

    def __getitem__(self, idx):
        # return a tuple: (audio_log_mel, txt)
        audio_file_path = self.path_list[idx]
        feature = get_log_mel(self.mode == 'train', audio_file_path,
                              self.audio_base.ms, self.audio_base.window, self.data_aug)
        
        # print('index:', self.indices[idx], type(self.indices[idx]))  # <class 'numpy.int64'>
        # feature = get_log_mel(self.audio_base.hdf5_dst, self.indices[idx],
        #                       self.audio_base.ms, self.audio_base.window, self.data_aug)

        if self.text_list is not None:
            text = self.text_list[idx]
            text_int = [self.word2int.get(ele, self.word2int['<unk>']) for ele in text]
            return feature, text_int  # numpy array: [l, d], list: [n]
        else:
            return (feature,)


class AudioLoader(object):
    def __init__(self, dst):
        mode = dst.mode
        bsz = gpd['batch_size'] if mode == 'train' else gpd['eval_batch_size']
        sampler = TrainSampler(dst.text_list, gpd['batch_size']) if mode == 'train' else None
        num_workers = gpd['num_workers'] if mode == 'train' else gpd['eval_num_workers']
        collate_fn = self.train_collate_fn if mode == 'train' else self.collate_fn

        self.loader = DataLoader(dst, batch_size=bsz, shuffle=False,
                                 sampler=sampler,
                                 num_workers=num_workers,
                                 pin_memory=False,
                                 collate_fn=collate_fn)

        self.mode = mode

    @staticmethod
    def train_collate_fn(batch):
        batch.sort(key=lambda x: len(x[1]), reverse=True)  # 按text从长到短排序
        t = [ele[0] for ele in batch]

        # 1. batch audio
        t, lens = AudioLoader.batch_audio(t)

        # 2. batch text
        text = [torch.LongTensor([gpd['sos']] + ele[1]) for ele in batch]
        text2 = [torch.LongTensor(ele[1] + [gpd['eos']]) for ele in batch]
        text_lens = torch.IntTensor([ele.size(0) for ele in text])
        packed = nn.utils.rnn.pack_sequence(text)
        packed2 = nn.utils.rnn.pack_sequence(text2)

        return t, lens, packed, packed2, text_lens

    @staticmethod
    def collate_fn(batch):
        t = [ele[0] for ele in batch]

        # 1. batch audio
        t, lens = AudioLoader.batch_audio(t)

        if len(batch[0]) == 2:
            # 2. batch text, eval mode
            text = [ele[1] for ele in batch]
            return t, lens, text
        else:
            # no text, infer mode
            return t, lens, None

    @staticmethod
    def batch_audio(batch):
        lens = torch.IntTensor([t.size(0) for t in batch])
        if gpd['normalize']:
            # 采用instance normalization的方式
            if gpd['encoder_type'] != 'CNN2D':
                # [l, d]
                batch = [((ele - ele.mean(dim=0)) / (ele.std(dim=0) + 1e-7))
                         for ele in batch]
            else:
                # [w, c, h]
                batch = [(ele - ele.mean((0, 2), True)) /
                         (ele.transpose(1, 2).contiguous().view(
                             -1, ele.size(1)).std(0, keepdim=True)[..., None] + 1e-7)
                         for ele in batch]

        if gpd['encoder_type'] == 'CNN1D':
            batch = nn.utils.rnn.pad_sequence(batch, batch_first=True)  # [b, l, d]
            batch.transpose_(1, 2)  # [b, d, l]

        elif gpd['encoder_type'] == 'CNN2D':
            # [w, c, h]
            batch = nn.utils.rnn.pad_sequence(batch, batch_first=True)  # [b, w, c, h]
            batch = batch.permute(0, 2, 3, 1)  # [b, c, h, w]

        elif 'ATTENTION' in gpd['encoder_type']:
            # input: a list of [l, d]
            # output: [b, l, d]
            batch = nn.utils.rnn.pad_sequence(batch, True)

        return batch, lens


# test
def test_data():
    # case 1
    # audio_base = AudioBase()
    # dst = AudioDst(audio_base, 'train')
    # print(len(dst))  # 120098
    # dst = AudioDst(audio_base, 'eval', 'dev')
    # print(len(dst))  # 14326
    # dst = AudioDst(audio_base, 'eval', 'test')
    # print(len(dst))  # 7176
    # dst = AudioDst(audio_base, 'infer', path_list=['aa', 'bb'] * 123)
    # print(len(dst))

    # case 2
    # gpd['batch_size'] = 128
    # available_encoder_types = ['rnn_tanh', 'rnn_relu', 'lstm', 'gru',
    #                            'cnn1d', 'cnn2d', 'self_attention', 'self_local_attention']
    # audio_base = AudioBase()
    #
    # for et in available_encoder_types:
    #     print('*************************************')
    #     print(et)
    #     gpd['encoder_type'] = et
    #     dst = AudioDst(audio_base, 'train')
    #     loader = AudioLoader(dst).loader
    #     for i, ele in enumerate(loader, 1):
    #         ele = ele[0]
    #         print('train:', i, len(ele) if isinstance(ele, list) else ele.shape)
    #         if i == 2:
    #             break
    #
    #     dst = AudioDst(audio_base, 'eval', 'test')
    #     loader = AudioLoader(dst).loader
    #     for i, ele in enumerate(loader, 1):
    #         ele = ele[0]
    #         print(' test:', i, len(ele) if isinstance(ele, list) else ele.shape)
    #         if i == 2:
    #             break

    def get_aishell_1_info():

        ab = AudioBase()
        print(len(ab.train_text_list), len(ab.train_wav_path_list))  # 120098
        print(len(ab.dev_text_list), len(ab.dev_wav_path_list))  # 14326
        print(len(ab.test_text_list), len(ab.test_wav_path_list))  # 7176

        # random select 1000 wavs, and estimate mean size of each file
        n = 1000
        idx = np.random.permutation(len(ab.train_wav_path_list))[:n]
        size = 0  #
        for i in idx:
            rate, data = wavfile.read(ab.train_wav_path_list[i])
            size += data.size

        size /= n
        print(f'mean size: {size}')
        print(f"train size: {size * 120098 * 2 / 1024**3:.4f}GB"
              f"\ndev size: {size * 14326 * 2 / 1024**3:.4f}GB"
              f"\ntest size: {size * 7176 * 2 / 1024**3:.4f}GB")

        """
        train size: 16.3049GB
        dev size: 1.9449GB
        test size: 0.9742GB
        """

    def test_pre_load(pre_load):
        ab = AudioBase()
        n = 5000
        # np.random.seed(123)
        # idx = np.random.permutation(len(ab.train_wav_path_list))[:n]
        idx = np.random.permutation(n)

        if pre_load:
            print('preloading...')
            for i in range(n):
                fast_read(ab.train_wav_path_list[i])

        ts = time()
        for i in idx:
            fast_read(ab.train_wav_path_list[i])
        print(f"{'[preload]' if pre_load else '[no preload]'} time cost {time()-ts:.3f}s")

    def test_loader():
        gpd['encoder'] = 'RNN_TANH'

        audio_base = AudioBase()
        dst = AudioDst(audio_base)
        loader = AudioLoader(dst)

        for ele in loader.loader:
            print(len(ele))

    def test_kefu_audio_base():
        audio_base = AudioBase()
        # audio_base.verify(audio_base.train_wav_path_list)
        # audio_base.pick_and_show(audio_base.test_wav_path_list, audio_base.test_text_list)
        # audio_base.pick_and_show(audio_base.dev_wav_path_list, audio_base.dev_text_list)
        # audio_base.pick_and_show(audio_base.train_wav_path_list, audio_base.train_text_list)
        # audio_base.longest_text_len(audio_base.dev_wav_path_list, audio_base.dev_text_list)
        # audio_base.longest_text_len(audio_base.test_wav_path_list, audio_base.test_text_list)
        # audio_base.sorted_len(audio_base.train_text_list)
        # print(f"{audio_base.train_wav_path_list}\n{audio_base.train_text_list}")

    def test_hdf5_dst():
        gpd['batch_size'] = 128
        gpd['num_workers'] = 1

        audio_base = AudioBase()

        # dst = AudioDst(audio_base, 'train')
        dst = AudioDst(audio_base, 'eval', 'dev')

        loader = AudioLoader(dst)
        print('size of datset:', len(loader.loader))

        time_cost = []
        dur = Duration()
        dur.tic()
        for _ in range(10):
            for i, ele in enumerate(loader.loader, 1):
                time_cost.append(dur.toc())
                if i % 1 == 0:
                    print(i)
                if i == 50:
                    break

        # plt.plot(time_cost)
        # plt.grid()
        # plt.show()

    def save_dict():
        AudioBase.make_dict()
    # get_aishell_1_info()
    # test_pre_load(True)
    # test_loader()
    # test_kefu_audio_base()
    # test_hdf5_dst()

if __name__ == '__main__':
    # test_data()
    AudioBase()