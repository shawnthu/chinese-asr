import numpy as np


gpd = {
    'verbose': True,

    # audio
    'sample_rate': 16000,  # 8k, 16k, ...
    'bit_depth': 16,  # 8, 16
    'window_len': .025,  # window length second
    'window_step': .01,  # window step second

    'n_mels': 80,
    # stddev of Gaussian noise added to waveform to prevent quantization artefacts
    'dither': 1. / np.iinfo(np.int16).max,
    # 'dither': 0.,
    'preemphasis': .97,
    'delta_delta': True,
    'downsample': True,
    'normalize': True,  # 对输入语音特征采取类似instance normalization的方式

    # ----- data augmentation -----
    # volume
    'aug_prob': .0,  # .5 -> 87.5%, .4 -> 78.4%, .3 -> 65.7%, .2 -> 48.8%, .1 -> 27%
    # 'volume_perturbation': True,
    'volume_gain_max': 10.,
    'volume_gain_min': -10.,

    # 'speed_perturbation': True,
    'speed_rate_max': 1.05,
    'speed_rate_min': 0.95,

    # 'shift_perturbation': True,
    'shift_ms_max': 5,
    'shift_ms_min': -5,
    # ----- data augmentation end -----

    # dictionary
    'pad': 0,
    'sos': 1,
    'eos': 2,
    'unk': 3,
    # 'max_num_words': 3661,  # for aishell-1
    # 'max_num_words': 2900,  # 2899 ('肆', 2) 2900 ('倪', 1) for small aishell-1
    # 'max_num_words': 1106,  # 1874 ('狼', 10)
    # 1106 ('尔', 3)
    'max_num_words': 5000,  # 6502 ('嵒', 1) 6503 ('継', 1)

    # data loader
    # 'buffer_size': 5000,
    'shuffle_updates': 10,

    'fine_tune': False,
    # 'fine_tune_dir': './ckpt/base_arch/',

    # encoder
    # CNN1D, CNN2D, LSTM, GRU, RNN_TANH, RNN_RELU, SELF_ATTENTION, SELF_LOCAL_ATTENTION, CUSTOM
    'encoder_type': 'LSTM',
    'skip_step': 0,
    # rnn: list of [l, b, d]
    # cnn1d: [b, d, l]
    # cnn2d: [b, c, d, l]
    'encoder_hidden_size': 256,  # serves as out_channels when encoder is cnn
    'encoder_num_layers': 4,  # 3
    'residual': True,
    'encoder_bidirectional': True,  # only valid when encoder is rnn
    'norm': 'BN',  # BN, LN, IN, NONE, only valid when encoder is cnn
    # 'ks': (5, 5),  # (frequency, time)
    'ks': 3,  # time
    # 'stride': (2, 2),  # (frequency, time)
    'stride': [2, 2, 2, 1, 1],  # time
    'act': 'RELU',  # GLU, RELU, SIGMOID, TANH, valid when encoder is cnn
    'mha_proj': True,  # only valid when encoder is self (local) attention and multi heads
    'ws': 11,  # window size for SELF_LOCAL_ATTENTION encoder
    'ffn_size': 256,  # valid when encoder is self attention
    'self_attn_heads': 4,  # valid for self (local) attention model

    # decoder
    'decoder_type': 'LSTM',  # LSTM, GRU, RNN_TANH, RNN_RELU, SELF_ATTENTION, SELF_LOCAL_ATTENTION, CUSTOM
    'decoder_hidden_size': 512,  # 256
    'decoder_num_layers': 1,
    'embed_dim': 256,
    'temperature': 1.,
    'input_feeding': True,  # ref Effective Approaches to Attention-based
    'dec_init_cell_state_as_param': False,

    # attention
    'attn_type': 'B',  # B or L
    'attn_size': 128,  # 64
    'map_enc': False,
    'attn_hidden_size': 640,  # only valid when attn_type is L
    'heads': 1,  # for attention between encoder and decoder, not self attention
    'linear_map': False,  # only valid in case of multi-heads, 是否在concat muti heads之后，再linear map一下

    # loss
    'label_smooth': 0.1,  # .1

    # train
    'ss':  0,  # scheduled sampling probability, 0 means no
    'continue_train_ckpt_path': '//data/ckpt/map-enc_head-8/step-101860_wer-0.1800.ckpt',
    # 'continue_train_ckpt_path': '/data/ckpt/pretrained_model/pretrain-0.06328.ckpt',  # def None
    # 'continue_train_ckpt_path': '/data/ckpt/continue-train_no-ls/step-651904_wer-0.0626.ckpt',
    'batch_size': 256,  # 16: .2s,
    'epochs': 50,
    'optimizer': 'ADAM',  # ADAM or SGD
    'base_lr': 1e-3,  # from 1e-3 to 1e-4
    # 'base_lr': .1,  # .1 for SGD
    'momentum': .9,  # for SGD
    'min_lr': 1e-5,  # lr lower bound
    'clip': 0.,  # gradient clip: 1.
    'l2_decay': 1e-5,  # 1e-5
    'ramp_up_iters': 0,  # 0表示不ramp up

    # eval
    # 常规
    'eval_num_samples': 20,  # 14326 for full dev, 800 for small dataset
    'num_eval_steps': -1,  # 5000, -1 means one epoch
    'eval_batch_size': 256,
    'beam_width': 4,
    # 'lm_path': '/home/extend/shawn/data/zh_giga.no_cna_cmn.prune01244.klm',
    'lm_path': '/data/zh_giga.no_cna_cmn.prune01244.klm',
    'second_pass': True,

    # 解码和打分
    'max_len': 40,
    'lm_weight': 0.0,  # 0 means do not use language model
    'length_weight': 0.0,  # 0 means do not use length penalty

    # lower the lr
    'patience': 4,
    'dec_rate_threshold': 0,  # relative value
    'factor': .5,
}
