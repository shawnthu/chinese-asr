#!/usr/bin/env python

from data import *
from model import *
# import argparse
import tempfile


# def convert_audio(audio_bytes):
#     td = tempfile.mkdtemp()
#     with open(td + '/audio.amr', 'wb') as f:
#         f.write(audio_bytes)
#
#     tgt_audio = td + '/audio.wav'
#     os.system(f"ffmpeg -loglevel quiet -i {td + '/audio.amr'} -sample_fmt s16 -ar 16000 -ac 1 {tgt_audio}")
#     return td, tgt_audio


def convert_audio(path):
    if os.path.exists('tmp.wav'):
        os.remove('tmp.wav')
    os.system(f"ffmpeg -loglevel quiet -i {path} -sample_fmt s16 -ar 16000 -ac 1 tmp.wav")
    os.system(f"sox --norm=-1 tmp.wav a.wav && mv a.wav tmp.wav")
    return 'tmp.wav'


def parse(path, model, audio_base, lm_model, bw):
    # prepare data
    # td, audio_loc = convert_audio(audio_bytes)
    audio_loc = convert_audio(path)

    # infer
    # second_pass = True if (lm_path is not None and bw is not None) else False
    # print('[INFO] Begin to parse [Suggestion: audio duration no longer than 10 seconds]...')
    # ts = time()
    data = get_log_mel(False, audio_loc, audio_base.ms, audio_base.window, False)
    data = (data - data.mean(dim=0)) / (data.std(dim=0) + 1e-6)  # LD https://www.zhihu.com/question/48958503
    # 减均值对MFCC有效，但是对log mel，有效吗？此外这里用的除了减均值，还有除以标准差！对于某以特征，到底什么预处理姿势最正确？
    lens = torch.tensor([data.shape[0]])
    text = None
    data = [data]
    if bw is not None:
        if gpd['verbose']:
            print(f"[INFO] Beam Decode [bw={bw}]...")
        res = model.eval_one_batch_with_beam(model.device, bw, data, lens, text,
                                             audio_base.int2word,
                                             second_pass=True if lm_model is not None else False,
                                             lm_model=lm_model,
                                             lm_weight=1.5,
                                             length_weight=1.5
                                             )
    else:
        res = model.eval_one_batch_with_greedy(model.device, data, lens, audio_base.int2word, text)

    # print(model.encoder.training, model.attn_mechanism.training, model.decoder.training)
    # print(f"[PRED] {res.pred_text[0]}")
    # if res.text is not None:
    #     print(f"[REF]  {res.text[0]}")

    # print(f'[INFO] Time cost {time() - ts:.3f}s')

    # 4. other
    # print('[INFO] Delete the temp dir')
    # os.system(f"rm -r {td}")
    return res.pred_text[0]


class ASR:
    def __init__(self, lm_path=None, bw=None):
        # gpd['verbose'] = False
        # gpd['use_cuda'] = False
        # gpd['eval_num_workers'] = 0
        # gpd['temperature'] = 1
        # lm_path = '/home/extend/shawn/data/zh_giga.no_cna_cmn.prune01244.klm'
        # lm_path = None
        # bw = 8
        # bw = None

        if lm_path is not None and bw > 1:
            print('loading language model...')
            ts = time()
            lm_model = kenlm.LanguageModel(lm_path)
            print('loading cost %.3fs' % (time() - ts))
        else:
            lm_model = None

        # prepare model
        model = Model()
        model.load('./pretrain-0.06328.ckpt')
        model.model.eval()

        audio_base = AudioBase()

        self.audio_base = audio_base
        self.lm_model = lm_model
        self.model = model
        self.bw = bw
        # audio_loc = '/home/shawn/Desktop/record/hx/wav/audio9.wav'

    def __call__(self, path):
        text = parse(path, self.model, self.audio_base, self.lm_model, self.bw)
        return text


if __name__ == '__main__':

    # parser = argparse.ArgumentParser("saic-asr argument parser")
    # parser.add_argument('-f', type=str)
    # parser.add_argument('-cuda', action="store_true")
    # parser.add_argument('-weight', type=str)
    # parser.add_argument('-lm', type=str)
    # parser.add_argument('-bw', type=int)
    # # parser.add_argument('-v', action="store_true")
    #
    # args = parser.parse_args()
    # parse(os.path.abspath(os.path.expanduser(args.f)),
    #       args.cuda,
    #       os.path.abspath(os.path.expanduser(args.weight)),
    #       os.path.abspath(os.path.expanduser(args.lm)),
    #       args.bw)

    gpd['verbose'] = False
    gpd['use_cuda'] = False
    gpd['eval_num_workers'] = 0
    gpd['temperature'] = 1
    # lm_path = '/home/extend/shawn/data/nlp/zh_giga.no_cna_cmn.prune01244.klm'
    lm_path = None
    # bw = 4
    bw = None

    # if lm_path is not None and bw > 1:
    #     print('loading language model...')
    #     ts = time()
    #     lm_model = kenlm.LanguageModel(lm_path)
    #     print('loading cost %.3fs' % (time()-ts))
    # else:
    #     lm_model = None

    # # prepare model
    # model = Model()
    # model.load('./pretrain-0.06328.ckpt')
    # model.model.eval()
    #
    # audio_base = AudioBase()
    #
    # audio_loc = '/home/shawn/Desktop/record/hx/wav/audio9.wav'
    # print(parse(audio_loc, model, audio_base, lm_model, bw))

    asr = ASR(lm_path, bw)

    # path = '/home/extend/shawn/data/speech/AISHELL-2/iOS/data/wav/C0908/IC0908W0080.wav'
    path = '/home/extend/shawn/data/speech/kefu_data/wav/50000820190507112950_0_1.wav'
    text = asr(path)
    print(f'ENV: lm_path={lm_path}, bw={bw}\n'
          f'INPUT PATH : {path}\n'
          f'OUTPUT TEXT: {text}')