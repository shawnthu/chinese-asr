import torch
# from util import *
import argparse


# parser = argparse.ArgumentParser("saic-asr argument parser", add_help="fdfsfd")
# parser.add_argument('-f', type=str)
# parser.add_argument('-cuda', action="store_true")
# parser.add_argument('-weight', type=str)
# parser.add_argument('-lm', type=str)
#
# args = parser.parse_args()
# print(args.f, args.cuda, args.weight, args.lm)


x = torch.load('pretrain-0.06328-small.ckpt')

for k, v in x.items():
    print(k)
    for name, p in v.items():
        print(name, p.shape, p.dtype)
#
# torch.save({'encoder_state_dict': x['encoder_state_dict'],
#             'decoder_state_dict': x['decoder_state_dict']},
#            './pretrain-0.06328-small.ckpt')


import werobot