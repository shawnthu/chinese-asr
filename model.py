from gpd import *
from encoder import *
from decoder import *
from attention import *
from util import *
# from logger import *
from data import *

from datetime import datetime
from time import time
import os
from collections import defaultdict
import kenlm
from copy import copy
import json


class Model(object):
    def __init__(self):
        device = torch.device('cuda') if gpd['use_cuda'] else torch.device('cpu')
        # print('[INFO] use cuda' if gpd['use_cuda'] else '[WARN] use cpu')
        self.time_stamp = datetime.now().strftime('%Y-%m-%d_%H:%M')

        # 1. encoder
        enc_type = gpd['encoder_type']
        assert enc_type in ('CNN1D', 'CNN2D', 'LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU',
                            'SELF_ATTENTION', 'SELF_LOCAL_ATTENTION', 'CUSTOM')

        if enc_type == 'CNN1D':
            # self.encoder = CNN1DEncoder()
            # self.encoder = CNN1DSelfAttnEncoder()
            self.encoder = CNN1DRNNEncoder(); print('[INFO] Encoder: CNN1DRNN')
        elif enc_type == 'CNN2D':
            self.encoder = CNN2DEncoder()
            # self.encoder = CRNNEncoder(32, 64, 3)
        elif enc_type in ('LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU'):
            self.encoder = RNNEncoder()
        elif enc_type == 'SELF_ATTENTION':
            self.encoder = SelfAttentionEncoder()
        elif enc_type == 'SELF_LOCAL_ATTENTION':
            self.encoder = SelfLocalAttentionEncoder()
        else:
            self.encoder = None
        self.encoder.to(device)

        # 2. attention mechanism
        self.attn_mechanism = BauAttn(self.encoder.enc_size).to(device)

        # 3. decoder
        self.decoder = RNNDecoder(self.attn_mechanism).to(device)

        # 4. set criterion
        if gpd['label_smooth'] == 0:
            self.criterion = nn.CrossEntropyLoss(reduction='none')
        else:
            self.criterion = label_smoothing

        # 5. set optimizer
        self.optimizer = None

        # 6. set model_parameters, in case of fine_tune
        if gpd['fine_tune']:
            # self.model_parameters = list(self.decoder.proj_linear.parameters())
            # list(self.attn_mechanism.parameters())
            self.model_parameters = list(self.encoder.parameters()) + \
                                    list(self.decoder.parameters())
        else:
            self.model_parameters = list(self.encoder.parameters()) + \
                                    list(self.decoder.parameters())
        # print('number:', len(self.model_parameters))
        # self.model_parameters = list(set(self.model_parameters))
        # print(f'number of model params: {len(self.model_parameters)}'
        #       f'\n\ttrainable: {len([1 for p in self.model_parameters if p.requires_grad])}'
        #       f'\n\tnot trainable: {len([1 for p in self.model_parameters if not p.requires_grad])}')

        # 7. other
        # self.lm_model = None
        self.logger = None
        self.device = device
        self.model = nn.ModuleList([self.encoder,
                                    # self.attn_mechanism,
                                    self.decoder])

    # def train(self, loader, eval_loader):
    #     # save the config dictionary to json
    #     if not os.path.exists(gpd['json_config_path']):
    #         with open(gpd['json_config_path'], 'w') as f:
    #             json.dump(gpd, f)
    #         print('[INFO] save gpd to json')
    #     else:
    #         print('[WARN] config json file has existed')
    #
    #     # set the dir for log and checkpoint
    #     if not os.path.exists(gpd['save_dir']):
    #         os.makedirs(gpd['save_dir'])
    #         print('[INFO] ckpt dir created done')
    #     else:
    #         print('[WARN] ckpt dir has existed')
    #         # raise Exception('ckpt dir has existed!')
    #
    #     if self.logger is None:
    #         self.logger = Logger(gpd['log_dir'])
    #
    #     # set the optimizer
    #     if self.optimizer is None:
    #         if gpd['optimizer'] == 'ADAM':
    #             self.optimizer = optim.Adam(params=self.model_parameters,
    #                                         lr=gpd['base_lr'],
    #                                         weight_decay=gpd['l2_decay'],
    #                                         )
    #         elif gpd['optimizer'] == 'SGD':
    #             self.optimizer = optim.SGD(params=self.model_parameters,
    #                                        lr=gpd['base_lr'],
    #                                        momentum=gpd['momentum'],
    #                                        weight_decay=gpd['l2_decay'])
    #         else:
    #             raise Exception('unknown optimizer!')
    #
    #     print(f"[INFO] optimizer: {gpd['optimizer']}, base_lr: {gpd['base_lr']}")
    #     optimizer = self.optimizer
    #
    #     # 不需要save的变量
    #     should_stop = False
    #     int2word = eval_loader.loader.dataset.audio_base.int2word
    #     # steps_per_epoch = len(loader.loader.dataset) // gpd['batch_size'] + 1
    #     steps_per_epoch = len(loader.loader)
    #     num_eval_steps = get_steps(gpd['num_eval_steps'], steps_per_epoch)
    #     ramp_up_steps = get_steps(gpd['ramp_up_iters'], steps_per_epoch)
    #     print(f'[INFO] steps per epoch: {steps_per_epoch}, eval steps: {num_eval_steps}')
    #
    #     # 额外需要save的变量，每次中断后接着训练，需要保存的是：
    #     # 1. 训练参数
    #     # 2. step，loss，总训练时间，best_wer，base_lr，num_no_imprv
    #     tv = TrainVar(0, float('inf'), float('inf'), gpd['base_lr'], Duration(), 0)  # train_var
    #     # 'step', 'loss', 'best_wer', 'base_lr', 'duration', 'num_no_imprv'
    #
    #     # checkpoint manager
    #     ckpt_manager = Checkpoint()
    #     # ckpt_path = ckpt_manager.best_checkpoint(gpd['save_dir'])
    #     # ckpt_path = 'step-162560_wer-1.4166.ckpt'
    #     ckpt_path = gpd['continue_train_ckpt_path']
    #     if ckpt_path:
    #         # ckpt_path = os.path.join(gpd['save_dir'], ckpt_path)
    #         # iteration, loss, total_time, best_wer, num_no_imprv, base_lr = self.load(ckpt_path)
    #         tv = self.load(ckpt_path)
    #         epoch = round_down(tv.step / steps_per_epoch, 2)
    #         print(f'[Last checkpoint:]\n[step {tv.step:<4d} | epoch {epoch}] loss={tv.loss:.3f},'
    #               f' total time={tv.duration.cum_dur / 3600.:.4f}h, lr={tv.lr},'
    #               f' num_no_imprv: {tv.num_no_imprv}')
    #
    #         # 修改best_wer、num_no_imprv
    #         tv.best_wer = float('inf')
    #         tv.num_no_imprv = 0
    #
    #         # 主要就是修改lr
    #         print(f"[WARN] set the base_lr to {gpd['base_lr']:.5f}")
    #         tv.lr = gpd['base_lr']
    #         set_opt_lr(self.optimizer, tv.lr)
    #
    #     else:
    #         if gpd['fine_tune']:
    #             ckpt_path = ckpt_manager.best_checkpoint(gpd['fine_tune_dir'], min)
    #             if ckpt_path:
    #                 ckpt_path = 'iter-296000_-wer-0.073.ckpt'
    #                 ckpt_path = os.path.join(gpd['fine_tune_dir'], ckpt_path)
    #                 _ = self.load(ckpt_path)
    #
    #     # # ctrl-c 中断处理
    #     # TODO
    #     # def handler(sig, frame):
    #     #     print('KeyboardInterrupt')
    #     #     self.save((iteration, loss, total_time, best_wer, num_no_imprv),
    #     #               os.path.join(gpd['save_dir'], f'iter-{iteration}_-wer-{eval_res.wer:.3f}.ckpt'),
    #     #               )
    #     #     sys.exit(0)
    #     #
    #     # signal.signal(signal.SIGINT, handler)
    #
    #     tv.duration.tic()
    #     ema_100 = EMA(.99)  # 100
    #     # ema_500 = EMA(.998)  # 500
    #     while not should_stop:
    #         for data, lens, packed, packed2, text_lens in loader.loader:
    #
    #             # ramp up lr
    #             if tv.step < ramp_up_steps:
    #                 set_opt_lr(self.optimizer, tv.step / ramp_up_steps * tv.lr)
    #
    #             train_output = self.train_one_batch(self.device, data, lens, packed, packed2, text_lens, optimizer,
    #                                                 random.random() < gpd['ss'])
    #             if train_output is None:
    #                 # 此次前向无效，continue
    #                 continue
    #             tv.loss = train_output.loss.item()
    #             max_audio_len = train_output.max_audio_len
    #             max_text_len = train_output.max_text_len
    #
    #             if 'train_loss_ema_100' not in ema_100.shadow:
    #                 ema_100.register('train_loss_ema_100', tv.loss)
    #             else:
    #                 ema_100.update('train_loss_ema_100', tv.loss)
    #
    #             # if 'train_loss_ema_500' not in ema_500.shadow:
    #             #     ema_500.register('train_loss_ema_500', tv.loss)
    #             # else:
    #             #     ema_500.update('train_loss_ema_500', tv.loss)
    #
    #             tv.step += 1
    #             epoch = round_down(tv.step / steps_per_epoch, 2)
    #
    #             time_cost = tv.duration.toc()
    #             total_time = tv.duration.cum_dur
    #
    #             # terminal display
    #             cuda_info = f"CUDA {os.environ['CUDA_VISIBLE_DEVICES']} " if gpd['use_cuda'] else ''
    #             print(f"{cuda_info}[step {tv.step:<4d}"
    #                   f' ({tv.step % steps_per_epoch or steps_per_epoch:>6d} / {steps_per_epoch}) '
    #                   f'| epoch {epoch}] loss={tv.loss:.2f},'
    #                   f" ema_100={ema_100.shadow['train_loss_ema_100']:.2f},"
    #                   f' time={time_cost:.3f}s, total time={(total_time / 3600.):.3f}h,'
    #                   # f' max_audio_len={max_audio_len}, max_text_len={max_text_len},'
    #                   f' lr={tv.lr},'
    #                   f' best_wer={tv.best_wer * 100:.2f}%,'
    #                   f" num_no_imprv=[{tv.num_no_imprv}/{gpd['patience']}]", end='')
    #
    #             # train log
    #             self.logger.scalar('step_time_cost', time_cost, tv.step)
    #             self.logger.scalar('train_loss', tv.loss, tv.step)
    #             if train_output.grad_norm:
    #                 self.logger.scalar('train_grad_norm', train_output.grad_norm, tv.step)
    #
    #             # evaluation, run when loss < 1
    #             # 这里的loss不合理！应该计算loss的EMA！
    #             # 计算EMA: 100，500，1000，5000每一个对应不同的decay rate
    #             # TODO
    #             # if tv.step % num_eval_steps == 0 and tv.loss < 2.:
    #             # if tv.step % num_eval_steps == 0 and \
    #             #         (ema_100.shadow['train_loss_ema_100'] < 2. or \
    #             #         tv.duration.cum_dur > 5 * 3600):
    #             if tv.step % num_eval_steps == 0:
    #                 eval_nb = 0
    #                 eval_wer = 0.
    #                 eval_pred_text = []
    #                 eval_text = []
    #                 eval_alignment = []
    #                 eval_audio_len = []
    #                 eval_text_len = []
    #                 for data, lens, text in eval_loader.loader:
    #                     # eval_res = self.eval_one_batch_with_beam(self.device, gpd['beam_width'], data, lens, text,
    #                     #                                   eval_loader.dst.audio_base.int2word,
    #                     #                                   lm_weight=gpd['lm_weight'],
    #                     #                                   length_weight=gpd['length_weight'])
    #                     eval_res = self.eval_one_batch_with_greedy(self.device, data, lens, int2word, text)
    #                     eval_wer += eval_res.wer * lens.size(0)
    #                     eval_nb += lens.size(0)
    #                     eval_pred_text.extend(eval_res.pred_text)
    #                     eval_text.extend(eval_res.text)
    #                     eval_alignment.append(eval_res.alignment)
    #                     eval_audio_len.append(eval_res.audio_feat_len)
    #                     eval_text_len.append(eval_res.text_len)
    #                 eval_wer /= eval_nb
    #
    #                 print(f', eval_wer={eval_wer:.4f},'
    #                       f' dec_rate={(1 - eval_wer / (tv.best_wer + 1e-8)) * 100:.3f}%,'
    #                       f' best_wer={tv.best_wer:.4f}')
    #                 rand_disp_list(eval_pred_text, eval_text, n=20)
    #
    #                 # logger
    #                 # 1. log wer
    #                 self.logger.scalar('eval_wer', eval_wer, tv.step)
    #
    #                 # 2. log alignment
    #                 eval_imgs, eval_idx = parse_multi_batch_alignment(eval_alignment,
    #                                                                   eval_audio_len,
    #                                                                   eval_text_len)
    #                 log_img(self.logger, eval_imgs, 'dev', tv.step)
    #
    #                 # 3. log text
    #                 self.logger.text('eval_transcript',
    #                                  [(eval_pred_text[i], eval_text[i]) for i in eval_idx],
    #                                  tv.step)
    #
    #                 # lr schedule
    #                 # reduce_lr_on_plateau(
    #                 #     eval_wer, tv, gpd['dec_rate_threshold'], gpd['factor'], gpd['patience'], gpd['min_lr'])
    #                 if reduce_lr_on_plateau(
    #                         eval_wer, tv, gpd['dec_rate_threshold'], gpd['factor'], gpd['patience'], gpd['min_lr']):
    #                     pass
    #
    #                 # 每次依据tv的base_lr设置
    #                 set_opt_lr(self.optimizer, tv.lr)
    #
    #                 # save weights: 每次eval都save！！！
    #                 self.save(tv, os.path.join(gpd['save_dir'], f'step-{tv.step}_wer-{eval_wer:.4f}.ckpt'))
    #                     # TODO
    #                     # 务必要考虑训练很长时间都没有save的case！！！！！！！！！！！！
    #                     # 务必要考虑训练很长时间都没有save的case！！！！！！！！！！！！
    #                     # 务必要考虑训练很长时间都没有save的case！！！！！！！！！！！！
    #                     # 务必要考虑训练很长时间都没有save的case！！！！！！！！！！！！
    #                     # 务必要考虑训练很长时间都没有save的case！！！！！！！！！！！！
    #
    #             else:
    #                 print()
    #
    #             # lr schedule
    #             # if epoch == 200:
    #             #     base_lr /= 2  # 5e-2
    #             #     set_opt_lr(self.optimizer, base_lr)
    #             # elif epoch == 250:
    #             #     base_lr /= 10
    #             #     set_opt_lr(self.optimizer, base_lr)
    #
    #             # if (epoch - 200) >= 0 and ((epoch - 200) % 10 == 0):
    #             #     tv.1lr /= 2  # 200个epoch以后每隔10个epoch lr减半
    #             #     set_opt_lr(self.optimizer, tv.lr)
    #
    #             # # 2019/06/07晚上8点以后改的lr schedule
    #             # if epoch == 60:
    #             #     base_lr /= 10
    #             #     set_opt_lr(self.optimizer, base_lr)
    #             # elif epoch == 90:
    #             #     base_lr /= 10
    #             #     set_opt_lr(self.optimizer, base_lr)
    #
    #             # lr schedule for kefu data
    #             # 1 ~ 150: .1
    #             # 150 ~ 200: .01
    #             # 200 ~ 250: .001
    #             # if epoch == 800:
    #             #     base_lr /= 10
    #             #     set_opt_lr(self.optimizer, base_lr)
    #             # elif epoch == 900:
    #             #     base_lr /= 10
    #             #     set_opt_lr(self.optimizer, base_lr)
    #
    #             # # save weight in case of last iteration
    #             # save_point = os.path.join(gpd['save_dir'], f'iter-{iteration}-final-step.ckpt')
    #             # if iteration >= gpd['epochs'] * iters_per_epoch and not os.path.exists(save_point):
    #             #     should_stop = True
    #             #     # save最后一次
    #             #     print('save the last iteration!')
    #             #     self.save((iteration, loss, total_time, best_wer, num_no_imprv, base_lr), save_point)
    #             #     break
    #
    #     self.logger.close()

    def save(self, args, path):
        print('[INFO] Saving weights (encoder, attention, decoder)...', end='')
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'args': args,
        }, path)
        print(' Saving done.')

    def load(self, path):
        if gpd['verbose']:
            print(f'[INFO] Loading weights from {path}...', end='')
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        # if not gpd['fine_tune']:
        #     if checkpoint['optimizer_state_dict'] and self.optimizer:
        #         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if gpd['verbose']:
            print(' Loading done.')
        if 'args' in checkpoint:
            return checkpoint['args']

    # # train mode
    # def train_one_batch(self, device, data, lens, packed, packed2, text_lens, opt, ss=False):
    #     # input generated from dataloader
    #     self.model.train()
    #
    #     if isinstance(data, (tuple, list)):
    #         data = [ele.to(device) for ele in data]
    #     else:
    #         data = data.to(device)
    #
    #     lens = lens.to(device)
    #     packed = packed.to(device)
    #     packed2 = packed2.to(device)
    #     text_lens = text_lens.to(device)
    #
    #     max_text_len = text_lens[0].item()
    #
    #     # 1. encoder computation
    #     if gpd['encoder_type'] == 'SELF_LOCAL_ATTENTION':
    #         enc_output = self.encoder(gpd['ws'], data, lens)
    #     else:
    #         enc_output = self.encoder(data, lens)  # named tuple
    #     mask = get_mask_for_softmax(enc_output.out_lens)  # [l, b]
    #
    #     # 2. some pre computation for attention
    #     keys, values = self.attn_mechanism.compute_key_value(enc_output.out)
    #
    #     src_texts, batch_sizes, _, _ = packed
    #     tgt_texts, _, _, _ = packed2
    #
    #     alignments = []
    #     attn_hidden_states = []
    #
    #     start_idx = 0
    #     cell_state = self.decoder.get_initial_state(lens.size(0), enc_output.state)
    #     attn_hidden_state = None
    #
    #     if ss:
    #         loss = 0
    #         valid_num = 0
    #         src_mask = torch.ones(batch_sizes[0], dtype=torch.uint8, device=self.device)
    #
    #     cell_states = []
    #     for l in range(max_text_len):
    #         bsz = batch_sizes[l]
    #         src_tokens = src_texts[start_idx: start_idx + bsz]  # [b]
    #         if ss and l > 0:
    #             src_tokens = pred_tokens[:bsz]
    #
    #         if cell_state is not None:
    #             if isinstance(cell_state[0], tuple):
    #                 cell_state = [(ele[0][:bsz], ele[1][:bsz]) for ele in cell_state]
    #             else:
    #                 cell_state = [ele[:bsz] for ele in cell_state]
    #
    #         if attn_hidden_state is not None:
    #             attn_hidden_state = attn_hidden_state[:bsz]
    #
    #         dec_output = self.decoder(
    #             enc_output.out[:, :bsz], mask[:, :bsz], keys[:, :bsz], values[:, :bsz],  # 用于计算attention
    #             src_tokens, cell_state, attn_hidden_state, compute_logit=ss)  # if ss, then compute logit
    #
    #         if ss:
    #             pred_tokens = dec_output.logit[:bsz].argmax(1)
    #             tgt_tokens = tgt_texts[start_idx: start_idx + bsz]
    #             # tgt_tokens = pred_tokens.new_full(pred_tokens.shape, gpd['eos'])
    #             # tgt_tokens[start_idx: start_idx + batch_sizes[l]] = tgt_texts[start_idx: start_idx + batch_sizes[l]]
    #
    #             src_mask = src_mask[:bsz]
    #             loss += (self.criterion(dec_output.logit[:bsz], tgt_tokens) * src_mask.type_as(dec_output.logit)).sum()
    #             valid_num += src_mask.sum().type_as(loss)
    #
    #             src_mask &= (pred_tokens == tgt_tokens)
    #
    #         cell_state = dec_output.cell_state
    #         attn_hidden_state = dec_output.attn_hidden_state
    #
    #         cell_states.append(cell_state)
    #         attn_hidden_states.append(attn_hidden_state)
    #
    #         alignments.append(dec_output.alignment)  # for visual
    #
    #         start_idx += bsz
    #
    #     # calculate the logits, must be no ss
    #     if not ss:
    #         if gpd['attn_type'] == 'L':
    #             logits = self.decoder.proj_linear(torch.cat(attn_hidden_states, dim=0))
    #         else:
    #             hs = []
    #             for cell_state in cell_states:
    #                 last_layer_h = cell_state[-1][0] if gpd['decoder_type'] == 'LSTM' else cell_state[-1]
    #                 hs.append(last_layer_h)
    #             h_a = torch.cat([torch.cat((ele1, ele2), -1) for ele1, ele2 in zip(hs, attn_hidden_states)], 0)
    #             logits = self.decoder.proj_linear(h_a)
    #
    #         # forward and get loss
    #         loss = self.criterion(logits, tgt_texts)
    #         loss = loss.mean()
    #     else:
    #         loss = loss / (1e-7 + valid_num)
    #
    #     if torch.isnan(loss).all() or torch.isinf(loss).all():
    #         # 此时函数结束调用，loss被释放，整个图也被释放掉
    #         return None
    #
    #     # backward
    #     opt.zero_grad()
    #     loss.backward()
    #
    #     # maybe clip gradient, and get gradient norm
    #     if gpd['clip'] > 0:
    #         grad_norm = nn.utils.clip_grad_norm_(self.model_parameters, max_norm=gpd['clip'])
    #     else:
    #         grad_norm = 0
    #         parameters = list(filter(lambda p: p.grad is not None, self.model_parameters))
    #         for p in parameters:
    #             param_norm = (p.grad.data ** 2).sum()
    #             grad_norm += param_norm.item()
    #         grad_norm = grad_norm ** .5
    #
    #     opt.step()
    #
    #     return TrainOutput(loss=loss,
    #                        max_audio_len=lens.max().item(),
    #                        max_text_len=max_text_len,
    #                        grad_norm=grad_norm,
    #                        alignment=alignments,
    #                        audio_feat_len=enc_output.out_lens,
    #                        text_len=text_lens,)

    # eval mode
    @torch.no_grad()
    def eval_one_batch_with_greedy(self, device, data, lens, int2word=None, text=None):
        """
        :param data: [b, d, l] for cnn encoder or list of tensor for rnn encoder
        :param lens: [b], int32
        :param int2word: dict
        :param text: [b]
        :return: nametuple
        """
        self.model.eval()

        if isinstance(data, (list, tuple)):
            data = [ele.to(device) for ele in data]
            bsz = len(data)
        else:
            data = data.to(device)  # for cnn encoder
            bsz = data.size(0)

        lens = lens.to(device)

        # encoder computation
        if gpd['encoder_type'] == 'SELF_LOCAL_ATTENTION':
            enc_outputs, enc_len, cell_state = self.encoder(gpd['ws'], data, lens)
        else:
            enc_outputs, enc_len, cell_state = self.encoder(data, lens)  # named tuple
        # print('enc:', enc_outputs.shape, enc_outputs.sum(), enc_outputs.mean(), enc_outputs.std())
        # print('enc_len:', enc_len)
        mask = get_mask_for_softmax(enc_len)
        cell_state = self.decoder.get_initial_state(lens.size(0), cell_state)

        # attention pre computation
        keys, values = self.attn_mechanism.compute_key_value(enc_outputs)  # [l, b, d]
        # print('keys:', keys.shape, keys.sum())

        # decoder initialization
        tokens = torch.full((bsz,), gpd['sos'], dtype=torch.long, device=device)  # 初始tokens
        attn_hidden_state = None

        # other
        outputs = []
        alignments = []
        # finished = torch.zeros(bsz, dtype=torch.uint8, device=device)
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)
        final_lens = torch.zeros(bsz, dtype=torch.int32, device=device)
        accum_scores = torch.zeros(bsz, dtype=torch.float32, device=device)  # [b]

        for l in range(gpd['max_len']):
            dec_out = self.decoder(enc_outputs, mask, keys, values,
                                   tokens, cell_state, attn_hidden_state, compute_logit=True)
            logit, attn_hidden_state, alignment, cell_state = dec_out  # [b, v]
            alignments.append(dec_out.alignment)
            logp = logit - torch.logsumexp(logit, dim=1).view(-1, 1)  # [b, v]
            # print(l, 'logp:', logp.shape, logp.sum())

            # # for debug
            # _, chars = logp.topk(5, 1)
            # print([int2word[ele] for ele in chars[0].tolist()])
            # print([round(ele, 3) for ele in torch.exp(logp[0][chars[0]]).tolist()])
            # # debug end

            logp, tokens = logp.max(dim=1)
            # print(f"logp sum - {l}:", logp.sum().item())

            outputs.append(tokens)
            cur_finished = tokens == gpd['eos']

            # 判断是否首次结束，只有在之前没有结束并且当前结束的情况下，计算score
            torch.add(accum_scores, ((~finished) & cur_finished).type_as(logp) * logp, out=accum_scores)

            finished |= cur_finished
            final_lens += (~finished).type_as(final_lens)

            # 只有没有结束，则计算score
            torch.add(accum_scores, (~finished).type_as(logp) * logp, out=accum_scores)

            if finished.all():
                break

        # the indices of eos
        outputs = torch.stack(outputs, dim=1)  # [b, l]
        outputs = [seq[:l] for seq, l in zip(outputs.tolist(), final_lens.tolist())]

        pred_text = []
        score = []
        for i, ele in enumerate(outputs):
            if len(ele) == 0:
                pred_text.append('')
                score.append(.0)
            else:
                pred_text.append(''.join([int2word[e] for e in ele]))
                score.append(accum_scores[i].item() / (final_lens[i].item() + finished[i].item()))

        if text is not None:
            # wer = np.mean([get_wer(pred, ref) for pred, ref in zip(outputs, text)])
            text = [''.join([int2word[e] for e in ele]) for ele in text]
            wer = np.mean([get_wer(pred, ref) for pred, ref in zip(pred_text, text)])
        else:
            wer = None
        return EvalOutput(pred_text=pred_text, score=score, text=text, wer=wer, n=bsz, alignment=alignments,
                          audio_feat_len=enc_len, text_len=final_lens)

    @torch.no_grad()
    def eval_one_batch_with_beam(self, device, bmsz, data, lens, text, int2word,
                                 second_pass=gpd['second_pass'],
                                 # lm_path=gpd['lm_path'],
                                 lm_model=None,
                                 lm_weight=gpd['lm_weight'],
                                 length_weight=gpd['length_weight'],
                                 # logp_normalize=False,
                                 # logp_norm_coeff=1.,  # 0～1，越大归一化越明显，0则完全没有归一化
                                 # lm_normalize=False,
                                 ):
        # bmsz: int
        # data: [l, b, d]
        # lens: [b]
        self.model.eval()

        dict_words = sorted(int2word.items(), key=itemgetter(0))
        dict_words = [ele[1] for ele in dict_words]  # a list of word

        real_vcb_sz = self.decoder.real_vcb_sz

        # if second_pass:
        #     if self.lm_model is None:
        #         assert lm_path, "language model path must be provided!"
        #         if gpd['verbose']:
        #             print('loading language model...')
        #         ts = time()
        #         self.lm_model = kenlm.LanguageModel(lm_path)
        #         if gpd['verbose']:
        #             print('loading cost %.3fs' % (time()-ts))
        #
        # lm_model = self.lm_model

        if isinstance(data, (list, tuple)):
            data = [ele.to(device) for ele in data]
            bsz = len(data)
        else:
            data = data.to(device)  # [b, d, l] for cnn
            bsz = data.size(0)

        lens = lens.to(device)
        bbsz = bsz * bmsz  # b * k
        cand_size = 2 * bmsz

        # encoder computation
        if gpd['encoder_type'] == 'SELF_LOCAL_ATTENTION':
            enc_outputs, enc_len, cell_state = self.encoder(gpd['ws'], data, lens)
        else:
            enc_outputs, enc_len, cell_state = self.encoder(data, lens)  # named tuple
        mask = get_mask_for_softmax(enc_len)  # [l, b]
        cell_state = self.decoder.get_initial_state(lens.size(0), cell_state)

        # attention pre computation
        keys, values = self.attn_mechanism.compute_key_value(enc_outputs)  # [l, b, d]

        # tile batch: [l, b, d] -> [l, b * k, d]
        mask = tile_batch(mask, bmsz)
        keys = tile_batch(keys, bmsz)
        values = tile_batch(values, bmsz)
        enc_outputs = tile_batch(enc_outputs, bmsz)
        if cell_state is not None:
            if isinstance(cell_state[0], tuple):
                cell_state = [(tile_batch(ele[0], bmsz, True), tile_batch(ele[1], bmsz, True))
                              for ele in cell_state]
            else:
                cell_state = [tile_batch(ele, bmsz, True) for ele in cell_state]

        attn_hidden_state = None

        # language model states
        # null_states = [kenlm.State() for _ in range(bbsz)]

        # if lm_weight > 0.:
        #     lm_states = [kenlm.State() for _ in range(bbsz)]
        #     for s in lm_states:
        #         lm_model.BeginSentenceWrite(s)
        #         # lm_model.NullContextWrite(s)
        #     states_tmp = [kenlm.State() for _ in range(bbsz)]

        # set buffers
        hist_tokens = torch.empty(gpd['max_len'] + 1, bbsz, dtype=torch.long).to(device).fill_(gpd['pad'])  # [l, b*k]
        hist_tokens[0] = gpd['sos']

        # beam search initialization
        logp_scores_buf = torch.zeros(bbsz, dtype=torch.float32, device=device)  # [b*k]
        lm_scores_buf = torch.zeros(bbsz, dtype=torch.float32, device=device)  # [b*k]
        # about candidate
        cand_scores_buf = torch.zeros(bsz, cand_size, dtype=torch.float32).to(device)  # [b, 2*k]
        cand_beams_buf = torch.zeros(bsz, cand_size, dtype=torch.long).to(device)  # [b, 2*k]
        cand_indices_buf = torch.zeros(bsz, cand_size, dtype=torch.long).to(device)  # [b, 2*k]
        cand_indices = torch.zeros(bsz, cand_size, dtype=torch.long).to(device)  # [b, 2*k]

        # offsets
        bb_offsets = bmsz * torch.arange(bsz, dtype=torch.long).to(device)  # [b]
        cand_offsets = torch.arange(cand_size, device=device).view(1, -1).expand(bsz, -1)  # [b, 2*k]

        nb_finished_beams = torch.zeros(bsz, dtype=torch.int32, device=device)
        finished_tensors = []  # element: 3-tuple
        top_beam_finished = torch.zeros(bsz, dtype=torch.bool, device=device)

        # convert int text to text
        if text is not None:
            text = [''.join([int2word[idx] for idx in ele]) for ele in text]

        def parse_finished_tensors(t):
            # t: list[3-tuple]
            # tokens, batches, scores, length
            # [l, b], [b], [b], int
            res = defaultdict(list)
            for tokens, batches, scores in t:
                if batches.nelement() == 0:
                    # empty
                    continue

                tokens = tokens.t().tolist()  # [b, l]
                batches = batches.tolist()
                scores = scores.tolist()
                for i in range(len(batches)):
                    # parse tokens
                    # s = ''.join([int2word[ele] for ele in tokens[i]]) if tokens[i] else ''
                    # print(f's: {s}')
                    # if lm_weight > 0:
                    #     lm_score = lm_model.score(' '.join([int2word[k] for k in tokens[i]]))
                    #     # # debug info
                    #     # print(f"score: logp={scores[i]:.4f}, "
                    #     #       f"lm={lm_weight * lm_score:.4f}, "
                    #     #       f"length={length_weight * len(tokens[i])}")
                    #     scores[i] += lm_weight * (lm_score / ((len(tokens[i]) + 1e-5) if lm_normalize else 1)) + \
                    #                  length_weight * len(tokens[i])

                    res[batches[i]].append((tokens[i], scores[i]))

            res = dict(res)
            # max_res = dict()
            # sort by decreasing order
            # for k in res:
            #     # res[k] = sorted(res[k], key=itemgetter(1), reverse=True)
            #     # # debug info
            #     # out = [''.join(int2word[e] for e in ele[0]) for ele in sorted(res[k], key=itemgetter(1), reverse=True)]
            #     # for ele in out:
            #     #     print(ele)
            #
            #     res[k] = max(res[k], key=itemgetter(1))
            # length_std = [np.std([len(e[0]) for e in ele]) for ele in res.values()]
            # print(f"lenght std: mean = {np.mean(length_std)}, std = {np.std(length_std)}")
            if second_pass:
                for k, v in res.items():
                    if len(v) == 1:
                        res[k] = v[0]
                        continue

                    lm_score = [lm_model.score(' '.join([int2word[idx] for idx in ele[0]]), bos=True)
                                for ele in v]
                    length = [len(ele[0]) for ele in v]
                    logp = [ele[1] for ele in v]
                    score = [ele[0] + lm_weight * ele[1] + length_weight * ele[2]
                             for ele in zip(logp, lm_score, length)]  # 参考Google论文中second-pass re-score
                    i = np.argmax(score)
                    res[k] = v[i]
                return res

            return {k: max(v, key=itemgetter(1)) for k, v in res.items()}
            # return {k: max(v, key=itemgetter(1)) for k, v in res.items()}, \
            #        {k: [(''.join(int2word[e] for e in ele[0]), ele[1])
            #             for ele in sorted(v, key=itemgetter(1), reverse=True)]
            #         for k, v in res.items()}

        # def base_score(in_state, s, out_state):
        #     # change out_state
        #     # s: string with space separated or single char
        #     # s: list
        #     # words = s.split()
        #     in_state = copy(in_state)
        #     score = 0.
        #     if len(s) % 2 == 1:
        #         for w in s:
        #             score += lm_model.BaseScore(in_state, w, out_state)
        #             in_state, out_state = out_state, in_state
        #     else:
        #         for w in s[:-1]:
        #             score += lm_model.BaseScore(in_state, w, out_state)
        #             in_state, out_state = out_state, in_state
        #         score += lm_model.BaseScore(copy(in_state), s[-1], in_state)
        #     return score
        #
        # def get_s_from_tensor(t):
        #     t = t.t().tolist()  # [b*k, l]
        #     s = [[int2word[i] for i in seq] for seq in t]
        #     return s
        #
        # def get_lm_scores(words):
        #     # update the previous states
        #     # return torch.zeros(bbsz, real_vcb_sz, device=device)
        #     # s = get_s_from_tensor(t)
        #     # for i in range(bbsz):
        #     #     base_score(null_states[i], s[i], lm_states[i])
        #     cur_scores = []
        #     for i in range(bbsz):
        #         tmp_scores = []
        #         in_state = lm_states[i]
        #         out_state = states_tmp[i]
        #         for w in words:
        #             tmp_scores.append(lm_model.BaseScore(in_state, w, out_state))
        #         cur_scores.append(tmp_scores)
        #     return torch.tensor(cur_scores, device=device)  # [b*k, v]
        #
        # def update_lm_states(bb_indices, new_tokens):
        #     # bb_indices: [b*k]
        #     # new_tokens: [b*k]
        #     bb_indices = bb_indices.tolist()
        #     new_tokens = new_tokens.tolist()
        #     for i in range(len(bb_indices)):
        #         lm_model.BaseScore(copy(lm_states[bb_indices[i]]), int2word[new_tokens[i]], lm_states[i])
        #     return 0

        for l in range(gpd['max_len']):
            # 1. get decoder input
            tokens = hist_tokens[l]  # [b*k]

            # if l == 0:
            #     tokens = tokens[::bmsz]  # [b]
            #     h, c = self.decoder(tokens, attns, h, c)  # [b, d]
            #     attns, _ = self.attn(enc_outputs, )

            # 2. decoder cell computation
            dec_out = self.decoder(enc_outputs, mask, keys, values,
                                   tokens, cell_state, attn_hidden_state, compute_logit=True)
            logit, attn_hidden_state, alignment, cell_state = dec_out

            # calculate logp
            logit /= gpd['temperature']  # add softmax temperature
            logp = logit - torch.logsumexp(logit, dim=1).view(-1, 1)  # [b*k, v]
            torch.add(logp, logp_scores_buf.view(-1, 1), out=logp)
            # if not logp_normalize:
            #     torch.add(logp, logp_scores_buf.view(-1, 1), out=logp)
            # else:
            #     torch.add(logp * (1 / (l+1)),
            #               logp_scores_buf.view(-1, 1) * (l / (l+1)), out=logp)

            # # calculate language model scores
            # if lm_weight > 0.:
            #     lms = get_lm_scores()  # [b*k, v]
            #     if not lm_normalize:
            #         torch.add(lms, lm_scores_buf.view(-1, 1), out=lms)
            #     else:
            #         torch.add(lms * (1 / (l + 2)),
            #                   lm_scores_buf.view(-1, 1) * ((l + 1) / (l + 2)),
            #                   out=lms)

            # logp + weight1 * lm + weight2 * len(s)
            # if lm_weight > 0.:
            #     scores = (logp + lm_weight * lms + length_weight * (l+1)).view(bsz, -1)  # [b, k*v]
            # else:
            #     scores = (logp + length_weight * (l + 1)).view(bsz, -1)  # [b, k*v]

            # scores = (logp + length_weight * (l + 1)).view(bsz, -1)  # [b, k*v]
            scores = logp.view(bsz, -1)  # [b, k * v]

            if l == 0:
                torch.topk(scores[:, :real_vcb_sz], cand_size, out=(cand_scores_buf, cand_indices))  # [b, 2*k]
            else:
                torch.topk(scores, cand_size, out=(cand_scores_buf, cand_indices))  # [b, 2*k]
            torch.div(cand_indices, real_vcb_sz, out=cand_beams_buf)  # [b, 2*k]
            torch.fmod(cand_indices, real_vcb_sz, out=cand_indices_buf)  # [b, 2*k]

            # print debug info
            # print(l + 1, [[int2word[e] for e in l] for l in cand_indices_buf.tolist()][0],
            #       list(map(PrettyFloat, torch.gather(scores,  # F.softmax(logits, dim=1).view(bsz, -1)
            #                                          dim=1,
            #                                          index=cand_indices_buf)[0].tolist())))

            # 1. handle eos
            k_beams = cand_beams_buf[:, :bmsz]  # [b, k]
            k_indices = cand_indices_buf[:, :bmsz]  # [b, k]
            k_bb_indices = k_beams + bb_offsets.view(-1, 1)

            finished_mask = k_indices == gpd['eos']  # [b, k]，多个eos不可能来自同一个beam
            cur_nb_finished_beams = finished_mask.sum(dim=1).type_as(nb_finished_beams)  # [b] torch.int32
            # 1-D tensor, maybe empty tensor: t.nelement() == 0
            finished_bb_indices = k_bb_indices.masked_select(finished_mask)  # maybe empty
            finished_tokens = hist_tokens[1:(l+1), finished_bb_indices]  # 包含sos，但不包括eos
            torch.add(nb_finished_beams, cur_nb_finished_beams, out=nb_finished_beams)
            finished_batches = finished_bb_indices / bmsz  # maybe empty
            finished_scores = cand_scores_buf[:, :bmsz].masked_select(finished_mask)  # maybe empty
            # tensor, batch, score
            finished_tensors.append((finished_tokens, finished_batches, finished_scores))
            # 如何判定解码完成？有以下几种方案：
            # 方案1. 对于每个样本，解码出k个finished的句子，就认为已结束
            # if (nb_finished_beams >= bmsz).all():
                # early stop
                # break

            # 方案2. 对于每个样本，解码出一个top finished的句子，就认为已结束
            top_beam_finished |= (k_indices[:, 0] == gpd['eos'])
            if top_beam_finished.all():
                if gpd['verbose']:
                    print('Each sample has top decoded beam. Decode early stop!')
                break

            # 2. update states for next iteration
            finished_mask = cand_indices_buf == gpd['eos']  # [b, 2*k]
            finished_mask = cand_offsets + finished_mask.type_as(cand_offsets) * cand_size  # [b, 2*k]
            _ignore, active_hypos = torch.topk(finished_mask, bmsz, largest=False)
            k_beams = torch.gather(cand_beams_buf, dim=1, index=active_hypos)  # [b, k]
            k_bb_indices = (k_beams + bb_offsets.view(-1, 1)).view(-1)  # [b*k]
            k_indices = torch.gather(cand_indices_buf, dim=1, index=active_hypos)
            # print('active hypos:', active_hypos)

            # update decoder next input
            enc_outputs = enc_outputs[:, k_bb_indices]
            mask = mask[:, k_bb_indices]
            keys = keys[:, k_bb_indices]
            values = values[:, k_bb_indices]
            attn_hidden_state = attn_hidden_state[k_bb_indices]

            # update cell state: h, c
            # cell_state = (cell_state[0][k_bb_indices], cell_state[1][k_bb_indices])
            cell_state = [(ele[0][k_bb_indices], ele[1][k_bb_indices])
                          for ele in cell_state]

            # update hist_tokens
            hist_tokens = hist_tokens[:, k_bb_indices]
            hist_tokens[l+1] = k_indices.view(-1)

            # update the logp_scores_buf, lm_scores_buf
            torch.gather(cand_scores_buf, 1, active_hypos, out=logp_scores_buf.view(bsz, -1))  # [b*k]

            # 显示k_indices
            # disp_s = [''.join(int2word[i] for i in ele) for ele in hist_tokens[1:(l+2)].t().tolist()]
            # for ele in disp_s:
            #     print(ele)

            # if lm_weight > 0.:
            #     torch.gather(lms.view(bsz, -1), 1, active_indices, out=lm_scores_buf.view(bsz, -1))

            # # update lm_states
            # if lm_weight > 0.:
            #     update_lm_states(k_bb_indices, hist_tokens[l+1])

        # get the final predicted text
        # 以下代码几乎不占时间
        eos_batches = parse_finished_tensors(finished_tensors)  # dict
        if isinstance(eos_batches, tuple):
            eos_batches, all_batches = eos_batches
            # eos_batches, all_batches = parse_finished_tensors(finished_tensors)  # dict
            print('****' * 20)
            for k, v in all_batches.items():
                print(f"{k} {text[k]}")
                if len(v) == 1:
                    print(f"  {v[0][1]:.3f}: {v[0][0]}")
                else:
                    for i, ele in enumerate(v, 1):
                        print(f"  {i} {ele[1]:.3f}: {ele[0]}")
            print('****' * 20)
        # for ele in eos_batches[0]:
        #     print('score: %.3f' % ele[1], ''.join(int2word[idx] for idx in ele[0]))

        if len(eos_batches) < bsz:
            unfinished_batches = torch.tensor(list(set(range(bsz)).difference(set(eos_batches.keys()))))
            # print(unfinished_batches)
            active_scores = logp_scores_buf + lm_weight * lm_scores_buf + length_weight * (l+1)  # [b*k]
            scores, idx = torch.topk(active_scores.view(bsz, -1)[unfinished_batches], k=1, dim=1)  # [b, 1]
            scores = scores.view(-1).tolist()
            idx = idx.view(-1)
            bb_idx = idx + bb_offsets[unfinished_batches]
            t = (hist_tokens[1:l + 2, bb_idx]).t().tolist()
            # s = [(''.join([int2word[i] for i in seq]), _score) for (seq, _score) in zip(t, scores)]
            d = dict(zip(unfinished_batches.tolist(), zip(t, scores)))
            eos_batches.update(d)
            # full_batches.update({k: [v[1]] for k, v in d.items()})

        outputs = sorted(eos_batches.items(), key=itemgetter(0))
        outputs = [ele[1] for ele in outputs]
        pred_text = [''.join([int2word[idx] for idx in ele[0]]) for ele in outputs]
        score = [ele[1] for ele in outputs]
        # for ele in outputs:
        #     print(ele)

        # calculate the wer
        wer = None
        if text is not None:
            wer = np.mean([get_wer(pred, ref) for pred, ref in zip(pred_text, text)])
        return EvalOutput(pred_text=pred_text, score=score, text=text, wer=wer, n=bsz, alignment=None,
                          audio_feat_len=None, text_len=None)

    @torch.no_grad()
    def eval_with_lm(self, device, bmsz, data, lens, text, int2word,
                     lm_model_path='/home/extend/shawn/data/zh_giga.no_cna_cmn.prune01244.klm',
                     lm_weight=.0,
                     length_weight=.01,
                     logp_normalize=False,
                     logp_norm_coeff=1.,  # 0～1，越大归一化越明显，0则完全没有归一化
                     lm_normalize=False,
                     ):
        """
        softmax的概率不该作为评分
        bmsz: int
        data: [l, 1, d]
        lens: [1]
        """
        self.model.eval()

        dict_words = sorted(int2word.items(), key=itemgetter(0))
        dict_words = [ele[1] for ele in dict_words]  # a list of word

        real_vcb_sz = self.decoder.real_vcb_sz

        if lm_weight > 0.:
            if self.lm_model is None:
                assert lm_model_path, "language model path must be provided!"
                print('loading language model...')
                ts = time()
                self.lm_model = kenlm.LanguageModel(lm_model_path)
                print('loading done, time cost %.3fs' % (time() - ts))
        else:
            self.lm_model = None

        lm_model = self.lm_model

        if isinstance(data, (list, tuple)):
            data = [ele.to(device) for ele in data]
            bsz = len(data)
        else:
            data = data.to(device)  # [b, d, l] for cnn
            bsz = data.size(0)

        lens = lens.to(device)
        bbsz = bsz * bmsz  # b * k
        cand_size = 2 * bmsz

        # encoder computation
        if gpd['encoder_type'] == 'SELF_LOCAL_ATTENTION':
            enc_outputs, enc_len, cell_state = self.encoder(gpd['ws'], data, lens)
        else:
            enc_outputs, enc_len, cell_state = self.encoder(data, lens)  # named tuple
        mask = get_mask_for_softmax(enc_len)  # [l, b]
        cell_state = self.decoder.get_initial_state(lens.size(0), cell_state)

        # attention pre computation
        keys = self.attn_mechanism.pre_compute(enc_outputs)  # [l, b, d]

        # tile batch: [l, b, d] -> [l, b * k, d]
        mask = tile_batch(mask, bmsz)
        keys = tile_batch(keys, bmsz)
        enc_outputs = tile_batch(enc_outputs, bmsz)
        if cell_state is not None:
            if isinstance(cell_state[0], tuple):
                cell_state = [(tile_batch(ele[0], bmsz, True), tile_batch(ele[1], bmsz, True))
                              for ele in cell_state]
            else:
                cell_state = [tile_batch(ele, bmsz, True) for ele in cell_state]

        attn_hidden_state = None

        # language model states
        # null_states = [kenlm.State() for _ in range(bbsz)]

        # if lm_weight > 0.:
        #     lm_states = [kenlm.State() for _ in range(bbsz)]
        #     for s in lm_states:
        #         lm_model.BeginSentenceWrite(s)
        #         # lm_model.NullContextWrite(s)
        #     states_tmp = [kenlm.State() for _ in range(bbsz)]

        # set buffers
        hist_tokens = torch.empty(gpd['max_len'] + 1, bbsz, dtype=torch.long).to(device).fill_(gpd['pad'])  # [l, b*k]
        hist_tokens[0] = gpd['sos']

        # beam search initialization
        logp_scores_buf = torch.zeros(bbsz, dtype=torch.float32, device=device)  # [b*k]
        lm_scores_buf = torch.zeros(bbsz, dtype=torch.float32, device=device)  # [b*k]
        # about candidate
        cand_scores_buf = torch.zeros(bsz, cand_size, dtype=torch.float32).to(device)  # [b, 2*k]
        cand_beams_buf = torch.zeros(bsz, cand_size, dtype=torch.long).to(device)  # [b, 2*k]
        cand_indices_buf = torch.zeros(bsz, cand_size, dtype=torch.long).to(device)  # [b, 2*k]
        cand_indices = torch.zeros(bsz, cand_size, dtype=torch.long).to(device)  # [b, 2*k]

        # offsets
        bb_offsets = bmsz * torch.arange(bsz, dtype=torch.long).to(device)  # [b]
        cand_offsets = torch.arange(cand_size, device=device).view(1, -1).expand(bsz, -1)  # [b, 2*k]

        nb_finished_beams = torch.zeros(bsz, dtype=torch.int32, device=device)
        finished_tensors = []  # element: 3-tuple
        top_beam_finished = torch.zeros(bsz, dtype=torch.uint8, device=device)

        def parse_finished_tensors(t):
            # t: list[3-tuple]
            # tokens, batches, scores, length
            # [l, b], [b], [b], int
            res = defaultdict(list)
            for tokens, batches, scores in t:
                if batches.nelement() == 0:
                    # empty
                    continue

                tokens = tokens.t().tolist()  # [b, l]
                batches = batches.tolist()
                scores = scores.tolist()
                for i in range(len(batches)):
                    # parse tokens
                    # s = ''.join([int2word[ele] for ele in tokens[i]]) if tokens[i] else ''
                    # print(f's: {s}')
                    if lm_weight > 0:
                        lm_score = lm_model.score(' '.join([int2word[k] for k in tokens[i]]))
                        # # debug info
                        # print(f"score: logp={scores[i]:.4f}, "
                        #       f"lm={lm_weight * lm_score:.4f}, "
                        #       f"length={length_weight * len(tokens[i])}")
                        scores[i] += lm_weight * (lm_score / ((len(tokens[i]) + 1e-5) if lm_normalize else 1)) + \
                                     length_weight * len(tokens[i])

                    res[batches[i]].append((tokens[i], scores[i]))

            res = dict(res)
            # max_res = dict()
            # sort by decreasing order
            for k in res:
                # res[k] = sorted(res[k], key=itemgetter(1), reverse=True)
                # # debug info
                # out = [''.join(int2word[e] for e in ele[0]) for ele in sorted(res[k], key=itemgetter(1), reverse=True)]
                # for ele in out:
                #     print(ele)

                res[k] = max(res[k], key=itemgetter(1))

            return res

        def base_score(in_state, s, out_state):
            # change out_state
            # s: string with space separated or single char
            # s: list
            # words = s.split()
            in_state = copy(in_state)
            score = 0.
            if len(s) % 2 == 1:
                for w in s:
                    score += lm_model.BaseScore(in_state, w, out_state)
                    in_state, out_state = out_state, in_state
            else:
                for w in s[:-1]:
                    score += lm_model.BaseScore(in_state, w, out_state)
                    in_state, out_state = out_state, in_state
                score += lm_model.BaseScore(copy(in_state), s[-1], in_state)
            return score

        def get_s_from_tensor(t, transpose=True):
            if transpose:
                t = t.t().tolist()  # [b*k, l]
            else:
                t = t.tolist()  # [l, b*k]
            s = [[int2word[i] for i in seq] for seq in t]
            return s

        def get_lm_scores(words):
            # update the previous states
            # return torch.zeros(bbsz, real_vcb_sz, device=device)
            # s = get_s_from_tensor(t)
            # for i in range(bbsz):
            #     base_score(null_states[i], s[i], lm_states[i])
            cur_scores = []
            for i in range(bbsz):
                tmp_scores = []
                in_state = lm_states[i]
                out_state = states_tmp[i]
                for w in words:
                    tmp_scores.append(lm_model.BaseScore(in_state, w, out_state))
                cur_scores.append(tmp_scores)
            return torch.tensor(cur_scores, device=device)  # [b*k, v]

        def update_lm_states(bb_indices, new_tokens):
            # bb_indices: [b*k]
            # new_tokens: [b*k]
            bb_indices = bb_indices.tolist()
            new_tokens = new_tokens.tolist()
            for i in range(len(bb_indices)):
                lm_model.BaseScore(copy(lm_states[bb_indices[i]]), int2word[new_tokens[i]], lm_states[i])
            return 0

        def calc_lm_score(hist_token, token):
            # [l, b*k], [b*k, n]
            assert hist_token.size(1) == token.size(0)
            hist = get_s_from_tensor(hist_token)  # [b*k, l]
            now = get_s_from_tensor(token, False)  # [b*k, n]

            score = []
            for i in range(len(hist)):
                score.append([])
                for j in range(token.size(1)):
                    s = lm_model.score(' '.join(hist[i] + now[i][j: j + 1]), bos=False, eos=False)
                    score[-1].append(s)
            return score

        score = logp_scores_buf.new_full((bbsz, real_vcb_sz), -np.inf)  # [b*k, v]
        for l in range(gpd['max_len']):
            # 1. get decoder input
            tokens = hist_tokens[l]  # [b*k]

            # if l == 0:
            #     tokens = tokens[::bmsz]  # [b]
            #     h, c = self.decoder(tokens, attns, h, c)  # [b, d]
            #     attns, _ = self.attn(enc_outputs, )

            # 2. decoder cell computation
            dec_out = self.decoder(enc_outputs, mask, keys,
                                   tokens, cell_state, attn_hidden_state, compute_logit=True)
            logit, attn_hidden_state, alignment, cell_state = dec_out

            # calculate logp
            logit /= gpd['temperature']  # add softmax temperature

            # 计算整个字典范围内的lm得分没有必要
            # 为了减小计算lm得分的范围，取logit前n = 20个token，
            logit, token = logit.topk(20, 1)  # [b*k, n], [b*k, n]
            # 计算hist_tokens 和 token的lm得分
            lm_score = calc_lm_score(hist_tokens[: (l + 1)], token)  # list: [b*k, n]
            lm_score = torch.tensor(lm_score, dtype=logit.dtype, device=logit.device)

            score[:] = -np.inf
            score[torch.arange(len(score))[:, None].expand(-1, lm_score.size(1)), token] = lm_score
            scores = score.view(bsz, -1)

            # logp = logit - torch.logsumexp(logit, dim=1).view(-1, 1)  # [b*k, v]
            # if not logp_normalize:
            #     torch.add(logp, logp_scores_buf.view(-1, 1), out=logp)
            # else:
            #     torch.add(logp * (1 / (l + 1)),
            #               logp_scores_buf.view(-1, 1) * (l / (l + 1)), out=logp)
            #
            # # # calculate language model scores
            # # if lm_weight > 0.:
            # #     lms = get_lm_scores()  # [b*k, v]
            # #     if not lm_normalize:
            # #         torch.add(lms, lm_scores_buf.view(-1, 1), out=lms)
            # #     else:
            # #         torch.add(lms * (1 / (l + 2)),
            # #                   lm_scores_buf.view(-1, 1) * ((l + 1) / (l + 2)),
            # #                   out=lms)
            #
            # # logp + weight1 * lm + weight2 * len(s)
            # # if lm_weight > 0.:
            # #     scores = (logp + lm_weight * lms + length_weight * (l+1)).view(bsz, -1)  # [b, k*v]
            # # else:
            # #     scores = (logp + length_weight * (l + 1)).view(bsz, -1)  # [b, k*v]
            #
            # # scores = (logp + length_weight * (l + 1)).view(bsz, -1)  # [b, k*v]
            # scores = logp.view(bsz, -1)  # [b, k * v]

            if l == 0:
                torch.topk(scores[:, :real_vcb_sz], cand_size, out=(cand_scores_buf, cand_indices))  # [b, 2*k]
            else:
                torch.topk(scores, cand_size, out=(cand_scores_buf, cand_indices))  # [b, 2*k]
            torch.div(cand_indices, real_vcb_sz, out=cand_beams_buf)  # [b, 2*k]
            torch.fmod(cand_indices, real_vcb_sz, out=cand_indices_buf)  # [b, 2*k]

            # print debug info
            # print(l + 1, [[int2word[e] for e in l] for l in cand_indices_buf.tolist()][0],
            #       list(map(PrettyFloat, torch.gather(scores,  # F.softmax(logits, dim=1).view(bsz, -1)
            #                                          dim=1,
            #                                          index=cand_indices_buf)[0].tolist())))

            # 1. handle eos
            k_beams = cand_beams_buf[:, :bmsz]  # [b, k]
            k_indices = cand_indices_buf[:, :bmsz]  # [b, k]
            k_bb_indices = k_beams + bb_offsets.view(-1, 1)

            finished_mask = k_indices == gpd['eos']  # [b, k]，多个eos不可能来自同一个beam
            cur_nb_finished_beams = finished_mask.sum(dim=1).type_as(nb_finished_beams)  # [b] torch.int32
            # 1-D tensor, maybe empty tensor: t.nelement() == 0
            finished_bb_indices = k_bb_indices.masked_select(finished_mask)  # maybe empty
            finished_tokens = hist_tokens[1:(l + 1), finished_bb_indices]  # 包含sos，但不包括eos
            torch.add(nb_finished_beams, cur_nb_finished_beams, out=nb_finished_beams)
            finished_batches = finished_bb_indices / bmsz  # maybe empty
            finished_scores = cand_scores_buf[:, :bmsz].masked_select(finished_mask)  # maybe empty
            # tensor, batch, score
            finished_tensors.append((finished_tokens, finished_batches, finished_scores))
            # 如何判定解码完成？有以下几种方案：
            # 方案1. 对于每个样本，解码出k个finished的句子，就认为已结束
            # if (nb_finished_beams >= bmsz).all():
            # early stop
            # break

            # 方案2. 对于每个样本，解码出一个top finished的句子，就认为已结束
            top_beam_finished |= (k_indices[:, 0] == gpd['eos'])
            if top_beam_finished.all():
                print('Each sample has top decoded beam. Decode early stop!')
                break

            # 2. update states for next iteration
            finished_mask = cand_indices_buf == gpd['eos']  # [b, 2*k]
            finished_mask = cand_offsets + finished_mask.type_as(cand_offsets) * cand_size  # [b, 2*k]
            _ignore, active_hypos = torch.topk(finished_mask, bmsz, largest=False)
            k_beams = torch.gather(cand_beams_buf, dim=1, index=active_hypos)  # [b, k]
            k_bb_indices = (k_beams + bb_offsets.view(-1, 1)).view(-1)  # [b*k]
            k_indices = torch.gather(cand_indices_buf, dim=1, index=active_hypos)
            # print('active hypos:', active_hypos)

            # update decoder next input
            enc_outputs = enc_outputs[:, k_bb_indices]
            mask = mask[:, k_bb_indices]
            keys = keys[:, k_bb_indices]
            attn_hidden_state = attn_hidden_state[k_bb_indices]

            # update cell state: h, c
            # cell_state = (cell_state[0][k_bb_indices], cell_state[1][k_bb_indices])
            cell_state = [(ele[0][k_bb_indices], ele[1][k_bb_indices])
                          for ele in cell_state]

            # update hist_tokens
            hist_tokens = hist_tokens[:, k_bb_indices]
            hist_tokens[l + 1] = k_indices.view(-1)

            # update the logp_scores_buf, lm_scores_buf
            torch.gather(cand_scores_buf, 1, active_hypos, out=logp_scores_buf.view(bsz, -1))  # [b*k]

            # 显示k_indices
            # disp_s = [''.join(int2word[i] for i in ele) for ele in hist_tokens[1:(l+2)].t().tolist()]
            # for ele in disp_s:
            #     print(ele)

            # if lm_weight > 0.:
            #     torch.gather(lms.view(bsz, -1), 1, active_indices, out=lm_scores_buf.view(bsz, -1))

            # # update lm_states
            # if lm_weight > 0.:
            #     update_lm_states(k_bb_indices, hist_tokens[l+1])

        # get the final predicted text
        # 以下代码几乎不占时间
        eos_batches = parse_finished_tensors(finished_tensors)  # dict
        # for ele in eos_batches[0]:
        #     print('score: %.3f' % ele[1], ''.join(int2word[idx] for idx in ele[0]))
        if len(eos_batches) < bsz:
            unfinished_batches = torch.tensor(list(set(range(bsz)).difference(set(eos_batches.keys()))))
            # print(unfinished_batches)
            active_scores = logp_scores_buf + lm_weight * lm_scores_buf + length_weight * (l + 1)  # [b*k]
            scores, idx = torch.topk(active_scores.view(bsz, -1)[unfinished_batches], k=1, dim=1)  # [b, 1]
            scores = scores.view(-1).tolist()
            idx = idx.view(-1)
            bb_idx = idx + bb_offsets[unfinished_batches]
            t = (hist_tokens[1:l + 2, bb_idx]).t().tolist()
            # s = [(''.join([int2word[i] for i in seq]), _score) for (seq, _score) in zip(t, scores)]
            d = dict(zip(unfinished_batches.tolist(), zip(t, scores)))
            eos_batches.update(d)
            # full_batches.update({k: [v[1]] for k, v in d.items()})

        outputs = sorted(eos_batches.items(), key=itemgetter(0))
        outputs = [ele[1] for ele in outputs]
        pred_text = [''.join([int2word[idx] for idx in ele[0]]) for ele in outputs]
        score = [ele[1] for ele in outputs]
        # for ele in outputs:
        #     print(ele)

        # calculate the wer
        text = [''.join([int2word[idx] for idx in ele]) for ele in text]
        wer = np.mean([get_wer(pred, ref) for pred, ref in zip(pred_text, text)])
        return EvalOutput(pred_text=pred_text, score=score, text=text, wer=wer, n=bsz, alignment=None,
                          audio_feat_len=None, text_len=None)

    # infer mode
    @torch.no_grad()
    def infer_one_batch(self, wav_path):
        self.model.eval()

        pass


def test_model():
    gpd['encoder_type'] = 'cnn'
    gpd['encoder_hidden_size'] = 256
    gpd['beam_width'] = 4

    gpd['temperature'] = 1.

    gpd['lm_weight'] = 0.

    gpd['length_weight'] = 0.

    print(f"beam width={gpd['beam_width']}, lm weight={gpd['lm_weight']}, lengh weight={gpd['length_weight']}")

    model = Model()
    audio_base = AudioBase()
    dst = AudioDst(audio_base, 'eval', 'dev')
    loader = AudioLoader(dst)

    # weights_path = '/home/share/anji/asr/code/SR_with_Pytorch/aishell-1/ckpt/' \
    #                'base_arch-clip_1-att_size_256/iter-45048_wer-0.118.ckpt'

    # _ = model.load(weights_path)
    total_pred_text = []
    total_text = []
    eval_nb = 0
    eval_wer = 0.
    wer_list = []
    ts = time()

    # print('begin running beam search decoder...')
    for i, (data, lens, text) in enumerate(loader.loader, 1):
        # for ele in text:
        #     print(f"{i:<2d} {''.join([audio_base.int2word[i] for i in ele])}")
        print(lens[:20].tolist())
        # output = model.eval_one_batch_with_greedy(data, lens, audio_base.int2word, text)

        output = model.eval_one_batch_with_beam(gpd['beam_width'], data, lens, text, audio_base.int2word,
                                                lm_weight=gpd['lm_weight'],
                                                length_weight=gpd['length_weight'],
                                                logp_normalize=False,
                                                logp_norm_coeff=1.,
                                                lm_normalize=False,
                                                )
        eval_nb += lens.size(0)
        eval_wer += output.wer * lens.size(0)
        wer_list.append(output.wer)
        # print(output.wer)

        # beam_text_score = zip(output.pred_text, output.score)

        # output2 = model.eval_greedy(data, lens, audio_base.int2word, text)
        # text_score = zip(output2.pred_text, output2.score)

        # for i, ele in enumerate(zip(beam_text_score, text_score), 1):
        #     print(f'{i:<3d} [beam]   score: {ele[0][1]:.4f}, text: {ele[0][0]}')
        #     print(f'{i:<3d} [greedy] score: {ele[1][1]:.4f}, text: {ele[1][0]}')

        total_pred_text.extend(output.pred_text)
        total_text.extend(output.text)

    eval_wer /= eval_nb
    print(f"beam width: {gpd['beam_width']}, parse {eval_nb} wavs, wer: {eval_wer:.3f},"
          f" time cost: {time()-ts:.3f}s, wer_list: {[f'{ele:.3f}' for ele in wer_list]}")

    assert len(total_pred_text) == len(total_text) == eval_nb
    idx = []
    for i in range(eval_nb):
        if total_pred_text[i] != total_text[i]:
            idx.append(i)
    print(f'error rate: {len(idx) / eval_nb:.4f}')


if __name__ == '__main__':
    test_model()


