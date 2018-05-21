import os, sys
import time

import hyperopt
from hyperopt import hp

from lib.data_pro import DataBuilder, batch_gen
from lib.define import *
from lib.framework import Seq2Seq


best_score = sys.float_info.max
hp_cnt = 0
score_board = []
base_dir = '../'
if DBG:
    batch_size = 64
else:
    batch_size = 1024
    batch_size = 128


def hp_search(max_iter_num=1000000):
    par_space = {
        'encode_len': hp.choice('encode_len', [360, 240, 480, 600, 720, 840, 960]),
        'n_hidden': hp.choice('n_hidden', [60, 80, 100, 150, 200]),
        'n_layers': hp.choice('n_layers', [1, 2, 3]),
        'dropout': hp.choice('dropout', [0.5, 0.3, 0.1]),
        'lr': hp.loguniform('lr', np.log(1e-4), np.log(0.01)),
        'with_space_attn': False,
        #'early_stopping_steps': hp.choice('early_stopping_steps', [400, 600, 800]),
        'amsgrad': hp.choice('amsgrad', [True, False]),
        'l2_scale': hp.loguniform('l2_scale', np.log(0.001), np.log(0.5)),
        'teacher_forcing_ratio': hp.choice('teacher_forcing_ratio', [0.3, 0.5, 0.7]),
    }

    trials = hyperopt.Trials()
    best = hyperopt.fmin(fn=hp_core, space=par_space, algo=hyperopt.tpe.suggest, trials=trials, max_evals=max_iter_num, rstate=np.random)
    print('Hyperopt over, best score is {}'.format(best))



def hp_core(pars):
    global hp_cnt, best_score, score_board, prefix
    hp_cnt += 1
    print('****************** Hyper Search round %d start at %s ************************' % (hp_cnt, now2str()))

    hp_pars = {
        'with_tblog': False,
        'enc_file': None,
        'dec_file': None,
        'encode_len': pars['encode_len'],
        'with_space_attn': pars['with_space_attn'],
        'dec_type': 0,
        'clip': 10,
        'lr': pars['lr'],
        'batch_size': batch_size,
        'n_dynamic_features': 6,
        'n_fixed_features': 8,
        'n_emb_features': 3,
        'n_hidden': pars['n_hidden'],
        'n_enc_layers': pars['n_layers'],
        'n_dec_layers': pars['n_layers'],
        'dropout': pars['dropout'],
        'log_interval': 20,
        'min_steps_to_checkpoint': 100,
        #'early_stopping_steps': pars['early_stopping_steps'],
        'early_stopping_steps': 400,
        'loss_type': 'SMAPE',
        'encoder': {
            'optimizer': {'type': 'Adam', 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 'l2_scale': pars['l2_scale'], 'amsgrad': pars['amsgrad']},
            #'optimizer': {'type': 'SGD', 'momentum': 0.9, 'nesterov': False, 'dampening': 0.0, 'epsilon': 1e-8, 'l2_scale': 1e-2},
            #'optimizer': {'type': 'AdamW', 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 'l2_scale': 1e-4},
        },
        'decoder': {
            'optimizer': {'type': 'Adam', 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 'l2_scale': pars['l2_scale'], 'amsgrad': pars['amsgrad']},
            #'optimizer': {'type': 'SGD', 'momentum': 0.9, 'nesterov': False, 'dampening': 0.0, 'epsilon': 1e-8, 'l2_scale': 1e-2},
            #'optimizer': {'type': 'AdamW', 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 'l2_scale': 1e-4},
        },
        'teacher_forcing_ratio': pars['teacher_forcing_ratio'],
    }
    hp_pars['val_to_end'] = hp_pars['encode_len'] + 100

    dg = DataBuilder(hp_pars)
    nn = Seq2Seq(hp_pars)

    score = nn.fit(dg.train_bb, dg.val_bb)

    if score is None:
        score = sys.float_info.max

    if score < best_score:
        best_score = score
        print('$$$$$$$$$$$$$$ Bingo! Got best score {} in round {} $$$$$$$$$$$$$$$$$$$$$$'.format(score, hp_cnt))
    return score


if __name__ == '__main__':

    hp_search()






