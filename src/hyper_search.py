import os, sys

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
    batch_size = 512


def hp_search(max_iter_num=1000000):
    par_space = {
        'encode_len': hp.choice('encode_len', [120, 144, 168, 192, 240]),
        'n_hidden': hp.choice('n_hidden', [60, 80, 100, 150]),
        'n_layers': hp.choice('n_layers', [1, 2, 3]),
        'dropout': hp.choice('dropout', [0.5, 0.4, 0.3]),
        #'early_stopping_steps': hp.choice('early_stopping_steps', [400, 600, 800]),
        'amsgrad': hp.choice('amsgrad', [True, False]),
        'l2_scale': hp.uniform('l2_scale', 0.001, 0.1),
        'teacher_forcing_ratio': hp.choice('teacher_forcing_ratio', [0.3, 0.5, 0.7]),
    }

    trials = hyperopt.Trials()
    best = hyperopt.fmin(fn=hp_core, space=par_space, algo=hyperopt.tpe.suggest, trials=trials, max_evals=max_iter_num)
    print('Hyperopt over, best score is {}'.format(best))



def hp_core(pars):
    global hp_cnt, best_score, score_board, prefix
    hp_cnt += 1
    print('****************** Hyper Search round %d start at %s ************************' % (hp_cnt, now2str()))

    hp_pars = {
        'with_tblog': True,
        'enc_file': None,
        'dec_file': None,
        'encode_len': pars['encode_len'],
        'val_to_end': 320,
        'dec_type': 1,
        'clip': 10,
        'lr': 0.001,
        'batch_size': batch_size,
        'n_dynamic_features': 6,
        'n_fixed_features': 8,
        'n_emb_features': 3,
        'n_hidden': pars['n_hidden'],
        'n_enc_layers': pars['n_layers'],
        'n_dec_layers': pars['n_layers'],
        'dropout': pars['dropout'],
        'log_interval': 10,
        'min_steps_to_checkpoint': 100,
        #'early_stopping_steps': pars['early_stopping_steps'],
        'early_stopping_steps': 300,
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





