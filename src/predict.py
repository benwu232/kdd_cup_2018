import os
from collections import OrderedDict
import pandas as pd

from lib.data_pro import DataBuilder, batch_gen
from lib.define import *
from lib.framework import Seq2Seq

def preds_to_df(preds):
    preds = preds[:, :, :3]
    preds = preds.reshape(-1, 3)
    df = pd.DataFrame()
    id_list = []
    stations = bj_stations + ld_stations[:13]
    for st_cnt, st in enumerate(stations):
        for h in range(48):
            id_list.append('{}#{}'.format(st, h))
            if st_cnt >= 35:
                preds[st_cnt*48+h, 2] = np.nan
    df['test_id'] = id_list
    df['PM2.5'] = preds[:, 0]
    df['PM10'] = preds[:, 1]
    df['O3'] = preds[:, 2]
    #df = df['test_id', 'PM2.5', 'PM10', 'O3']
    return df

def predict(enc_file, dec_file, out_file):
    pars = {
        'with_tblog': False,
        'enc_file': enc_file,
        'dec_file': dec_file,
        'encode_len': 168,
        'val_to_end': 260,
        #'encode_len': 960,
        #'val_to_end': 1080,
        'dec_type': 1,
        'clip': 10,
        'lr': 0.001,
        'batch_size': 32,
        'n_dynamic_features': 6,
        'n_fixed_features': 6,
        'n_emb_features': 3,
        'n_hidden': 100,
        'n_enc_layers': 2,
        'n_dec_layers': 2,
        'dropout': 0.5,
        'log_interval': 10,
        'min_steps_to_checkpoint': 100,
        'early_stopping_steps': 1000,
        'loss_type': 'SMAPE',
        'encoder': {
            'optimizer': {'type': 'Adam', 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 'l2_scale': 1e-2, 'amsgrad': False},
            #'optimizer': {'type': 'SGD', 'momentum': 0.9, 'nesterov': False, 'dampening': 0.0, 'epsilon': 1e-8, 'l2_scale': 1e-2},
            #'optimizer': {'type': 'AdamW', 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 'l2_scale': 1e-4},
        },
        'decoder': {
            'optimizer': {'type': 'Adam', 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 'l2_scale': 1e-2, 'amsgrad': False},
            #'optimizer': {'type': 'SGD', 'momentum': 0.9, 'nesterov': False, 'dampening': 0.0, 'epsilon': 1e-8, 'l2_scale': 1e-2},
            #'optimizer': {'type': 'AdamW', 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 'l2_scale': 1e-4},
        },
        'teacher_forcing_ratio': 0.5,
    }


    dg = DataBuilder(pars)
    nn = Seq2Seq(pars)

    print('predicting ...')
    preds = nn.predict(dg.test_bb, DECODE_STEPS)
    print('converting to DataFrame ...')
    pdf = preds_to_df(preds)
    print('converting to CSV ...')
    pdf.to_csv(out_file, encoding='utf-8', index=False)



if __name__ == '__main__':
    base_dir = '../'

    enc_file = '../clf/kdd_2018-04-30_00-37-03-006971_410_enc.pth'
    dec_file = '../clf/kdd_2018-04-30_00-37-03-006971_410_dec.pth'


    pars = {
        'with_tblog': True,
        'enc_file': enc_file,
        'dec_file': dec_file,
        'encode_len': 720,
        'val_to_end': 800,
        #'encode_len': 960,
        #'val_to_end': 1080,
        'clip': 10,
        'lr': 0.001,
        'batch_size': 32,
        'n_dynamic_features': 6,
        'n_fixed_features': 3,
        'n_hidden': 100,
        'n_enc_layers': 2,
        'n_dec_layers': 2,
        'dropout': 0.5,
        'loss_type': 'SMAPE',
        'encoder': {
            'optimizer': {'type': 'Adam', 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 'l2_scale': 1e-2, 'amsgrad': True},
            #'optimizer': {'type': 'SGD', 'momentum': 0.9, 'nesterov': False, 'dampening': 0.0, 'epsilon': 1e-8, 'l2_scale': 1e-2},
            #'optimizer': {'type': 'AdamW', 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 'l2_scale': 1e-4},
        },
        'decoder': {
            'optimizer': {'type': 'Adam', 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 'l2_scale': 1e-2, 'amsgrad': True},
            #'optimizer': {'type': 'SGD', 'momentum': 0.9, 'nesterov': False, 'dampening': 0.0, 'epsilon': 1e-8, 'l2_scale': 1e-2},
            #'optimizer': {'type': 'AdamW', 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 'l2_scale': 1e-4},
        },
        'teacher_forcing_ratio': 0.0,
    }

    dg = DataBuilder(pars)
    nn = Seq2Seq(pars)

    print('predicting ...')
    preds = nn.predict(dg.test_bb, DECODE_STEPS)
    print('converting to DataFrame ...')
    pdf = preds_to_df(preds)
    print('converting to CSV ...')
    pdf.to_csv('../submit/submit.csv', encoding='utf-8', index=False)

