import os


from lib.data_pro import DataBuilder, batch_gen
from lib.define import *
from lib.framework import Seq2Seq

if __name__ == '__main__':
    base_dir = '../'
    if DBG:
        batch_size = 64
    else:
        batch_size = 128

    pars = {
        'with_tblog': True,
        'enc_file': None,
        'dec_file': None,
        'encode_len': 720,
        'val_to_end': 800,
        #'encode_len': 960,
        #'val_to_end': 1080,
        'dec_type': 2,
        'clip': 10,
        'lr': 0.001,
        'batch_size': batch_size,
        'n_dynamic_features': 6,
        'n_fixed_features': 8,
        'n_hidden': 100,
        'n_enc_layers': 2,
        'n_dec_layers': 2,
        'dropout': 0.5,
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


    nn.fit(dg.train_bb, dg.val_bb)
    #nn.restore()
    #nn.predict(batch_size)

