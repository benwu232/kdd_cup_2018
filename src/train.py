import os


from lib.data_pro import DataBuilder, batch_gen
from lib.wavenet import WaveNetEncDec
from lib.define import *

if __name__ == '__main__':
    base_dir = '../'


    if DBG:
        batch_size = 32
    else:
        batch_size = 64

    dg = DataBuilder(batch_size=batch_size)
    nn = WaveNetEncDec(
        reader=dg,
        log_dir=os.path.join(base_dir, 'logs'),
        checkpoint_dir=os.path.join(base_dir, 'checkpoints'),
        prediction_dir=os.path.join(base_dir, 'predictions'),
        optimizer='adam',
        learning_rate=.001,
        batch_size=batch_size,
        num_training_steps=200000,
        early_stopping_steps=5000,
        warm_start_init_step=0,
        regularization_constant=0.0,
        keep_prob=0.6,
        enable_parameter_averaging=False,
        num_restarts=2,
        min_steps_to_checkpoint=500,
        log_interval=10,
        num_validation_batches=1,
        grad_clip=20,
        residual_channels=32,
        skip_channels=32,
        dilations=[2**i for i in range(8)]*1,
        filter_widths=[2 for i in range(8)]*1,
        num_decode_steps=DECODE_STEPS,
    )


    nn.fit()
    nn.restore()
    nn.predict(batch_size)

