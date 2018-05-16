import datetime as dt
import numpy as np
import os
import random
import glob
import torch
from torch.autograd import Variable
from collections import deque
from lib.define import USE_CUDA, device, load_dump, save_dump, init_logger

from torch.optim import *

from tensorboard_logger import log_value as tblog_value
from tensorboard_logger import configure as tblog_configure
from lib.model import *

class SmapeLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets, targets_nan):
        predictions = predictions[:, :, :3]
        targets = targets[:, :, :3]
        targets_nan = targets_nan[:, :, :3]
        numerator = torch.abs(predictions - targets) * 200.0
        denominator = torch.abs(predictions) + torch.abs(targets)
        # for kaggle, avoid 0 / 0
        denominator[numerator<1e-2] = 1.0
        targets_not_nan = 1 - targets_nan
        smape = (numerator / denominator) * targets_not_nan
        #avg_smape = smape.sum(dim=1) / targets_not_nan.sum(dim=1)
        avg_smape = smape.sum() / targets_not_nan.sum()
        return avg_smape



class EncDec(object):
    def __init__(self, model_pars):
        self.n_dynamic_features = model_pars['n_dynamic_features']*3
        self.n_fixed_features = model_pars['n_fixed_features']
        self.n_emb_features = model_pars['n_emb_features']
        self.n_enc_input = self.n_dynamic_features + self.n_fixed_features + self.n_emb_features
        self.n_hidden = model_pars['n_hidden']
        self.n_enc_layers = model_pars['n_enc_layers']
        self.n_dec_layers = model_pars['n_dec_layers']
        self.dec_type = model_pars['dec_type']
        self.dropo = model_pars['dropout']
        self.n_out = 6
        self.n_dec_input = self.n_out + self.n_fixed_features

        if 'enc_file' in model_pars and model_pars['enc_file']:
            self.load_encoder(model_pars['enc_file'])
        else:
            self.encoder = self.init_encoder()

        if 'dec_file' in model_pars and model_pars['dec_file']:
            self.load_decoder(model_pars['dec_file'])
        else:
            self.decoder = self.init_decoder(self.dec_type)

        self.clip = model_pars['clip']
        self.lr = model_pars['lr']
        #self.train_batch_per_epoch = model_pars['train_batch_per_epoch']
        #self.validate_batch_per_epoch = model_pars['validate_batch_per_epoch']
        self.timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        self.model_file = None

        self.n_training_steps = 100000000
        self.loss_averaging_window = 20
        self.log_interval = model_pars['log_interval']
        self.min_steps_to_checkpoint = model_pars['min_steps_to_checkpoint']
        self.early_stopping_steps = model_pars['early_stopping_steps']
        self.lr_scheduler = None
        self.with_weights = False
        self.logger = init_logger(log_file='../logs/{}.log'.format(self.timestamp))

        self.with_tblog = model_pars['with_tblog']
        if self.with_tblog:
            tblog_configure('../tblog/' + self.timestamp)

        self.set_loss_fn(model_pars['loss_type'])
        self.encoder_optimizer = self.set_optimizer(self.encoder, model_pars['encoder']['optimizer'])
        self.decoder_optimizer = self.set_optimizer(self.decoder, model_pars['decoder']['optimizer'])

        if USE_CUDA:
            self.encoder.cuda()
            self.decoder.cuda()

    def tblog_value(self, name, value, step):
        if self.with_tblog:
            tblog_value(name, value, step)

    def enable_train(self, mode=True):
        self.encoder.train(mode)
        self.decoder.train(mode)

    def set_optimizer(self, model, optim_pars):
        if optim_pars['type'] == 'SGD':
            optimizer = SGD(model.parameters(), lr=self.lr, weight_decay=optim_pars['l2_scale'], momentum=optim_pars['momentum'], dampening=optim_pars['dampening'], nesterov=optim_pars['nesterov'])
        elif optim_pars['type'] == 'Adadelta':
            optimizer = Adadelta(model.parameters(), lr=self.lr, rho=optim_pars['rho'], weight_decay=optim_pars['l2_scale'], eps=optim_pars['epsilon'])
        elif optim_pars['type'] == 'Adam':
            optimizer = Adam(model.parameters(), lr=self.lr, betas=(optim_pars['beta1'], optim_pars['beta2']), eps=optim_pars['epsilon'], weight_decay=optim_pars['l2_scale'])
        elif optim_pars['type'] == 'RMSprop':
            optimizer = RMSprop(model.parameters(), lr=self.lr, alpha=optim_pars['rho'], eps=optim_pars['epsilon'], weight_decay=optim_pars['l2_scale'], momentum=optim_pars['momentum'], centered=optim_pars['centered'])
        return optimizer

    def set_loss_fn(self, type='L1Loss'):
        if type == 'L1Loss':
            self.criterion = torch.nn.L1Loss(size_average=False)
        elif type == 'SMAPE':
            self.criterion = SmapeLoss()

    def init_encoder(self):
        self.encoder = EncoderRNN(self.n_enc_input, self.n_hidden, self.n_enc_layers, dropout=self.dropo)
        return self.encoder

    def init_decoder(self, type=0):
        if type == 0:
            self.decoder = DecoderRNN(self.n_dec_input, self.n_hidden, self.n_out, self.n_dec_layers, dropout=self.dropo)
        elif type == 1:
            self.decoder = BahdanauAttnDecoderRNN(attn_model='general', input_size=self.n_dec_input,
                                                  hidden_size=self.n_hidden, output_size=self.n_out,
                                                  n_layers=self.n_dec_layers, dropout_p=self.dropo, n_enc_input=self.n_enc_input)
        elif type == 2:
            self.decoder = LuongAttnDecoderRNN(attn_model='general', input_size=self.n_dec_input,
                                               hidden_size=self.n_hidden, output_size=self.n_out,
                                               n_layers=self.n_dec_layers, dropout_p=self.dropo)


        return self.decoder

    def save_model(self, model_prefix):
        enc_file = model_prefix + '_enc.pth'
        dec_file = model_prefix + '_dec.pth'
        torch.save(self.encoder, enc_file)
        torch.save(self.decoder, dec_file)

    def load_encoder(self, enc_file):
        self.encoder = torch.load(enc_file)

    def load_decoder(self, dec_file):
        self.decoder = torch.load(dec_file)

    def load_models(self, model_files):
        enc_file = model_files[0]
        dec_file = model_files[1]
        self.encoder = torch.load(os.path.join(self.model_dir, enc_file))
        self.decoder = torch.load(os.path.join(self.model_dir, dec_file))

        #if USE_CUDA:
        #    self.encoder.cuda()
        #    self.decoder.cuda()

    def set_train(self, mode=True):
        self.encoder.train(mode)
        self.decoder.train(mode)

    def train_batch(self, input_batches, target_batches):
        input_batches = Variable(input_batches)
        target_batches = Variable(target_batches)

        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        #target_batches = target_batches.float()

        # Run words through encoder
        if USE_CUDA:
            input_batches = input_batches.cuda()

        input_batches = self.transform(input_batches)
        encoder_outputs, encoder_hidden = self.encoder(input_batches, hidden=None)

        decoder_outputs, decoder_hidden = self.decoder(encoder_outputs, encoder_hidden)
        predictions = self.inv_transform(decoder_outputs)

        # Move new Variables to CUDA
        if USE_CUDA:
            target_batches = target_batches.cuda()

        # Loss calculation and backpropagation
        loss = self.criterion(predictions, target_batches)

        loss.backward()

        # Clip gradient norms
        enc_clip = torch.nn.utils.clip_grad_norm(self.encoder.parameters(), self.clip)
        dec_clip = torch.nn.utils.clip_grad_norm(self.decoder.parameters(), self.clip)

        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.data[0], enc_clip, dec_clip


    def train(self, train_bb):
        self.set_train(True)
        epoch_loss = 0
        #train epoch

        tq = tqdm(range(1, self.train_batch_per_epoch+1), unit='batch')
        #for batch_cnt in range(self.train_batch_per_epoch):
        for batch_cnt in tq:
            tq.set_description('Batch %i/%i' % (batch_cnt, self.train_batch_per_epoch))
            #input_seq, _, _, target_seq = self.train_bb.build_batch()
            input_seq, _, _, target_seq = train_bb.build_batch()

            # Run the train function
            batch_loss, enc_clip, dec_clip = self.train_batch(input_seq, target_seq)
            epoch_loss += batch_loss
            tq.set_postfix(train_loss=round(epoch_loss/batch_cnt, 3), enc_clip=round(enc_clip, 4), dec_clip=round(dec_clip, 4))
        epoch_loss /= self.train_batch_per_epoch
        return epoch_loss

    def validate_batch(self, input_batches, target_batches):
        return 0

    def validate(self, validate_bb):
        self.set_train(False)
        epoch_loss = 0
        #validate epoch
        for batch_cnt in range(self.validate_batch_per_epoch):
            #input_seq, _, _, target_seq = self.validate_bb.build_batch()
            input_seq, _, _, target_seq = validate_bb.build_batch()

            batch_loss = self.validate_batch(input_seq, target_seq)
            epoch_loss += batch_loss
        epoch_loss /= self.validate_batch_per_epoch
        return epoch_loss

    def reconfig_model(self, config_file):
        with open(config_file, 'r') as f:
            pars = yaml.safe_load(f)
        active_model(self.encoder)
        self.set_optimizer(self.encoder, pars['encoder']['optimizer'])
        active_model(self.decoder)
        self.set_optimizer(self.decoder, pars['decoder']['optimizer'])

    def fit(self, TrainGen, ValGen, kwargs={}):
        #summary = torch_summarize_df((self.model.n_features, self.model.past_len), self.model)
        #self.logger.info(summary)
        #self.logger.info('total trainable parameters: {}'.format(summary['nb_params'].sum()))
        clf_dir = '../clf_attn_pos/'

        max_score = 1.0
        sb_len = 11
        kf = ''
        if 'kf' in kwargs:
            kf = 'kf{}_'.format(kwargs['kf'])
        prefix = 'kdd_' + kf

        save_freq = 0
        if 'save_freq' in kwargs:
            save_freq = int(kwargs['save_freq'])

        #tblg.configure('../output/tblog/{}'.format(self.timestamp), flush_secs=10)

        train_loss_history = deque(maxlen=self.loss_averaging_window)
        train_accuracy_history = deque(maxlen=self.loss_averaging_window)
        val_loss_history = deque(maxlen=self.loss_averaging_window // self.log_interval)
        val_accuracy_history = deque(maxlen=self.loss_averaging_window)

        step = 0
        scoreboard_file = clf_dir + 'scoreboard.pkl'
        if os.path.isfile(scoreboard_file):
            scoreboard = load_dump(scoreboard_file)
        else:
            scoreboard = []

        last_save_step = 0
        best_val_loss = 200.0
        best_val_loss_step = 0
        last_change_step = 0
        best_score = 0.0
        best_score_step = 0
        best_val_accuracy = 0.4
        best_val_accuracy_step = 0
        best_val_f1 = 0.4
        best_val_f1_step = 0

        loss_cnt = 0
        acc_cnt = 0
        f1_cnt = 0
        while step < self.n_training_steps:
            #self.logger.info(step)
            #self.kb_adjust()
            #if self.lr_scheduler and not isinstance(self.lr_scheduler, ReduceLROnPlateau):
            #    self.lr_scheduler.step()
            if self.lr_scheduler:
                self.lr_scheduler.step()

            # train step
            train_batch = next(TrainGen)
            train_fn_loss, enc_clip, dec_clip = self.train_batch(train_batch)

            if step % self.log_interval == 0:
                lr = self.lr
                if self.lr_scheduler:
                    lr = self.lr_scheduler.get_lr()[0]
                self.logger.info('lr = {}'.format(lr))
                self.tblog_value('lr', lr, step)
                # validation evaluation
                if self.with_weights:
                    val_batch, val_target, val_weight = next(ValGen)
                else:
                    val_batch = next(ValGen)
                val_fn_loss = self.validate_batch(val_batch)

                train_loss = train_fn_loss
                train_loss_history.append(train_loss)
                val_loss = val_fn_loss
                val_loss_history.append(val_loss)

                self.logger.info('\n')
                #self.logger.info('accuracy: {}, regularization_loss: {}'.format(accuracy, reg_loss))
                avg_train_loss = sum(train_loss_history) / len(train_loss_history)
                avg_val_loss = sum(val_loss_history) / len(val_loss_history)
                metric_log = (
                    "[step {:6d}]]      "
                    "[[train]]      loss: {:10.3f}     "
                    "[[val]]      loss: {:10.3f}     "
                ).format(step, round(avg_train_loss, 3), round(avg_val_loss, 3))
                #).format(step, round(avg_train_loss, 3), round(avg_val_loss, 3))
                self.logger.info(metric_log)

                self.tblog_value('train_fn_loss', train_loss, step)
                self.tblog_value('val_fn_loss', val_loss, step)

                if step > self.min_steps_to_checkpoint:
                    if avg_val_loss < best_val_loss - 0.0001:
                        best_val_loss = avg_val_loss
                        best_val_loss_step = step
                        self.logger.info('$$$$$$$$$$$$$ Best loss {} at training step {} $$$$$$$$$'.format(best_val_loss, best_val_loss_step))

                    #    model_prefix = clf_dir + prefix + self.timestamp + '_' + str(step)
                    #    self.logger.info('save to {}'.format(model_prefix))
                    #    self.save_model(model_prefix)

                    if len(scoreboard) == 0 or val_loss < scoreboard[-1][0]:
                        last_change_step = step
                        model_prefix = clf_dir + prefix + self.timestamp + '_' + str(step)
                        self.logger.info('$$$$$$$$$$$$$ Good loss {} at training step {} $$$$$$$$$'.format(val_loss, step))
                        self.logger.info('save to {}'.format(model_prefix))
                        self.save_model(model_prefix)

                        scoreboard.append([val_loss, step, self.timestamp, kwargs, model_prefix])
                        scoreboard.sort(key=lambda e: e[0], reverse=False)

                        #remove useless files
                        if len(scoreboard) > sb_len:
                            del_file = scoreboard[-1][-1]
                            tmp_file_list = glob.glob(os.path.basename(del_file))
                            for f in tmp_file_list:
                                if os.path.isfile(f):
                                    os.remove(f)

                        scoreboard = scoreboard[:sb_len]
                        save_dump(scoreboard, scoreboard_file)

                    #early stopping
                    if self.early_stopping_steps >= 0 and step - last_change_step > self.early_stopping_steps:
                        if 'hp_cnt' in kwargs:
                            self.logger.info('$$$$$$$$$$$$$ Hyper Search {} $$$$$$$$$$$$$$$$$$$$$$$'.format(kwargs['hp_cnt']))
                        self.logger.info('early stopping - ending training at {}.'.format(step))
                        break

                    # prevent overfitting
                    #if abs(avg_val_loss - avg_train_loss) / (avg_train_loss + avg_val_loss) > 0.13:
                    #    if 'hp_cnt' in hp_pars:
                    #        self.logger.info('$$$$$$$$$$$$$ Hyper Search {} $$$$$$$$$$$$$$$$$$$$$$$'.format(hp_pars['hp_cnt']))
                    #    self.logger.info('found overfitting at {}, val_loss {}, train_loss {}'.format(step, avg_val_loss, avg_train_loss))
                    #    break
            step += 1

        self.logger.info('best validation loss of {} at training step {}'.format(best_val_loss, best_val_loss_step))
        return best_val_loss


    def predict_batch(self, input_batches, predict_seq_len):
        return 0

    def predict(self, predict_bb, predict_seq_len):
        self.set_train(False)

        predict_results = []
        batch_cnt = 0
        for batch in predict_bb:
            batch_cnt += 1
            if batch_cnt % 100 == 0:
                print('Predicted %d batchs' % (batch_cnt))
            #print(batch_data.size())
            pred_batch = self.predict_batch(batch, predict_seq_len)
            predict_results.append(pred_batch.cpu().data.numpy())

        #predict_results = torch.cat(predict_results)
        #predict_results = predict_results.cpu().data.numpy()
        predict_results = np.concatenate(predict_results)
        #predict_results = predict_results.clip(0)
        return predict_results


class Seq2Seq(EncDec):
    def __init__(self, model_pars):
        super().__init__(model_pars)
        self.len_aq_bj = len(bj_stations)

        self.teacher_forcing_ratio = model_pars['teacher_forcing_ratio']

        if 'file_prefix' in model_pars:
            prefix = model_pars['file_prefix']
            self.load_model(prefix)

    def load_model(self, prefix, model_dir='../clf/'):
        enc_file = model_dir + prefix + '_enc.pth'
        dec_file = model_dir + prefix + '_dec.pth'
        self.encoder = torch.load(enc_file)
        self.decoder = torch.load(dec_file)

    def save_model(self, model_prefix):
        enc_file = model_prefix + '_enc.pth'
        dec_file = model_prefix + '_dec.pth'
        torch.save(self.encoder, enc_file)
        torch.save(self.decoder, dec_file)

    def transform(self, x):
        #x = torch.log1p(x)
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x_trans = x - x_mean
        #x_mean = self.x_mean.repeat((1, x.shape[1], 1, 1))
        return x_trans, x_mean

    def inv_transform(self, x, x_mean):
        y = x + x_mean[:, :, :x.shape[2]]
        #y = torch.expm1(y)
        return y

    def train_batch(self, batch):
        self.enable_train(True)
        with torch.set_grad_enabled(True):
            target_len = batch['dec_targets'].shape[1]
            batch_size = batch['enc_fixed'].shape[0]

            dec_targets = Variable(torch.from_numpy(batch['dec_targets'])).to(device)
            dec_targets.requires_grad_()
            dec_targets_nan = Variable(torch.from_numpy(batch['dec_targets_nan'])).to(device)
            dec_targets_nan.requires_grad_()
            dec_fixed = torch.from_numpy(batch['dec_fixed']).to(device)
            dec_fixed.requires_grad_()

            encoder_outputs, encoder_hidden, enc_dynamic_mean, emb_aqst_enc = self.encoder(batch, None)

            # Prepare input and output variables
            decoder_input = encoder_outputs[:, -1:, :]
            decoder_hidden = encoder_hidden[:self.n_dec_layers, :, :] # Use last (forward) hidden state from encoder

            predictions = torch.zeros(batch_size, target_len, self.n_out, requires_grad=True)

            predictions = predictions.to(device)
            target_batches = dec_targets.to(device)

            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
            for t in range(target_len):
                #decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                decoder_output, decoder_hidden, context, attn_weights = self.decoder(decoder_input, decoder_hidden, encoder_outputs, batch, emb_aqst_enc)

                #print(predictions[t].size(), decoder_output[0].size())
                predictions[:, t, :] = decoder_output[:, 0, :]
                if use_teacher_forcing:
                    decoder_input = torch.cat([target_batches[:, t:t+1, :], dec_fixed[:, -1, :].unsqueeze(1)], dim=2)
                else:
                    decoder_input = torch.cat([decoder_output, dec_fixed[:, -1, :].unsqueeze(1)], dim=2)      # Next input is current prediction

            # Loss calculation and backpropagation
            #predictions = predictions.permute(1, 0, 2)
            predictions = self.inv_transform(predictions, enc_dynamic_mean[:, :1, :])
            loss = self.criterion(predictions, target_batches, dec_targets_nan)

            # Zero gradients of both optimizers
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            loss.backward()
            #for param in self.decoder.parameters():
            #    print(param.grad.data.sum())

            # Clip gradient norms
            enc_clip = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
            dec_clip = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)

            # Update parameters with optimizers
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            return loss.item(), enc_clip, dec_clip

    def validate_batch(self, batch):
        self.enable_train(False)
        with torch.no_grad():
            dec_targets = (torch.from_numpy(batch['dec_targets'])).to(device)
            enc_dynamic = (torch.from_numpy(batch['enc_dynamic'])).to(device)
            enc_fixed = (torch.from_numpy(batch['enc_fixed'])).to(device)
            enc_dynamic_nan = (torch.from_numpy(batch['enc_dynamic_nan'])).to(device)
            dec_targets_nan = (torch.from_numpy(batch['dec_targets_nan'])).to(device)
            dec_fixed = torch.from_numpy(batch['dec_fixed']).to(device)
            enc_dynamic_trans, enc_dynamic_mean = self.transform(enc_dynamic)
            time_len = enc_dynamic_trans.shape[1]
            enc_dynamic_mean = enc_dynamic_mean.repeat(1, time_len, 1)
            data_batch = torch.cat([enc_fixed, enc_dynamic_trans, enc_dynamic_nan, enc_dynamic_mean], dim=2)

            encoder_outputs, encoder_hidden, enc_dynamic_mean, emb_aqst_enc = self.encoder(batch, None)

            # Prepare input and output variables
            decoder_input = encoder_outputs[:, -1:, :]
            decoder_hidden = encoder_hidden[:self.n_dec_layers, :, :] # Use last (forward) hidden state from encoder

            target_len = dec_targets.shape[1]
            batch_size = enc_dynamic.shape[0]
            predictions = torch.zeros(batch_size, target_len, self.n_out, requires_grad=False)

            predictions = predictions.to(device)
            target_batches = dec_targets.to(device)

            for t in range(target_len):
                #decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                decoder_output, decoder_hidden, context, attn_weights = self.decoder(decoder_input, decoder_hidden, encoder_outputs, batch, emb_aqst_enc)

                #print(all_decoder_outputs[t].size(), decoder_output[0].size())
                predictions[:, t, :] = decoder_output[:, 0, :]
                decoder_input = torch.cat([decoder_output, dec_fixed[:, -1, :].unsqueeze(1)], dim=2)      # Next input is current prediction

            # Loss calculation and backpropagation
            predictions = self.inv_transform(predictions, enc_dynamic_mean[:, :1, :])
            loss = self.criterion(predictions, target_batches, dec_targets_nan)
            return loss.item()

    def predict_batch(self, batch, predict_seq_len):
        self.enable_train(False)
        with torch.no_grad():
            enc_dynamic = (torch.from_numpy(batch['enc_dynamic'])).to(device)
            enc_fixed = (torch.from_numpy(batch['enc_fixed'])).to(device)
            enc_dynamic_nan = (torch.from_numpy(batch['enc_dynamic_nan'])).to(device)
            dec_fixed = torch.from_numpy(batch['dec_fixed']).to(device)
            enc_dynamic_trans, enc_dynamic_mean = self.transform(enc_dynamic)

            encoder_outputs, encoder_hidden, enc_dynamic_mean, emb_aqst_enc = self.encoder(batch, None)

            # Prepare input and output variables
            decoder_input = encoder_outputs[:, -1:, :]
            decoder_hidden = encoder_hidden[:self.n_dec_layers, :, :] # Use last (forward) hidden state from encoder

            batch_size = enc_dynamic.shape[0]
            predictions = torch.zeros(batch_size, predict_seq_len, self.n_out, requires_grad=False)
            predictions = predictions.to(device)

            for t in range(predict_seq_len):
                #decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                decoder_output, decoder_hidden, context, attn_weights = self.decoder(decoder_input, decoder_hidden, encoder_outputs, batch, emb_aqst_enc)

                #print(all_decoder_outputs[t].size(), decoder_output[0].size())
                predictions[:, t, :] = decoder_output[:, 0, :]
                decoder_input = torch.cat([decoder_output, dec_fixed[:, -1, :].unsqueeze(1)], dim=2)      # Next input is current prediction

            # Loss calculation and backpropagation
            predictions = self.inv_transform(predictions, enc_dynamic_mean[:, :1, :])
            return predictions

    def predict(self, predict_bb, predict_seq_len):
        predict_results = super().predict(predict_bb, predict_seq_len)
        return predict_results.clip(0)



