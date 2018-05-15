import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.define import *

def transform(x):
    #x = torch.log1p(x)
    x_mean = torch.mean(x, dim=1, keepdim=True)
    x_trans = x - x_mean
    #x_mean = self.x_mean.repeat((1, x.shape[1], 1, 1))
    return x_trans, x_mean


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1, bidirectional=False):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.emb_aqst_enc = nn.Embedding(len(aq_stations), 3)
        self.emb_weather_enc = nn.Embedding(len(whether_list), 2)
        #self.conv1 = ResBac(self.input_size, self.hidden_size, kernel_size=3, stride=1, using_bn=False, padding=1, res=False)
        self.rnn1 = nn.GRU(input_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=self.bidirectional, batch_first=True)

    def forward(self, batch, hidden=None):
        with torch.set_grad_enabled(True):
            enc_dynamic = torch.from_numpy(batch['enc_dynamic']).to(device)
            enc_dynamic.requires_grad_()
            enc_dynamic_nan = torch.from_numpy(batch['enc_dynamic_nan']).to(device)
            enc_dynamic_nan.requires_grad_()
            enc_fixed = torch.from_numpy(batch['enc_fixed']).to(device)
            enc_fixed.requires_grad_()
            emb_station_id = torch.from_numpy(batch['enc_emb'][:, :, 0]).to(device)
            emb_station_id.requires_grad_()
            emb_weather = torch.from_numpy(batch['enc_emb'][:, :, 1]).to(device)
            emb_weather.requires_grad_()

            enc_dynamic_trans, enc_dynamic_mean = transform(enc_dynamic)
            time_len = enc_dynamic_trans.shape[1]
            enc_dynamic_mean = enc_dynamic_mean.repeat(1, time_len, 1)

            emb_aqst = self.emb_aqst_enc(emb_station_id.long())
            emb_weather = self.emb_weather_enc(emb_weather.long())
            input_seqs = torch.cat([enc_fixed, enc_dynamic_trans, enc_dynamic_nan, enc_dynamic_mean, emb_aqst, emb_weather], dim=2)
            outputs, hidden = self.rnn1(input_seqs, hidden)
            if self.bidirectional:
                outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:] # Sum bidirectional outputs
            return outputs, hidden, enc_dynamic_mean, self.emb_aqst_enc, self.emb_weather_enc


class EncoderRNN2(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1, bidirectional=False):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        #self.conv1 = ResBac(self.input_size, self.hidden_size, kernel_size=3, stride=1, using_bn=False, padding=1, res=False)
        self.emb_aqst_enc = nn.Embedding(len(aq_stations), 3)
        self.emb_weather_enc = nn.Embedding(len(whether_list), 2)
        self.rnn1 = nn.GRU(input_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=self.bidirectional, batch_first=True)

    def forward(self, input_seqs, hidden=None):
        emb_aqst = self.emb_aqst_enc(input_seqs[:, :, -1].long())
        input_seqs = torch.cat([input_seqs[:, :, :-1], emb_aqst], dim=2)
        outputs, hidden = self.rnn1(input_seqs, hidden)
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:] # Sum bidirectional outputs
        return outputs, hidden

class EncoderRNN1(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1, bidirectional=False):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        #self.conv1 = ResBac(self.input_size, self.hidden_size, kernel_size=3, stride=1, using_bn=False, padding=1, res=False)
        self.rnn1 = nn.GRU(input_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=self.bidirectional, batch_first=True)

    def forward(self, input_seqs, hidden=None):
        outputs, hidden = self.rnn1(input_seqs, hidden)
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:] # Sum bidirectional outputs
        return outputs, hidden


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout=0.1, bidirectional=False):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_direction = 1
        if self.bidirectional == True:
            self.num_direction = 2

        #self.conv1 = ResBac(self.input_size, self.hidden_size, kernel_size=3, stride=1, using_bn=False, padding=1, res=False)
        self.rnn1 = nn.GRU(self.input_size, self.hidden_size, n_layers, dropout=self.dropout, bidirectional=self.bidirectional, batch_first=True)
        self.rnn2 = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=self.bidirectional, batch_first=True)
        self.fc = nn.Linear(self.hidden_size * self.num_direction, self.output_size)
        self.dropout_layer = nn.Dropout(0.5)

    def forward(self, x, h=None):
        #print(input_seqs.size(), hidden.size())
        if x.shape[-1] != self.hidden_size:
            x, h = self.rnn1(x, h)
            if self.bidirectional:
                x = x[:, :, :self.hidden_size] + x[:, :, self.hidden_size:] # Sum bidirectional outputs
        #x = x.contiguous()
        h = h.contiguous()
        x, h = self.rnn2(x, h)
        #if self.bidirectional:
        #    x = x[:, :, :self.hidden_size] + x[:, :, self.hidden_size:] # Sum bidirectional outputs
        #x, h = self.rnn3(x, h)
        x = self.dropout_layer(x)
        x = self.fc(x)
        #outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
        return x, h


############################ Attention model ############################################################
class TimeAttn(nn.Module):
    def __init__(self, method, hidden_size):
        super(TimeAttn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        batch_size = encoder_outputs.size(0)
#         print('[attn] seq len', seq_len)
#         print('[attn] encoder_outputs', encoder_outputs.size()) # S x B x N
#         print('[attn] hidden', hidden.size()) # S=1 x B x N

        # Create variable to store attention energies
        attn_energies = torch.zeros(batch_size, seq_len, ).to(device) # B x S

        # For each batch of encoder outputs
        #for k in range(seq_len-1, 0, -24):
        #for k in range(seq_len-1, seq_len-7*24, -3):
        #for k in range(seq_len-7*24, seq_len):
            #attn_energies[:, k] = self.cal_energy_batch(hidden[:, 0], encoder_outputs[:, k])
        #    attn_energies[:, k] = self.cal_energy_batch(hidden, encoder_outputs[:, k])

        #for k in range(seq_len-24, seq_len):
        #    attn_energies[:, k] = self.cal_energy_batch(hidden[:, 0], encoder_outputs[:, k])

        #for k in range(seq_len-240, seq_len):
        for k in range(seq_len):
            attn_energies[:, k] = self.cal_energy_batch(hidden, encoder_outputs[:, k])

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        attn_score = F.softmax(attn_energies, dim=1)
        #attn_score = F.softmax(attn_energies).unsqueeze(1)
        return attn_score

    def cal_energy_batch(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            #energy = torch.bmm(energy.unsqueeze(1), hidden.permute(1, 2, 0))[:, 0, 0]
            energy = torch.bmm(energy.unsqueeze(1), hidden.permute(1, 2, 0))#[:, 0, 0]
            return energy.sum(dim=2).sum(dim=1)

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.dot(energy)
            return energy

    def cal_energy(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.dot(energy)
            return energy


class SpaceAttn(TimeAttn):
    def __init__(self, method, input_size, hidden_size):
        super().__init__(method, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.input_size, hidden_size)

    def forward(self, hidden, batch, emb_aqst_enc, emb_weather_enc):
        with torch.set_grad_enabled(True):
            enc_dynamic_all = torch.from_numpy(batch['enc_dynamic_all']).to(device)
            enc_dynamic_all.requires_grad_()
            enc_dynamic_nan_all = torch.from_numpy(batch['enc_dynamic_nan_all']).to(device)
            enc_dynamic_nan_all.requires_grad_()
            enc_fixed_all = torch.from_numpy(batch['enc_fixed_all']).to(device)
            enc_fixed_all.requires_grad_()
            enc_emb_all = torch.from_numpy(batch['enc_emb_all']).to(device)
            enc_emb_all.requires_grad_()
            seq_len = enc_fixed_all.shape[1]
            batch_size = enc_fixed_all.shape[0]

            enc_emb_all = enc_emb_all.squeeze(2)
            emb_aqst = emb_aqst_enc(enc_emb_all[:, :, 0, :].long())
            emb_aqst = emb_aqst.permute(0, 1, 3, 2)
            emb_weather = emb_weather_enc(enc_emb_all[:, :, 1, :].long())
            emb_weather = emb_weather.permute(0, 1, 3, 2)
            enc_dynamic_trans_all, enc_dynamic_mean_all = transform(enc_dynamic_all)
            enc_dynamic_mean_all = enc_dynamic_mean_all.repeat(1, seq_len, 1, 1)
            enc_batch = torch.cat([enc_fixed_all, enc_dynamic_trans_all, enc_dynamic_nan_all, enc_dynamic_mean_all, emb_aqst, emb_weather], dim=2)
            feature_size = enc_batch.shape[2]


            # Create variable to store attention energies
            attn_scores = torch.zeros(batch_size, seq_len*35).to(device) # B x S
            attn_energies = torch.zeros(batch_size, seq_len*35, self.hidden_size).to(device) # B x S

            for k in range(seq_len):
                for st_idx in range(35):
                    attn_scores[:, k*35+st_idx], attn_energies[:, k*35+st_idx, :] = self.cal_energy_batch(hidden, enc_batch[:, k, :, st_idx])

            # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
            attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(1)

            #enc_batch = enc_batch.permute(0, 1, 3, 2)
            #enc_batch = enc_batch.contiguous()
            #enc_batch = enc_batch.view(batch_size, -1, feature_size)

            space_context = attn_weights.bmm(attn_energies)
            return space_context

    def cal_energy_batch(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy0 = self.attn(encoder_output)
            #energy = torch.bmm(energy.unsqueeze(1), hidden.permute(1, 2, 0))[:, 0, 0]
            energy = torch.bmm(energy0.unsqueeze(1), hidden.permute(1, 2, 0))#[:, 0, 0]
            return energy.sum(dim=2).sum(dim=1), energy0

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.dot(energy)
            return energy



class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, input_size, hidden_size, output_size, n_layers=1, dropout_p=0., bidirectional=False, n_enc_input=29):
        super().__init__()

        # Keep parameters for reference
        self.attn_model = attn_model
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.bidirectional = bidirectional
        self.num_direction = 1
        if self.bidirectional == True:
            self.num_direction = 2
        self.n_enc_input = n_enc_input

        # Define layers
        self.rnn1 = nn.GRU(self.input_size+self.hidden_size*2, hidden_size, n_layers, dropout=dropout_p, bidirectional=self.bidirectional, batch_first=True)
        self.rnn2 = nn.GRU(hidden_size*3, hidden_size, n_layers, dropout=dropout_p, bidirectional=self.bidirectional, batch_first=True)
        self.out = nn.Linear(hidden_size * 2, output_size)
        self.fc = nn.Linear(self.hidden_size * self.num_direction, self.output_size)

        # Choose attention model
        if attn_model != 'none':
            self.time_attn = TimeAttn(attn_model, hidden_size)
            self.space_attn = SpaceAttn(attn_model, n_enc_input, hidden_size)

    def forward(self, decoder_input, last_hidden, encoder_outputs, batch, emb_aqst_enc, emb_weather_enc):
        # Note: we run this one step at a time
        with torch.set_grad_enabled(True):
            enc_dynamic_all = torch.from_numpy(batch['enc_dynamic_all']).to(device)
            enc_dynamic_all.requires_grad_()
            enc_dynamic_nan_all = torch.from_numpy(batch['enc_dynamic_nan_all']).to(device)
            enc_dynamic_nan_all.requires_grad_()
            enc_fixed_all = torch.from_numpy(batch['enc_fixed_all']).to(device)
            enc_fixed_all.requires_grad_()
            enc_emb_all = torch.from_numpy(batch['enc_emb_all']).to(device)
            enc_emb_all.requires_grad_()

            dec_targets = torch.from_numpy(batch['dec_targets']).to(device)
            dec_targets.requires_grad_()
            dec_targets_nan = torch.from_numpy(batch['dec_targets_nan']).to(device)
            dec_targets_nan.requires_grad_()
            dec_fixed = torch.from_numpy(batch['dec_fixed']).to(device)
            dec_fixed.requires_grad_()

            # Calculate space attention weights
            space_context = self.space_attn(last_hidden, batch, emb_aqst_enc, emb_weather_enc)

            # Calculate time attention weights and apply to encoder outputs
            time_attn_weights = self.time_attn(last_hidden, encoder_outputs)
            time_attn_weights = time_attn_weights.unsqueeze(1)
            time_context = time_attn_weights.bmm(encoder_outputs) # B x 1 x F

            # Combine embedded input word and attended context, run through RNN
            input_seq = torch.cat((decoder_input, time_context, space_context), 2)
            if input_seq.shape[-1] != self.hidden_size*3:
                input_seq, hidden = self.rnn1(input_seq, last_hidden)
                if self.bidirectional:
                    input_seq = input_seq[:, :, :self.hidden_size] + input_seq[:, :, self.hidden_size:] # Sum bidirectional outputs
                rnn_output = input_seq
            else:
                rnn_output, hidden = self.rnn2(input_seq, last_hidden)

            # Final output layer
            output = self.fc(rnn_output)

            # Return final output, hidden state, and attention weights (for visualization)
            return output, hidden, time_context, time_attn_weights


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, input_size, hidden_size, output_size, n_layers=1, dropout_p=0.1, bidirectional=False):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout_p
        self.bidirectional = bidirectional
        self.num_direction = 1
        if self.bidirectional == True:
            self.num_direction = 2
        self.dropout_layer = nn.Dropout(0.5)

        # Define layers
        self.rnn1 = nn.GRU(self.input_size, hidden_size, n_layers, dropout=dropout_p, bidirectional=self.bidirectional, batch_first=True)
        self.rnn2 = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p, bidirectional=self.bidirectional, batch_first=True)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.fc = nn.Linear(self.hidden_size * self.num_direction, self.output_size)
        self.dropout_layer = nn.Dropout(0.5)

        # Choose attention model
        if attn_model != 'none':
            self.attn = TimeAttn(attn_model, hidden_size)

    def forward1(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time (in order to do teacher forcing)
        if input_seq.shape[-1] != self.hidden_size:
            input_seq, last_hidden = self.rnn1(input_seq, last_hidden)
            if self.bidirectional:
                input_seq = input_seq[:, :, :self.hidden_size] + input_seq[:, :, self.hidden_size:] # Sum bidirectional outputs

        rnn_output, hidden = self.rnn2(input_seq, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        #attn_weights = self.attn(rnn_output, encoder_outputs)
        #context = attn_weights.unsqueeze(dim=1).bmm(encoder_outputs) # B x S=1 x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        #rnn_output = rnn_output.squeeze(1) # S=1 x B x N -> B x N
        #context = context.squeeze(1)       # B x S=1 x N -> B x N
        #concat_input = torch.cat((rnn_output, context), 1)
        #concat_output = F.tanh(self.concat(concat_input))

        rnn_output = self.dropout_layer(rnn_output)
        # Finally predict next token (Luong eq. 6)
        output = self.out(rnn_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, None, None

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time (in order to do teacher forcing)

        if input_seq.shape[-1] != self.hidden_size:
            input_seq, last_hidden = self.rnn1(input_seq, last_hidden)
            if self.bidirectional:
                input_seq = input_seq[:, :, :self.hidden_size] + input_seq[:, :, self.hidden_size:] # Sum bidirectional outputs

        rnn_output, hidden = self.rnn2(input_seq, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.unsqueeze(dim=1).bmm(encoder_outputs) # B x S=1 x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(1) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
#         print('[decoder] rnn_output', rnn_output.size())
#         print('[decoder] context', context.size())
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))
        concat_output = self.dropout_layer(concat_output)

        # Finally predict next token (Luong eq. 6)
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output.unsqueeze(dim=1), hidden, context, attn_weights






class ResNet2D(nn.Module):
    def __init__(self, block, layers, num_classes=500):
        self.inplanes = 32
        super().__init__()
        self.conv1 = nn.Conv2d(14, self.inplanes, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 6, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 2, layers[2], stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x_out = []
        for k in range(x.shape[2]):
            y = x[:, :, k, :, :]
            y = self.conv1(y)
            y = self.bn1(y)
            y = self.relu(y)
            y = self.layer1(y)
            y = self.layer2(y)
            y = self.layer3(y)
            y = y.unsqueeze(2)
            x_out.append(y)
        y = torch.cat(x_out, dim=2)
        y = y.permute((0, 2, 1, 3, 4)).contiguous()
        return y.view(y.size(0), y.size(1), -1)

def grid_res2d(**kwargs):
    model = ResNet2D(BasicBlock, [2, 2, 2], **kwargs)
    return model



