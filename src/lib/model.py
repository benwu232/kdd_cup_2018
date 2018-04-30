import torch.nn as nn
import torch.nn.functional as F


class EncoderRnn(nn.Module):
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


class DecoderRnn(nn.Module):
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

