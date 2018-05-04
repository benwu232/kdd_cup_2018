import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import *

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


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 16, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(2, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x