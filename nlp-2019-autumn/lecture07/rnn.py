# encoding:utf8

import torch
import torch.nn as nn


class RNNMLClassification(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNMLClassification, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.h2h = nn.Linear(hidden_size, 64)
        self.h2o = nn.Linear(64, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        # output = self.i2o(combined)
        # hidden = self.h2h(hidden)
        output = self.h2o(self.h2h(hidden))
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


class LSTMClassification(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
                 dropout=0.2):
        super(LSTMClassification, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        output = self.h2o(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
