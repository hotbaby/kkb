# encoding:utf8

import torch
import torch.nn as nn


class RNNMLClassification(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNMLClassification, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.o2o(torch.cat((hidden, output), 1))
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


class GRU(nn.Module):
    def __init__(self, vocab_size, embb_size, hidden_size, output_size, n_layers=1):
        super(GRU, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(vocab_size, embb_size)
        self.rnn = nn.GRU(embb_size, hidden_size, n_layers)

        self.decoder = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.rnn(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self,):
        return torch.randn(self.n_layers, 1, self.hidden_size)


class LSTM(nn.Module):
    def __init__(self, vocab_size, embb_size, hidden_size, output_size, n_layers=1):
        super(LSTM, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(vocab_size, embb_size)
        self.rnn = nn.LSTM(embb_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        ouput, hidden = self.rnn(input.view(1, 1, -1), hidden)
        ouput = self.decoder(ouput.view(1, -1))
        ouput = self.softmax(ouput)
        return ouput, hidden

    def init_hidden(self,):
        return (torch.randn(self.n_layers, 1, self.hidden_size),
                torch.randn(self.n_layers, 1, self.hidden_size))


class StackLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_1_size, hidden_2_size, output_size, n_layers=1):
        super(StackLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_1_size = hidden_1_size
        self.hidden_2_size = hidden_2_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encode = nn.Embedding(vocab_size, embedding_dim)
        self.rnn1 = nn.LSTM(embedding_dim, hidden_1_size, n_layers)
        self.rnn2 = nn.LSTM(hidden_1_size, hidden_2_size, n_layers)
        self.decode = nn.Linear(hidden_2_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden):
        input = self.encode(input.view(1, -1))
        output, hidden1 = self.rnn1(input.view(1, 1, -1), hidden[0])
        output, hidden2 = self.rnn2(output, hidden[1])
        output = self.decode(output)
        output = self.softmax(output)
        return output, [hidden1, hidden2]

    def init_hidden(self):
        return [
            (torch.zeros(self.n_layers, 1, self.hidden_1_size),
             torch.zeros(self.n_layers, 1, self.hidden_1_size)),

            (torch.zeros(self.n_layers, 1, self.hidden_2_size),
             torch.zeros(self.n_layers, 1, self.hidden_2_size))
        ]
