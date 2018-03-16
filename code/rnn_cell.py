"""Definition of various recurrent neural network cells."""
import math
import time
import pandas as pd
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn, rnn

''' Provide an exposition of different RNN models with gluon
    Based on the gluon.Block class.
'''
class RNN(gluon.Block):
    def __init__(self, mode, seed, vocab_size, num_embed, num_hidden,
                 num_layers, dropout, **kwargs):
        super(RNN, self).__init__(**kwargs)
        if seed:
            mx.random.seed(seed)

        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            # self.encoder = nn.Embedding(vocab_size, num_embed,
            #                             weight_initializer = mx.init.Uniform(0.1))
            if mode == 'rnn_relu':
                self.rnn = rnn.RNN(num_hidden, num_layers, activation='relu', dropout=dropout,
                                   input_size=num_embed)
            elif mode == 'rnn_tanh':
                self.rnn = rnn.RNN(num_hidden, num_layers, dropout=dropout,
                                   input_size=num_embed)
            elif mode == 'lstm':
                self.rnn = rnn.LSTM(num_hidden, num_layers, dropout=dropout,
                                    input_size=num_embed)
            elif mode == 'gru':
                self.rnn = rnn.GRU(num_hidden, num_layers, dropout=dropout,
                                   input_size=num_embed)
            else:
                raise ValueError("Invalid mode %s. Options are rnn_relu, "
                                 "rnn_tanh, lstm, and gru"%mode)
            self.decoder = nn.Dense(vocab_size, in_units = num_hidden)
            self.num_hidden = num_hidden

    def forward(self, inputs, hidden):
        with inputs.context:
            output, hidden = self.rnn(inputs, hidden)
            decoded = self.decoder(output.reshape((-1, self.num_hidden)))
            return decoded, hidden

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
