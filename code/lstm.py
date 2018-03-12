# coding: utf-8
import sys
import time
import numpy as np

from utils import *
from rnnmath import *
from sys import stdout
import itertools
import os
from os.path import join as pjoin
from tempfile import TemporaryFile
import logging
logging.basicConfig(level=logging.INFO)

import math
import time
import pandas as pd
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn

'''Define classes for indexing words of the input document
'''
class Corpus(object):
    def __init__(self, vocab_size, vocab_file):
        self.vocab_size = vocab_size
        self.word2idx, self.idx2word = self.build_vocab(vocab_file)

        # self.train_size = train_size
        # self.dev_size = dev_size
        # self.vocab = self.build_vocab(path + 'vocab.wiki.txt')
        # self.X_train, self.D_train = self.tokenize(path + 'wiki-train.txt', self.train_size)
        # self.X_dev, self.D_dev = self.tokenize(path + 'wiki-dev.txt', self.dev_size)
        # self.X_test, self.D_test = self.tokenize(path + 'wiki-test.txt')

    def build_vocab(self, vocab_file):
        '''build the vocabulary from given file'''
        vocab = pd.read_table(vocab_file, header=None, sep="\s+", index_col=0, names=['count', 'freq'], )
        idx2word = dict(enumerate(vocab.index[:self.vocab_size]))
        word2idx = invert_dict(idx2word)
        return word2idx, idx2word

    def tokenize(self, path, sequence_length, size=None):
        """Tokenizes a text file.
           return one hot representation of each token
        """
        assert os.path.exists(path)
        # Loads the context sentence by sentence
        if size:
            corpus = self.load_dataset(path, size) # (list of list of str) string tokens.
        else:
            corpus = self.load_dataset(path)
        corpus_indexing, _ = self.encode_sentences(corpus, vocab=self.word2idx,
                                                     unknown_token='UNK')

        print('Load {} data'.format(path))
        X_train, D_train = self.seqs_to_lmXY(corpus_indexing)
        print('Complete one hot making')

        # corpus_indexing = self.make_fix_length_sequence(corpus_indexing[0], sequence_length)
        return X_train, D_train

    def seqs_to_lmXY(self, seqs):
        print('Make X and D in one hot representation')
        X, Y = zip(*[self.offset_seq(s) for s in seqs])
        return np.array(X, dtype=object), np.array(Y, dtype=object)

    def offset_seq(self, seq):
        '''return one hot  '''
        one_hot_vectors = mx.ndarray.one_hot(mx.nd.array(seq), self.vocab_size)
        return one_hot_vectors[:-1], one_hot_vectors[1:]

    def load_dataset(self, fname, size=None):
        corpus = []
        cnt = 0
        with open(fname) as f:
            for line in f:
                if cnt == 0:
                    cnt += 1
                    continue
                items = line.strip().split('\t')
                # corpus += self.pad_sequence(items[0].split(), left=1, right=1)
                corpus.append(items[0].split())
                if size and cnt == size:
                    break
                cnt += 1

        return corpus

    def make_fix_length_sequence(self, corpus, sequence_length):
        '''transfer the whole sequence into matrix'''
        mod = len(corpus) % sequence_length
        rows = int(len(corpus)/sequence_length+1)
        corpus += [-1]*(sequence_length-mod)
        corpus = np.reshape(corpus, (rows, sequence_length))
        assert corpus.size == rows*sequence_length
        return corpus

    def pad_sequence(self, seq, left=1, right=1):
        return left*["<s>"] + seq + right*["</s>"] #+ ['\n']

    def encode_sentences(self, sentences, vocab=None, invalid_label=-1,
                         invalid_key='\n', start_label=0, unknown_token=None) :
        """Encode sentences and (optionally) build a mapping
        from string tokens to integer indices. Unknown keys
        will be added to vocabulary.

        Parameters
        ----------
        sentences : list of list of str
            A list of sentences to encode. Each sentence
            should be a list of string tokens.
        vocab : None or dict of str -> int
            Optional input Vocabulary
        invalid_label : int, default -1
            Index for invalid token, like <end-of-sentence>
        invalid_key : str, default '\\n'
            Key for invalid token. Use '\\n' for end
            of sentence by default.
        start_label : int
            lowest index.

        Returns
        -------
        result : list of list of int
            encoded sentences
        vocab : dict of str -> int
            result vocabulary
        """
        idx = start_label
        if vocab is None:
            vocab = {invalid_key: invalid_label}
            new_vocab = True
        elif unknown_token:
            new_vocab = True
        else:
            new_vocab = False
        res = []

        for sent in sentences:
            coded = []
            for word in sent:
                if word not in vocab:
                    assert new_vocab, "Unknown token %s"%word
                    if idx == invalid_label:
                        idx += 1
                    if unknown_token:
                        word = unknown_token
                    vocab[word] = idx
                    idx += 1
                coded.append(vocab[word])
            res.append(coded)
        return res, vocab
        # context = docs_to_indices(corpus, self.word2idx, 1, 1)
        # X, D = seqs_to_lmXY(context)
        # X, D = mx.nd.array(X, dtype=object), mx.nd.array(D, dtype=object)
        # if size:
        #     return X[:size], D[:size]
        # else:
        #     return X, D

''' Provide an exposition of different RNN models with gluon
    Based on the gluon.Block class.
'''
class RNNModel(gluon.Block):
    """A model with
    an encoder: if num_embed = vocab_size, then it is one hot encoding
    recurrent layer,
    a decoder.
    """
    def __init__(self, mode, vocab_size, num_embed, num_hidden,
                 num_layers, dropout, tie_weights, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(vocab_size, num_embed,
                                        weight_initializer = mx.init.Uniform(0.1))
            if mode == 'rnn_relu':
                self.rnn = rnn.RNN(num_hidden, num_layers, activation='relu',
                                   dropout=dropout,
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
            if tie_weights:
                self.decoder = nn.Dense(vocab_size, in_units = num_hidden,
                                        params = self.encoder.params)
            else:
                self.decoder = nn.Dense(vocab_size, in_units = num_hidden)
            self.num_hidden = num_hidden

    def forward(self, inputs, hidden):
        emb = self.drop(self.encoder(inputs))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.reshape((-1, self.num_hidden)))
        return decoded, hidden

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
