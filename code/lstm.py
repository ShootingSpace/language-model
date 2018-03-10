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

'''Define classes for indexing words of the input document,
   The Dictionary class is used by the Corpus class to
   index the words of the input document.
'''
class Corpus(object):
    def __init__(self, vocab_size=2000, train_size=1000, dev_size=1000, path):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = vocab_size
        self.train_size = train_size
        self.dev_size = dev_size
        self.vocab = self.build_vocab(path + 'vocab.wiki.txt')
        self.X_train, self.D_train = self.tokenize(path + 'wiki-train.txt', self.train_size)
        self.X_dev, self.D_dev = self.tokenize(path + 'wiki-dev.txt', self.dev_size)
        self.X_test, self.D_test = self.tokenize(path + 'wiki-test.txt')

    def build_vocab(self, path):
        '''build the vocabulary from given file'''
        vocab = pd.read_table(path + "/vocab.wiki.txt", header=None, sep="\s+", index_col=0, names=['count', 'freq'], )
        self.idx2word = dict(enumerate(vocab.index[:self.vocab_size]))
        self.word2idx = invert_dict(self.idx2word)

    def tokenize(self, path, size=None):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Loads the context sentence by sentence
        corpus = load_lm_dataset(path)
        context = docs_to_indices(corpus, self.word2idx, 1, 1)
        X, D = seqs_to_lmXY(context)
        X, D = mx.nd.array(X, dtype='int32'), mx.nd.array(D, dtype='int32')
        if size:
            return X[:size], D[:size]
        else:
            return X, D
