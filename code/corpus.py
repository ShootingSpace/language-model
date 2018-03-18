# coding: utf-8
import sys
import time
import numpy as np
from sys import stdout
import itertools
import os
from os.path import join as pjoin
from tempfile import TemporaryFile
import logging
from utils import *
from rnnmath import *

'''Define classes for indexing words of the input document
'''
class Corpus(object):
    def __init__(self, vocab_size, vocab_file):
        self.vocab_size = vocab_size
        self.word2idx, self.idx2word = self.build_vocab(vocab_file)

    def build_vocab(self, vocab_file):
        '''build the vocabulary from given file'''
        vocab = pd.read_table(vocab_file, header=None, sep="\s+", index_col=0, names=['count', 'freq'], )
        idx2word = dict(enumerate(vocab.index[:self.vocab_size]))
        word2idx = invert_dict(idx2word)
        return word2idx, idx2word

    def tokenize(self, path, sequence_length, size=None):
        """Tokenizes a text file.
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
        X_D = self.seqs_to_lmXY(corpus_indexing)

        # corpus_indexing = self.make_fix_length_sequence(corpus_indexing[0], sequence_length)
        return X_D

    def seqs_to_lmXY(self, seqs):
        ''' list of tuple'''
        # X, Y = zip(*[self.offset_seq(s) for s in seqs])
        data_label = [self.offset_seq(s) for s in seqs]
        return data_label
        # return np.array(X, dtype=object), np.array(Y, dtype=object)

    def offset_seq(self, seq):
        # one_hot_vectors = mx.ndarray.one_hot(mx.nd.array(seq), self.vocab_size)
        return (seq[:-1], seq[1:])

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
