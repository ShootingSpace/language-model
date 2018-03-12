import argparse
import time
import math
import mxnet as mx
from mxnet import gluon, autograd
from lstm import RNNModel, Corpus
import numpy as np
import bisect
parser = argparse.ArgumentParser(description='MXNet Autograd RNN/LSTM Language Model.')
parser.add_argument('--model', type=str, default='lstm',
                    help='type of recurrent net (rnn_tanh, rnn_relu, lstm, gru)')
parser.add_argument('--data_folder', type=str, default='data/', help='folder that store data')
parser.add_argument('--vocab_size', type=int, default=2000, help='vocab_size')
parser.add_argument('--train_size', type=int, default=1000, help='train_size')
parser.add_argument('--dev_size', type=int, default=1000, help='dev_size')
parser.add_argument('--num_embed', type=int, default=2000, help='size of word embeddings')
parser.add_argument('--num_hidden', type=int, default=50, help='number of hidden units per layer')
parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
parser.add_argument('--lr', type=float, default=0.5, help='initial learning rate')
parser.add_argument('--anneal', type=float, default=5,
                    help='if > 0, lowers the learning rate in a harmonically \
                    after each epoch')
parser.add_argument('--clip', type=float, default=0.2, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=30, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size')
parser.add_argument('--bptt', type=int, default=0, help='sequence length')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--cuda', action='store_true', help='Whether to use gpu')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.params',
                    help='path to save the final model')
parser.add_argument('--gctype', type=str, default='none',
                    help='type of gradient compression to use, takes `2bit` or `none` for now.')
parser.add_argument('--gcthreshold', type=float, default=0.5,
                    help='threshold for 2bit gradient compression')
args = parser.parse_args()


'''Train the model and evaluate on validation and testing data sets
helper functions that will be used during model training and evaluation.'''
def get_batch(source, i):
    pass

def detach(hidden):
    pass

def eval(data_source):
    pass

def train():
    stack = mx.rnn.SequentialRNNCell()
    for i in range(args.num_layers):
        stack.add(mx.rnn.LSTMCell(num_hidden=args.num_hidden, prefix='lstm_l%d_'%i))

def sym_gen(seq_len):
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('softmax_label')
    embed = mx.sym.Embedding(data=data, input_dim=args.vocab_size,
                             output_dim=args.num_embed, name='embed')

    stack.reset()
    outputs, states = stack.unroll(seq_len, inputs=embed, merge_outputs=True)

    pred = mx.sym.Reshape(outputs, shape=(-1, args.num_hidden))
    pred = mx.sym.FullyConnected(data=pred, num_hidden=args.num_embed, name='pred')

    label = mx.sym.Reshape(label, shape=(-1,))
    pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

    return pred, ('data',), ('softmax_label',)

if __name__ == '__main__':
    if args.cuda:
        context = mx.gpu(0)
    else:
        context = mx.cpu(0)
    ###############################################################################
    # Load data
    ###############################################################################
    '''To speed up the subsequent data flow in the RNN model,
       pre-process the loaded data as batches.
       This procedure is defined in the following batchify function.
    '''
    corpus = Corpus(vocab_size=args.vocab_size, vocab_file = args.data_folder+ 'vocab.wiki.txt')

    '''prepare data, support buckets batches
    '''
    # buckets for batch training
    # buckets = [ 10, 20, 30, 40, 50, 60]
    train_corpus_indexing = corpus.tokenize(args.data_folder + 'wiki-train.txt', args.train_size)
    dev_corpus_indexing   = corpus.tokenize(args.data_folder + 'wiki-dev.txt', args.dev_size)
    test_corpus_indexing   = corpus.tokenize(args.data_folder + 'wiki-test.txt')

    train_data  = mx.rnn.BucketSentenceIter(train_corpus_indexing, args.batch_size)
    dev_data    = mx.rnn.BucketSentenceIter(dev_corpus_indexing, args.batch_size)
    test_data    = mx.rnn.BucketSentenceIter(test_corpus_indexing, args.batch_size)

    '''Build the model, initialize model parameters,
    and configure the optimization algorithms for training the RNN model.'''
     model = mx.mod.BucketingModule(
        sym_gen             = sym_gen,
        default_bucket_key  = data_train.default_bucket_key,
        context             = contexts)
