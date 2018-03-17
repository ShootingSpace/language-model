import argparse
import time
import math
import mxnet as mx
from mxnet import gluon, autograd, nd
from lstm import RNNModel, Corpus
import numpy as np
import bisect
import random
import logging
import os
from os.path import join as pjoin
from tqdm import tqdm
from mxnet.gluon import nn, rnn
from corpus import Corpus
from gru import GRU
import pandas as pd
from utils import *
from rnnmath import *


parser = argparse.ArgumentParser(description='MXNet Autograd RNN/LSTM Language Model.')
parser.add_argument('--model', type=str, default='gru',
                    help='type of recurrent net (rnn_tanh, rnn_relu, lstm, gru)')
parser.add_argument('--data_folder', type=str, default='data/', help='folder that store data')
parser.add_argument('--output_dir', type=str, default='output', help='log directory')
parser.add_argument('--vocab_size', type=int, default=2000, help='vocab_size')
parser.add_argument('--sequence_length', type=int, default=60, help='sequence length for each inputs')
parser.add_argument('--train_size', type=int, default=1000, help='train_size')
parser.add_argument('--dev_size', type=int, default=1000, help='dev_size')
parser.add_argument('--test_size', type=int, default=None, help='test_size')
parser.add_argument('--num_embed', type=int, default=2000, help='size of word embeddings')
parser.add_argument('--num_hidden', type=int, default=50, help='number of hidden units per layer')
parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
parser.add_argument('--lr', type=float, default=0.5, help='initial learning rate')
parser.add_argument('--anneal', type=float, default=5,
                    help='if > 0, lowers the learning rate in a harmonically \
                    after each epoch')
parser.add_argument('--clip', type=float, default=0.2, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=30, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='batch size')
parser.add_argument('--acc_grad_size', type=int, default=50, metavar='N', help='accumulate grad size')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--cuda', action='store_true', help='Whether to use gpu')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.params',
                    help='path to save the final model')
parser.add_argument('--gctype', type=str, default='none',
                    help='type of gradient compression to use, takes `2bit` or `none` for now.')
parser.add_argument('--gcthreshold', type=float, default=0.5,
                    help='threshold for 2bit gradient compression')
args = parser.parse_args()

def softmax(y_linear, temperature=1.0):
    lin = (y_linear-nd.max(y_linear)) / temperature
    exp = nd.exp(lin)
    partition = nd.sum(exp, axis=0, exclude=True).reshape((-1,1))
    return exp / partition

def cross_entropy(yhat, y):
    return - nd.mean(nd.sum(y * nd.log(yhat), axis=0, exclude=True))

def average_ce_loss(outputs, labels):
    '''Averaging the loss over the sequence'''
    assert(len(outputs) == len(labels))
    total_loss = nd.array([0.], ctx=context)
    for (output, label) in zip(outputs,labels):
        total_loss = total_loss + cross_entropy(output, label)
    return total_loss / len(outputs)

def SGD(params, lr, delay_time=1):
    ''' If delay > 1, using aggregate gradients updates, needs to divded the change
        magnitude with delay_tiem
    '''
    for param in params:
        param[:] = param - lr * param.grad / delay_time

def grad_clipping(params, theta, ctx):
    if theta is not None:
        norm = nd.array([0.0], ctx)
        for p in params:
            norm += nd.sum(p.grad ** 2)
        norm = nd.sqrt(norm).asscalar()
        if norm > theta:
            for p in params:
                p.grad[:] *= theta / norm

def eval(data_source):
    total_loss = 0.0
    ntotal = 0

    data, label = zip(*data_source)

    for i in range(args.dev_size):
        hidden = nd.zeros(shape=(args.batch_size, args.num_hidden), ctx=context)
        # make one hot vectors
        idata = mx.ndarray.one_hot(mx.nd.array(data[i]), args.vocab_size)
        ilabel = mx.ndarray.one_hot(mx.nd.array(label[i]), args.vocab_size)

        idata = mx.nd.reshape(idata,(idata.shape[0],args.batch_size,idata.shape[1]))
        ilabel = mx.nd.reshape(ilabel,(ilabel.shape[0],args.batch_size,ilabel.shape[1]))

        output, hidden = model.forward(idata, hidden)
        loss = average_ce_loss(output, ilabel)

        total_loss += mx.nd.sum(loss).asscalar()
        # ntotal += idata.shape[0]
    # assert ntotal == args.dev_size, (ntotal)
    return total_loss / len(data_source)


def train():
    '''monitor the model performance on the training, validation,
    and testing data sets over iterations.
    '''
    best_val = float("Inf")

    for epoch in range(args.epochs):
        ############################
        # Attenuate the learning rate.
        ############################
        if args.anneal > 0:
            learning_rate = args.lr/((epoch+0.0+args.anneal)/args.anneal)


        hidden = nd.zeros(shape=(args.batch_size, args.num_hidden), ctx=context)
        total_L = 0.0
        start_time = time.time()
        random.shuffle(X_D_train)
        generate_data = ((x,y) for x,y in X_D_train)

        for i in tqdm(range(args.train_size)):
            hidden = nd.zeros(shape=(args.batch_size, args.num_hidden), ctx=context)
            data, label = next(generate_data)
            # make one hot vectors
            data = mx.ndarray.one_hot(mx.nd.array(data), args.vocab_size)
            data  = mx.nd.reshape(data,(data.shape[0], args.batch_size, data.shape[1]))
            label = mx.ndarray.one_hot(mx.nd.array(label), args.vocab_size)
            label = mx.nd.reshape(label,(label.shape[0],args.batch_size,label.shape[1]))

            with autograd.record():
                output, hidden = model.forward(data, hidden)
                loss = average_ce_loss(output, label)
                loss.backward()

            grad_clipping(model.params, args.acc_grad_size * args.clip, context)
            # SGD(model.params, learning_rate)
            if (i+1) % args.acc_grad_size == 0:
                SGD(model.params, learning_rate, args.acc_grad_size)
                # Now manually zero the grads
                for param in model.params:
                    param.grad[:] = 0.0

             ##########################
            #  Keep a moving average of the losses
            ##########################
            if (i == 0) and (epoch == 0):
                moving_loss = nd.mean(loss).asscalar()
            else:
                moving_loss = .99 * moving_loss + .01 * nd.mean(loss).asscalar()

        val_loss = eval(X_D_dev)
        logging.info('[Epoch %d] cost %.2fs, lr=%.2f, training loss %.2f, validation loss %.6f, perplexity %.6f' % (
            epoch + 1, time.time() - start_time, learning_rate, moving_loss, val_loss, math.exp(val_loss)))

        if val_loss < best_val:
            best_val = val_loss

def predict_lm():
    # get vocabulary
    vocab = pd.read_table(data_folder + "/vocab.wiki.txt", header=None, sep="\s+", index_col=0, names=['count', 'freq'], )
    num_to_word = dict(enumerate(vocab.index[:vocab_size]))
    word_to_num = invert_dict(num_to_word)
    # load test data
    sents, test_span = load_lm_np_dataset(data_folder + '/wiki-test.txt', args.test_size)
    S_np_test = docs_to_indices(sents, word_to_num, 1, 0)
    X_np_test, D_np_test = seqs_to_lmnpXY(S_np_test)
    logging.info("Load test set in lm-ln mode with size {}".format(X_np_test.size))
    np_acc_test = compute_acc_lmnp(X_np_test, D_np_test)
    logging.info('Number prediction accuracy on test set: {}'.format(np_acc_test))
    test_span = np.concatenate((np.array(stats), np.array(test_span, ndmin=2).T), axis = 1)
    np.savetxt(args.model+'test_span.csv', np.array(test_span), delimiter=',')

def load_lm_np_dataset(fname, size=None):
    sents = []
    # return the span between subj_idx & verb_idx
    span = []
    cnt = 0
    with open(fname) as f:
        for line in f:
            if cnt == 0:
                cnt += 1
                continue
            items = line.strip().split('\t')
            verb_idx = int(items[2])
            verb = items[4]
            inf_verb = items[5]
            sent = items[0].split()[:verb_idx] + [verb, inf_verb]
            sents.append(sent)
            span.append(int(items[2]) - int(items[1]))
            if size and cnt == size:
                break
            cnt += 1
    return sents, span

def compute_acc_lmnp(X, D):
        '''
        X            a list of input vectors, e.g.,         [[5, 4, 2], [7, 3, 8]]
        D            a list of pair verb forms (plural/singular), e.g.,     [[4, 9], [6, 5]]
        '''
        acc = sum([compare_num_pred(X[i], D[i]) for i in range(len(X))]) / len(X)
        return acc

def compare_num_pred(x, d):
    '''
        x        list of words, as indices, e.g.: [0, 4, 2]
        d        the desired verb and its (re)inflected form (singular/plural), as indices, e.g.: [7, 8]
        return 1 if p(d[0]) > p(d[1]), 0 otherwise
    '''
    hidden = model.begin_state(batch_size=args.batch_size, ctx=context)
    # make one hot vectors
    data = mx.ndarray.one_hot(mx.nd.array(x), args.vocab_size)
    data = mx.nd.reshape(data,(data.shape[0], args.batch_size, data.shape[1]))

    y, _ = model.forward(data, hidden) # output shape (batch_size, vocab_size).
    predict = 1 if y[-1][0, int(d[0])] > y[-1][0, int(d[1])] else 0

    stats.append([len(x), predict])
    return predict



if __name__ == '__main__':
    if args.cuda:
        context = mx.gpu(0)
    else:
        context = mx.cpu(0)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_handler = logging.FileHandler(pjoin(output_dir, "gru_log.txt"))
    logging.getLogger().addHandler(file_handler)

    ###############################################################################
    # Load data
    ###############################################################################
    '''To speed up the subsequent data flow in the RNN model,
       No batches: just use accumulative gradient
       Batches: pre-process the loaded data as batches. There is three possible solution:
       1. reshape data as equal length
       2. utilize the bucketsmodule(module)
       3. use explicit unroll with valid_length
    '''
    corpus = Corpus(vocab_size=args.vocab_size, vocab_file = args.data_folder+ 'vocab.wiki.txt')

    '''prepare data, support buckets batches
    '''
    print("prepare data" + '.'*10)
    # buckets for batch training
    # buckets = [ 10, 20, 30, 40, 50, 60]
    X_D_train = corpus.tokenize(args.data_folder + 'wiki-train.txt',
                            args.sequence_length, args.train_size)
    X_D_dev   = corpus.tokenize(args.data_folder + 'wiki-dev.txt',
                            args.sequence_length, args.dev_size)


    '''Build the model, initialize model parameters,
    and configure the optimization algorithms for training the RNN model.'''
    model = GRU(args.vocab_size, args.num_hidden, seed=2018, ctx=context)
    # Attach the gradients
    for param in model.params:
        param.attach_grad(grad_req='add')
        # param.attach_grad()

    train()

    data_folder = args.data_folder
    vocab_size = args.vocab_size
    stats = []
    X_D_test  = corpus.tokenize(args.data_folder + 'wiki-test.txt',
                            args.sequence_length, args.test_size)
    predict_lm()


    # test_L = eval(X_D_test)
    # print('Best test loss %.2f, test perplexity %.2f'%(test_L, math.exp(test_L)))
