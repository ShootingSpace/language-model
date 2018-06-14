import argparse
import time
import math
import mxnet as mx
from mxnet import gluon, autograd, nd
from lstm import RNNModel, Corpus
import numpy as np
import bisect
import random
from tqdm import tqdm
import os
from os.path import join as pjoin
from tempfile import TemporaryFile
import logging
from mxnet.gluon import nn, rnn
from corpus import Corpus
from gru import GRU
from rnn_cell import RNN
from utils import *
from rnnmath import *

parser = argparse.ArgumentParser(description='MXNet Autograd RNN/LSTM Language Model.')
parser.add_argument('--model', type=str, default='lstm',
                    help='type of recurrent net (rnn_tanh, rnn_relu, lstm, gru)')
parser.add_argument('--data_folder', type=str, default='data/', help='folder that store data')
parser.add_argument('--output_dir', type=str, default='output', help='log directory')
parser.add_argument('--vocab_size', type=int, default=2000, help='vocab_size')
parser.add_argument('--sequence_length', type=int, default=60, help='sequence length for each inputs')
parser.add_argument('--train_size', type=int, default=1000, help='train_size')
parser.add_argument('--dev_size', type=int, default=1000, help='dev_size')
parser.add_argument('--test_size', type=int, default=None, help='test_size')
parser.add_argument('--num_embed', type=int, default=0, help='size of word embeddings')
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
parser.add_argument('--seed', type=int, default=2018, help='seed')
parser.add_argument('--load_params', type=int, default=0, help='whether load pramas from file')
parser.add_argument('--pretrained_model', type=str, default='model.params', help='The saved model file')
parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer for gluon trainer')
parser.add_argument('--logfile', type=str, default='model', help='Log file')
args = parser.parse_args()

def softmax(y_linear, temperature=1.0):
    lin = (y_linear-nd.max(y_linear)) / temperature
    exp = nd.exp(lin)
    partition = nd.sum(exp, axis=0, exclude=True).reshape((-1,1))
    return exp / partition

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden

def cross_entropy(yhat, y):
    # return - nd.mean(nd.sum(y * nd.log_softmax(yhat), axis=0, exclude=True))
    return - nd.sum(y * nd.log_softmax(yhat))

def average_ce_loss(outputs, labels):
    '''Averaging the loss over all words in all sentences of the given corpus'''
    assert(len(outputs) == len(labels))
    total_loss = nd.array([0.], ctx=context)
    for (output, label) in zip(outputs,labels):
        # print("cross_entropy", cross_entropy(output, label))
        total_loss = total_loss + cross_entropy(output, label)
    # if math.isnan(float(total_loss.asnumpy()[0]/ len(outputs))):
    #     print(total_loss, len(outputs))
    # return total_loss / len(outputs)
    return total_loss

def eval(data_source):
    total_loss = 0.0
    total_words = 0

    data, label = zip(*data_source)

    for i in range(args.dev_size):
        hidden = model.begin_state(func = mx.nd.zeros, batch_size=args.batch_size, ctx=context)
        idata, ilabel = make_inputs(mx.nd.array(data[i]), mx.nd.array(label[i]))

        output, hidden = model.forward(idata, hidden)
        # val_loss = loss(output, ilabel)
        val_loss = average_ce_loss(output, ilabel)
        total_loss += mx.nd.sum(val_loss).asscalar()
        total_words += idata.shape[0]
    # assert ntotal == args.dev_size, (ntotal)
    return total_loss / total_words
    # return total_loss / len(data)


def train():
    '''monitor the model performance on the training, validation,
    and testing data sets over iterations.
    '''

    if args.seed:
        np.random.seed(args.seed)

    best_val = float("Inf")
    logging.info("Calculating initial mean loss on dev set: {}".format(eval(X_D_dev)))

    for epoch in range(args.epochs):
        ############################
        # Attenuate the learning rate.
        ############################
        if args.anneal > 0 and args.optimizer=='sgd':
            learning_rate = args.lr/((epoch+0.0+args.anneal)/args.anneal)
            trainer.set_learning_rate(learning_rate)

        # hidden = nd.zeros(shape=(args.batch_size, args.num_hidden), ctx=context)
        hidden = model.begin_state(func = mx.nd.zeros, batch_size=args.batch_size, ctx=context)

        total_sequence_size = 0
        total_loss = 0.0
        start_time = time.time()
        np.random.shuffle(X_D_train)
        generate_data = ((x,y) for x,y in X_D_train)

        for i in tqdm(range(args.train_size)):
            # hidden = detach(hidden)
            hidden = model.begin_state(func = mx.nd.zeros, batch_size=args.batch_size, ctx=context)

            data, label = next(generate_data)
            data, label = make_inputs(mx.nd.array(data), mx.nd.array(label))

            with autograd.record():
                output, hidden = model.forward(data, hidden)
                # train_loss = average_ce_loss(output, label)
                train_loss = loss(output, label)
                train_loss.backward()

            grads = [i.grad(context) for i in model.collect_params().values()]

            '''Here gradient is for the whole batch.
            So we multiply max_norm by batch_size and bptt size to balance it.'''
            gluon.utils.clip_global_norm(grads, args.clip * args.acc_grad_size)

            hidden = detach(hidden)
            if (i+1) % args.acc_grad_size == 0:
                trainer.step(args.acc_grad_size)
                '''Sets gradient buffer on all contexts to 0.
                   call zero_grad() on each parameter after Trainer.step().
                '''
                model.collect_params().zero_grad()


            ##########################
            #  Keep a moving average of the losses
            ##########################
            # total_loss += mx.nd.sum(train_loss).asscalar()
            # total_sequence_size += len(output)
            # if (i == 0) and (epoch == 0):
            #     moving_loss = nd.mean(total_loss).asscalar()
            # else:
            #     moving_loss = .99 * moving_loss + .01 * nd.mean(train_loss).asscalar()

        val_loss = eval(X_D_dev)
        logging.info('[Epoch %d] cost %.2fs, lr=%.6f, validation loss %.6f, perplexity %.6f' % (
            epoch + 1, time.time() - start_time, trainer.learning_rate, val_loss, math.exp(val_loss)))

        if val_loss < best_val:
            best_val = val_loss

            model.save_params(args.save)

    logging.info("Finished trianing with best dev error {}".format(best_val))
    test_loss = eval(X_D_test)
    logging.info('TEST loss {}, perplexity {}'.format(test_loss, math.exp(test_loss)))

def make_inputs(data, label=None):
    '''whether to make one hot encode and reshape or not'''
    if args.num_embed:
        data  = mx.nd.reshape(data,(-1, args.batch_size))
    else:
        # make one hot vectors
        data  = mx.ndarray.one_hot(data, args.vocab_size)
        data  = mx.nd.reshape(data,(data.shape[0], args.batch_size, data.shape[1]))

    if label is not None:
        label  = mx.ndarray.one_hot(label, args.vocab_size)
        label  = mx.nd.reshape(label,(label.shape[0], args.batch_size, label.shape[1]))

    return data, label


def predict_lm(data):
    # get vocabulary
    vocab = pd.read_table(data_folder + "/vocab.wiki.txt", header=None, sep="\s+", index_col=0, names=['count', 'freq'], )
    num_to_word = dict(enumerate(vocab.index[:vocab_size]))
    word_to_num = invert_dict(num_to_word)
    # load test data
    sents, span = load_lm_np_dataset(data_folder + '/wiki-'+ data + '.txt')
    S_np = docs_to_indices(sents, word_to_num, 1, 0)
    X_np, D_np = seqs_to_lmnpXY(S_np)
    logging.info("Load {} set in lm-ln mode with size {}".format(data, X_np.size))
    np_acc = compute_acc_lmnp(X_np, D_np)
    logging.info('Number prediction accuracy on {} set: {}'.format(data, np_acc))
    span = np.concatenate((np.array(stats), np.array(span, ndmin=2).T), axis = 1)
    np.savetxt(args.model+ '_' + data +'_span.csv', np.array(span), delimiter=',')

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

    hidden = model.begin_state(func = mx.nd.zeros, batch_size=args.batch_size, ctx=context)
    data, _ = make_inputs(mx.nd.array(x))

    y, _ = model.forward(data, hidden) # output shape (batch_size, vocab_size).
    # y = y.asnumpy()
    predict = 1 if y[len(x)-1, int(d[0])] > y[len(x)-1, int(d[1])] else 0
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
    file_handler = logging.FileHandler(pjoin(output_dir, args.logfile + "_log.txt"))
    logging.getLogger().addHandler(file_handler)

    run_loss = -1
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

    '''prepare data, support buckets batches '''
    logging.info("prepare data" + '.'*10)
    # buckets for batch training
    # buckets = [ 10, 20, 30, 40, 50, 60]
    X_D_train = corpus.tokenize(args.data_folder + 'wiki-train.txt',
                            args.sequence_length, args.train_size)
    X_D_dev   = corpus.tokenize(args.data_folder + 'wiki-dev.txt',
                            args.sequence_length, args.dev_size)
    X_D_test  = corpus.tokenize(args.data_folder + 'wiki-test.txt',
                            args.sequence_length, args.test_size)

    '''Build the model, initialize model parameters,
    and configure the optimization algorithms for training the RNN model.'''

    # Setup model and Parameter initialization
    logging.info('{} Using {}, with learning rate {}, training size {}, embedding size {}\
    hidden units {}, run {} epochs.'.format('='*10, args.model, args.lr, args.train_size,
                                    args.num_embed, args.num_hidden, args.epochs))
    model = RNN(mode=args.model, seed=args.seed, vocab_size=args.vocab_size, num_embed=args.num_embed,
                num_hidden=args.num_hidden, num_layers=args.num_layers, dropout=args.dropout)
    if args.load_params:
        logging.info('Load pretrianed model from {}'.format(args.pretrained_model))
        model.load_params(args.pretrained_model, ctx=context)
    else:
        model.collect_params().initialize(mx.init.Xavier(), ctx=context)

    trainer = gluon.Trainer(model.collect_params(), args.optimizer,
                            {'learning_rate': args.lr})

    loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False, batch_axis=1)

    for p in model.collect_params().values():
        p.grad_req = 'add'

    train()

    data_folder = args.data_folder
    vocab_size = args.vocab_size
    stats = []
    predict_lm('dev')
    stats = [] # clear
    predict_lm('test')


    # test_L = eval(X_D_test)
    # print('Best test loss %.2f, test perplexity %.2f'%(test_L, math.exp(test_L)))
