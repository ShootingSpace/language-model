import argparse
import time
import math
import mxnet as mx
from mxnet import gluon, autograd
from lstm import RNNModel, Corpus
import numpy as np
import bisect
import random

parser = argparse.ArgumentParser(description='MXNet Autograd RNN/LSTM Language Model.')
parser.add_argument('--model', type=str, default='lstm',
                    help='type of recurrent net (rnn_tanh, rnn_relu, lstm, gru)')
parser.add_argument('--data_folder', type=str, default='data/', help='folder that store data')
parser.add_argument('--vocab_size', type=int, default=2000, help='vocab_size')
parser.add_argument('--sequence_length', type=int, default=60, help='sequence length for each inputs')
parser.add_argument('--train_size', type=int, default=100, help='train_size')
parser.add_argument('--dev_size', type=int, default=100, help='dev_size')
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
parser.add_argument('--bptt', type=int, default=0, help='sequence length')
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


'''Train the model and evaluate on validation and testing data sets
helper functions that will be used during model training and evaluation.'''
def get_batch(source, i):
    pass

def eval(data_source):
    total_L = 0.0
    ntotal = 0
    hidden = model.begin_state(func = mx.nd.zeros, batch_size = args.batch_size, ctx=context)

    data, label = zip(*data_source)
    data = mx.nd.array(data)
    label = mx.nd.array(label)
    for i in range(args.dev_size):
        data  = mx.nd.reshape(data,(data.shape[0],args.batch_size,data.shape[1]))
        output, hidden = model(data, hidden)
        L = loss(data, label)
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    assert ntotal == args.dev_size
    return total_L / ntotal

def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden


def train():
    '''monitor the model performance on the training, validation,
    and testing data sets over iterations.
    '''
    best_val = float("Inf")

    np.random.seed(2018)


    for epoch in range(args.epochs):
        total_L = 0.0
        start_time = time.time()
        hidden = model.begin_state(batch_size=args.batch_size, func = mx.nd.zeros, ctx = context)

        random.shuffle(X_D_train)
        generate_data = ((x,y) for x,y in X_D_train)
        permutation = np.random.permutation(range(args.train_size))

        model.collect_params().zero_grad()
        batch_num = args.train_size // args.acc_grad_size

        for ibatch in range(batch_num):
            '''One mini batch'''
            hidden = detach(hidden)
            with autograd.record():
                for j in range(args.acc_grad_size):
                    L_list = []
                    data, label = next(generate_data)
                    data  = mx.nd.reshape(data,(data.shape[0],args.batch_size,data.shape[1]))
                    # label = mx.nd.reshape(label,(label.shape[0],args.batch_size,label.shape[1]))
                    label = mx.nd.array(label)
                    output, hidden = model(data, hidden)
                    L_list.append(loss(output, label))
                L = sum(L_list)
                L.backward()
                total_L += mx.nd.sum(L).asscalar()
            trainer.step(args.acc_grad_size)
            if ibatch % args.log_interval == 0 and ibatch > 0:
                cur_L = total_L / args.bptt / args.acc_grad_size / args.log_interval
                print('[Epoch %d Batch %d] loss %.2f, perplexity %.2f' % (
                    epoch + 1, ibatch, cur_L, math.exp(cur_L)))
                total_L = 0.0

        val_L = eval(X_D_dev)
        print('[Epoch %d] time cost %.2fs, validation loss %.2f, validation perplexity %.2f' % (
            epoch + 1, time.time() - start_time, val_L, math.exp(val_L)))

        if val_L < best_val:
            best_val = val_L
            # test_L = eval(test_data)
            model.save_params(args_save)
            print('DEV loss %.2f, DEV perplexity %.2f' % (val_L, math.exp(val_L)))
        else:
            # args_lr = args_lr * 0.25
            args_lr = args_lr/((epoch+0.0+args.anneal)/args.anneal)
            trainer._init_optimizer('sgd',
                                    {'learning_rate': args_lr,
                                     'momentum': 0,
                                     'wd': 0})
            model.load_params(args.save, context)


if __name__ == '__main__':
    if args.cuda:
        context = mx.gpu(0)
    else:
        context = mx.cpu(0)
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
    print("prepare data, support buckets batches")
    # buckets for batch training
    # buckets = [ 10, 20, 30, 40, 50, 60]
    X_D_train = corpus.tokenize(args.data_folder + 'wiki-train.txt',
                            args.sequence_length, args.train_size)
    X_D_dev   = corpus.tokenize(args.data_folder + 'wiki-dev.txt',
                            args.sequence_length, args.dev_size)
    # X_D_test  = corpus.tokenize(args.data_folder + 'wiki-test.txt',
    #                         args.sequence_length)

    # train_data  = mx.rnn.BucketSentenceIter(train_corpus_indexing, args.acc_grad_size, data_name='train_data')
    # dev_data    = mx.rnn.BucketSentenceIter(dev_corpus_indexing, args.acc_grad_size, data_name='dev_data')
    # test_data   = mx.rnn.BucketSentenceIter(test_corpus_indexing, args.acc_grad_size, data_name='test_data')


    '''Build the model, initialize model parameters,
    and configure the optimization algorithms for training the RNN model.'''

    model = RNNModel(args.model, args.vocab_size, args.num_embed, args.num_hidden,
                           args.num_layers, args.dropout, args.tied)

    # Parameter initialization
    model.collect_params().initialize(mx.init.Xavier(), ctx=context)
    for p in model.collect_params().values():
        p.grad_req = 'add'
    trainer = gluon.Trainer(model.collect_params(), 'sgd',
                            {'learning_rate': args.lr, 'momentum': 0, 'wd': 0})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    train()
    model.load_params(args.save, context)
    test_L = eval(X_D_test)
    print('Best test loss %.2f, test perplexity %.2f'%(test_L, math.exp(test_L)))
