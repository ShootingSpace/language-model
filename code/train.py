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
parser.add_argument('--sequence_length', type=int, default=60, help='sequence length for each inputs')
parser.add_argument('--train_size', type=int, default=5, help='train_size')
parser.add_argument('--dev_size', type=int, default=5, help='dev_size')
parser.add_argument('--num_embed', type=int, default=2000, help='size of word embeddings')
parser.add_argument('--num_hidden', type=int, default=50, help='number of hidden units per layer')
parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
parser.add_argument('--lr', type=float, default=0.5, help='initial learning rate')
parser.add_argument('--anneal', type=float, default=5,
                    help='if > 0, lowers the learning rate in a harmonically \
                    after each epoch')
parser.add_argument('--clip', type=float, default=0.2, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=30, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=50, metavar='N', help='batch size')
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
    total_L = 0.0
    ntotal = 0
    hidden = model.begin_state(func = mx.nd.zeros, batch_size = args_batch_size, ctx=context)
    for i in range(0, data_source.shape[0] - 1, args_bptt):
        data, target = get_batch(data_source, i)
        output, hidden = model(data, hidden)
        L = loss(output, target)
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal

def train():
    '''monitor the model performance on the training, validation,
    and testing data sets over iterations.
    '''
    best_val = float("Inf")
    for epoch in range(args.epochs):
        total_L = 0.0
        start_time = time.time()
        hidden = model.begin_state(func = mx.nd.zeros, ctx = context)
        for ibatch, i in enumerate(range(0, X_train.shape[0] - 1, args.bptt)):
            data, target = get_batch(train_data, i)
            hidden = detach(hidden)
            with autograd.record():
                output, hidden = model(data, hidden)
                L = loss(output, target)
                L.backward()

            grads = [i.grad(context) for i in model.collect_params().values()]
            # Here gradient is for the whole batch.
            # So we multiply max_norm by batch_size and bptt size to balance it.
            gluon.utils.clip_global_norm(grads, args_clip * args_bptt * args_batch_size)

            trainer.step(args_batch_size)
            total_L += mx.nd.sum(L).asscalar()

            if ibatch % args_log_interval == 0 and ibatch > 0:
                cur_L = total_L / args_bptt / args_batch_size / args_log_interval
                print('[Epoch %d Batch %d] loss %.2f, perplexity %.2f' % (
                    epoch + 1, ibatch, cur_L, math.exp(cur_L)))
                total_L = 0.0

        val_L = eval(val_data)

        print('[Epoch %d] time cost %.2fs, validation loss %.2f, validation perplexity %.2f' % (
            epoch + 1, time.time() - start_time, val_L, math.exp(val_L)))

        if val_L < best_val:
            best_val = val_L
            test_L = eval(test_data)
            model.save_params(args_save)
            print('test loss %.2f, test perplexity %.2f' % (test_L, math.exp(test_L)))
        else:
            args_lr = args_lr * 0.25
            trainer._init_optimizer('sgd',
                                    {'learning_rate': args_lr,
                                     'momentum': 0,
                                     'wd': 0})
            model.load_params(args_save, context)

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
       No batches: just use accumulative gradient
       Batches: pre-process the loaded data as batches. There is three possible solution:
       1. reshape data as equal length
       2. utilize the bucketsmodule(module)
       3. use explicit unroll with valid_length
    '''
    corpus = Corpus(vocab_size=args.vocab_size, vocab_file = args.data_folder+ 'vocab.wiki.txt')

    '''prepare data, support buckets batches
    '''
    # buckets for batch training
    # buckets = [ 10, 20, 30, 40, 50, 60]
    X_train, D_train = corpus.tokenize(args.data_folder + 'wiki-train.txt',
                            args.sequence_length, args.train_size)
    X_dev, D_dev   = corpus.tokenize(args.data_folder + 'wiki-dev.txt',
                            args.sequence_length, args.dev_size)
    # X_test, D_test  = corpus.tokenize(args.data_folder + 'wiki-test.txt',
    #                         args.sequence_length)

    print(X_train.shape)
    # train_data  = mx.rnn.BucketSentenceIter(train_corpus_indexing, args.batch_size, data_name='train_data')
    # dev_data    = mx.rnn.BucketSentenceIter(dev_corpus_indexing, args.batch_size, data_name='dev_data')
    # test_data   = mx.rnn.BucketSentenceIter(test_corpus_indexing, args.batch_size, data_name='test_data')


    '''Build the model, initialize model parameters,
    and configure the optimization algorithms for training the RNN model.'''

    model = RNNModel(args.model, args.vocab_size, args_emsize, args_nhid,
                           args_nlayers, args_dropout, args_tied)
    model.collect_params().initialize(mx.init.Xavier(), ctx=context)
    trainer = gluon.Trainer(model.collect_params(), 'sgd',
                            {'learning_rate': args.lr, 'momentum': 0, 'wd': 0})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
