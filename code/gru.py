from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
import numpy as np

def softmax(y_linear, temperature=1.0):
    lin = (y_linear-nd.max(y_linear)) / temperature
    exp = nd.exp(lin)
    partition = nd.sum(exp, axis=0, exclude=True).reshape((-1,1))
    return exp / partition

''' Implement a GRU model
'''
class GRU():
    def __init__(self, vocab_size, num_hidden, seed, ctx=mx.cpu(0)):
        if seed:
            mx.random.seed(2018)

        num_inputs = vocab_size
        num_outputs = vocab_size
        self.num_hidden = num_hidden

        ########################
        #  Weights connecting the inputs to the hidden layer
        ########################
        self.Wxz = nd.random_normal(shape=(num_inputs,num_hidden), ctx=ctx) * .01
        self.Wxr = nd.random_normal(shape=(num_inputs,num_hidden), ctx=ctx) * .01
        self.Wxh = nd.random_normal(shape=(num_inputs,num_hidden), ctx=ctx) * .01

        ########################
        #  Recurrent weights connecting the hidden layer across time steps
        ########################
        self.Whz = nd.random_normal(shape=(num_hidden,num_hidden), ctx=ctx)* .01
        self.Whr = nd.random_normal(shape=(num_hidden,num_hidden), ctx=ctx)* .01
        self.Whh = nd.random_normal(shape=(num_hidden,num_hidden), ctx=ctx)* .01

        ########################
        #  Bias vector for hidden layer
        ########################
        self.bz = nd.random_normal(shape=num_hidden, ctx=ctx) * .01
        self.br = nd.random_normal(shape=num_hidden, ctx=ctx) * .01
        self.bh = nd.random_normal(shape=num_hidden, ctx=ctx) * .01

        ########################
        # Weights to the output nodes
        ########################
        self.Why = nd.random_normal(shape=(num_hidden,num_outputs), ctx=ctx) * .01
        self.by = nd.random_normal(shape=num_outputs, ctx=ctx) * .01

        self.params = [self.Wxz, self.Wxr, self.Wxh, self.Whz, self.Whr, self.Whh,
                     self.bz, self.br, self.bh, self.Why, self.by]

    def forward(self, inputs, h, temperature=1.0):
        outputs = []
        for X in inputs:
            z = nd.sigmoid(nd.dot(X, self.Wxz) + nd.dot(h, self.Whz) + self.bz)
            r = nd.sigmoid(nd.dot(X, self.Wxr) + nd.dot(h, self.Whr) + self.br)
            g = nd.tanh(nd.dot(X, self.Wxh) + nd.dot(r * h, self.Whh) + self.bh)
            h = z * h + (1 - z) * g

            yhat_linear = nd.dot(h, self.Why) + self.by
            # print(yhat_linear)
            yhat = softmax(yhat_linear, temperature=temperature)
            outputs.append(yhat)
        return (outputs, h)

    def begin_state(self, batch_size, ctx):
        return nd.zeros(shape=(batch_size, self.num_hidden), ctx=ctx)
