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


class RNN(object):
    '''
    This class implements Recurrent Neural Networks.

    You should implement code in the following functions:
        predict                ->    predict an output sequence for a given input sequence
        acc_deltas            ->    accumulate update weights for the RNNs weight matrices, standard Back Propagation
        acc_deltas_bptt        ->    accumulate update weights for the RNNs weight matrices, using Back Propagation Through Time
        acc_deltas_np		->	accumulate update weights for the RNNs weight matrices, standard Back Propagation -- for number predictions
		acc_deltas_bptt_np	->	accumulate update weights for the RNNs weight matrices, using Back Propagation Through Time -- for number predictions
        compute_loss         ->    compute the (cross entropy) loss between the desired output and predicted output for a given input sequence
        compute_mean_loss    ->    compute the average loss over all sequences in a corpus
        generate_sequence    ->    use the RNN to generate a new (unseen) sequnce

    Do NOT modify any other methods!
    Do NOT change any method signatures!
    '''

    def __init__(self, vocab_size, hidden_dims, out_vocab_size):
        '''
        initialize the RNN with random weight matrices.

        DO NOT CHANGE THIS

        vocab_size        size of vocabulary that is being used
        hidden_dims        number of hidden units
        out_vocab_size    size of the output vocabulary
        '''
        self.vocab_size = vocab_size
        self.hidden_dims = hidden_dims
        self.out_vocab_size = out_vocab_size

        # matrices V (input -> hidden), W (hidden -> output), U (hidden -> hidden)
        self.U = np.random.randn(self.hidden_dims, self.hidden_dims)*np.sqrt(0.1)
        self.V = np.random.randn(self.hidden_dims, self.vocab_size)*np.sqrt(0.1)
        self.W = np.random.randn(self.out_vocab_size, self.hidden_dims)*np.sqrt(0.1)

        # matrices to accumulate weight updates
        self.deltaU = np.zeros((self.hidden_dims, self.hidden_dims))
        self.deltaV = np.zeros((self.hidden_dims, self.vocab_size))
        self.deltaW = np.zeros((self.out_vocab_size, self.hidden_dims))

        # for data analysis
        self.stats = []
        self.words_embeddings = np.zeros((self.out_vocab_size, self.hidden_dims+1))

    def apply_deltas(self, learning_rate):
        '''
        update the RNN's weight matrices with corrections accumulated over some training instances

        DO NOT CHANGE THIS

        learning_rate    scaling factor for update weights
        '''
        # apply updates to U, V, W
        self.U += learning_rate*self.deltaU
        self.W += learning_rate*self.deltaW
        self.V += learning_rate*self.deltaV

        # reset matrices
        self.deltaU.fill(0.0)
        self.deltaV.fill(0.0)
        self.deltaW.fill(0.0)

    def predict(self, x):
        '''
        predict an output sequence y for a given input sequence x

        x    list of words, as indices, e.g.: [0, 4, 2]

        returns    y,s
        y    matrix of probability vectors for each input word
        s    matrix of hidden layers for each input word

        '''

        # matrix s for hidden states, y for output states, given input x.
        # rows correspond to times t, i.e., input words
        # s has one more row, since we need to look back even at time 0 (s(t=0-1) will just be [0. 0. ....] )
        s = np.zeros((len(x) + 1, self.hidden_dims))
        y = np.zeros((len(x), self.out_vocab_size))

        for t in range(len(x)):
            ##########################
            # --- your code here --- #
            # make one-hot input
            xt = make_onehot(x[t], self.vocab_size)
            # compute next activation state using net(t) = Vx(t) + Us(t-1)
            s_next = sigmoid(self.V.dot(xt) + self.U.dot(s[t-1, :]))
            # compute output of the current cell yt = g(Ws(t))
            y[t,:] = softmax(self.W.dot(s_next))
            # Save the value of the new "next" hidden state in s
            s[t,:] = s_next


            ##########################

        return y, s

    def acc_deltas(self, x, d, y, s):
        '''
        accumulate updates for V, W, U
        standard back propagation

        this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time

        x    list of words, as indices, e.g.: [0, 4, 2]
        d    list of words, as indices, e.g.: [4, 2, 3]
        y    predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
            should be part of the return value of predict(x)
        s    predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
            should be part of the return value of predict(x)

        no return values
        '''
        for t in reversed(range(len(x))):
            ##########################
            # --- your code here --- #
            xt = make_onehot(x[t], self.vocab_size)
            dt = make_onehot(d[t], self.out_vocab_size)
            yt = y[t, :]
            st = s[t, :]

            # compute the gradient of Sigmoid w.r.t to hidden state
            dsigmoid = grad(st) #shape(hidd,)
            delta_out = dt - yt #shape(out,)
            delta_in = self.W.T.dot(delta_out) * dsigmoid #shape(hidd,)

            # compute the gradient of W
            self.deltaW += np.outer(delta_out, st) #shape(out, hidd)
            # compute the gradient of the loss with respect to V
            self.deltaV += np.outer(delta_in, xt) #shape(hidd, voc)
            # compute the gradient with respect to U
            self.deltaU += np.outer(delta_in, s[t-1, :]) #shape(hidd, hidd)

            ##########################


    def acc_deltas_np(self, x, d, y, s):
        '''
        accumulate updates for V, W, U
        standard back propagation

        this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
        for number prediction task, we do binary prediction, 0 or 1

        x    list of words, as indices, e.g.: [0, 4, 2]
        d    array with one element, as indices, e.g.: [0] or [1]
        y    predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
            should be part of the return value of predict(x)
        s    predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
            should be part of the return value of predict(x)

        no return values
        '''

        ##########################
        # --- your code here --- #
        t = len(x)-1
        xt = make_onehot(x[t], self.vocab_size)
        st = s[t, :]
        # compute the gradient of W
        dt = make_onehot(d[0], self.out_vocab_size)
        yt = y[t, :]
        delta_out = dt - yt #shape(out,)
        self.deltaW += np.outer(delta_out, st) #shape(out, hidd)
        # compute the gradient of Sigmoid w.r.t to hidden state
        dsigmoid = grad(st) #shape(hidd,)
        delta_in = dsigmoid * self.W.T.dot(delta_out) #shape(hidd,)
        # compute the gradient of the loss with respect to V
        self.deltaV += np.outer(delta_in, xt) #shape(hidd, voc)
        # compute the gradient with respect to U
        self.deltaU += np.outer(delta_in, s[t-1, :]) #shape(hidd, hidd)

        ##########################


    def acc_deltas_bptt(self, x, d, y, s, steps):
        '''
        accumulate updates for V, W, U
        back propagation through time (BPTT)

        this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time

        x        list of words, as indices, e.g.: [0, 4, 2]
        d        list of words, as indices, e.g.: [4, 2, 3]
        y        predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
                should be part of the return value of predict(x)
        s        predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
                should be part of the return value of predict(x)
        steps    number of time steps to go back in BPTT

        no return values
        '''
        for t in reversed(range(len(x))):
            # print("time {0}".format(t))
            ##########################
            # --- your code here --- #
            xt = make_onehot(x[t], self.vocab_size)
            dt = make_onehot(d[t], self.out_vocab_size)
            yt = y[t, :]
            st = s[t, :]

            # compute the gradient of Sigmoid w.r.t to hidden state
            dsigmoid = grad(st) #shape(hidd,)
            delta_out = dt - yt #shape(out,)
            delta_in = self.W.T.dot(delta_out) * dsigmoid #shape(hidd,)
            # compute the gradient of W
            self.deltaW += np.outer(delta_out, st) #shape(out, hidd)
            # compute the gradient of the loss with respect to V
            self.deltaV += np.outer(delta_in, xt) #shape(hidd, voc)
            # compute the gradient with respect to U
            self.deltaU += np.outer(delta_in, s[t-1, :]) #shape(hidd, hidd)

            # steps > 0
            delta_in_tau = delta_in
            for step in range(1, steps+1):
                tau = t - step
                if tau < 0:
                    break
                xtau = make_onehot(x[tau], self.vocab_size)
                stau = s[tau, :]
                # compute the gradient of Sigmoid w.r.t to hidden state
                dsigmoid_tau = grad(stau) #shape(hidd,)
                delta_in_tau = self.U.T.dot(delta_in_tau) * dsigmoid_tau #shape(hidd,)

                # compute the gradient of the loss with respect to V
                self.deltaV += np.outer(delta_in_tau, xtau) #shape(hidd, voc)
                # compute the gradient with respect to U
                self.deltaU += np.outer(delta_in_tau, s[tau-1, :]) #shape(hidd, hidd)

            ##########################


    def acc_deltas_bptt_np(self, x, d, y, s, steps):
        '''
        accumulate updates for V, W, U
        back propagation through time (BPTT)

        this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
        for number prediction task, we do binary prediction, 0 or 1

        x    list of words, as indices, e.g.: [0, 4, 2]
        d    array with one element, as indices, e.g.: [0] or [1]
        y        predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
                should be part of the return value of predict(x)
        s        predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
                should be part of the return value of predict(x)
        steps    number of time steps to go back in BPTT

        no return values
        '''

        ##########################
        # --- your code here --- #
        t = len(x)-1
        xt = make_onehot(x[t], self.vocab_size)
        st = s[t, :]
        # compute the gradient of W
        dt = make_onehot(d[0], self.out_vocab_size)
        yt = y[t, :]
        delta_out = dt - yt #shape(out,)
        self.deltaW += np.outer(delta_out, st) #shape(out, hidd)
        # compute the gradient of Sigmoid w.r.t to hidden state
        dsigmoid = grad(st) #shape(hidd,)
        delta_in = dsigmoid * self.W.T.dot(delta_out) #shape(hidd,)
        # compute the gradient of the loss with respect to V
        self.deltaV += np.outer(delta_in, xt) #shape(hidd, voc)
        # compute the gradient with respect to U
        self.deltaU += np.outer(delta_in, s[t-1, :]) #shape(hidd, hidd)

        # steps > 0
        delta_in_tau = delta_in
        for step in range(1, steps+1):
            tau = t - step
            if tau < 0:
                break
            xtau = make_onehot(x[tau], self.vocab_size)
            stau = s[tau, :]
            # compute the gradient of Sigmoid w.r.t to hidden state
            dsigmoid_tau = grad(stau) #shape(hidd,)
            delta_in_tau = self.U.T.dot(delta_in_tau) * dsigmoid_tau #shape(hidd,)
            # compute the gradient of the loss with respect to V
            self.deltaV += np.outer(delta_in_tau, xtau) #shape(hidd, voc)
            # compute the gradient with respect to U
            self.deltaU += np.outer(delta_in_tau, s[tau-1, :]) #shape(hidd, hidd)
        ##########################


    def compute_loss(self, x, d):
        '''
        compute the loss between predictions y for x, and desired output d.

        first predicts the output for x using the RNN, then computes the loss w.r.t. d

        x        list of words, as indices, e.g.: [0, 4, 2]
        d        list of words, as indices, e.g.: [4, 2, 3]

        return loss        the combined loss for all words
        '''

        loss = 0.

        ##########################
        # --- your code here --- #
        y , _ = self.predict(x)
        for t in range(len(d)):
            dt = make_onehot(d[t], self.out_vocab_size)

            loss -= dt.dot(np.log(y[t,:]))

        ##########################

        return loss


    def compute_loss_np(self, x, d):
        '''
        compute the loss between predictions y for x, and desired output d.

        first predicts the output for x using the RNN, then computes the loss w.r.t. d

        x        list of words, as indices, e.g.: [0, 4, 2]
        d        a word, as indices, e.g.: [0]

        return loss        we only take the prediction from the last time step
        '''

        loss = 0.

        ##########################
        # --- your code here --- #
        y , _ = self.predict(x)
        dt = make_onehot(d[-1], self.out_vocab_size)

        loss = -dt.dot(np.log(y[-1]))
        ##########################

        return loss


    def compute_acc_np(self, x, d):
        '''
        compute the accuracy prediction, y[t] compared to the desired output d.
        first predicts the output for x using the RNN, then computes the loss w.r.t. d

        x        list of words, as indices, e.g.: [0, 4, 2]
        d        a word class (plural/singular), as index, e.g.: [0] or [1]

        return 1 if argmax(y[t]) == d[0], 0 otherwise
        '''


        ##########################
        # --- your code here --- #
        y , _ = self.predict(x)
        ##########################

        return 1 if np.argmax(y[-1]) == d[0] else 0


    def compare_num_pred(self, x, d):
        '''
        compute the probability between predictions the desired output d[0] and it's (re)inflected form, d[1].
        first predicts the output for x using the RNN, then compare the probability of d[0] and d[1].

        x        list of words, as indices, e.g.: [0, 4, 2]
        d        the desired verb and its (re)inflected form (singular/plural), as indices, e.g.: [7, 8]

        return 1 if p(d[0]) > p(d[1]), 0 otherwise
        '''

        ##########################
        # --- your code here --- #
        y , _ = self.predict(x)
        predict = 1 if y[-1, d[0]] > y[-1, d[1]] else 0
        self.stats.append([len(x), predict])
        return predict
        # return 1 if y[-1, d[0]] > y[-1, d[1]] else 0
        ##########################

    def load_embeddings(self, embedding_path, word_to_num):
        '''load the embedding, save in array, start with word index'''
        print("Loading Glove Model")
        f = open(embedding_path,'r')
        model = {}
        words_embeddings = []
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            if word in word_to_num:
                embedding = [float(val) for val in splitLine[1:]]
                words_embeddings.append([word_to_num[word]] + embedding)

        words_embeddings.append(word_to_num['UNK'] + np.random.rand(len(embedding)))
        words_embeddings = np.array(words_embeddings)
        print('debug ', words_embeddings.shape )
        # sort the array according to word index
        words_embeddings = words_embeddings[words_embeddings[:,0].argsort()]
        print(words_embeddings.shape," embedding loaded!")
        return words_embeddings

    def compare_num_pred_with_embedding(self, x, d):
        '''use pretrained embedding for softmax
        x        list of words, as indices, e.g.: [0, 4, 2]
        d        the desired verb and its (re)inflected form (singular/plural), as indices, e.g.: [7, 8]

        return 1 if p(d[0]) > p(d[1]), 0 otherwise
        '''
        _ , s = self.predict(x) # s shape (x+1, hidden)
        y = softmax(self.words_embeddings[:,1:].dot(s[len(x)-1,:]))
        return 1 if y[-1, d[0]] > y[-1, d[1]] else 0

    def compute_acc_lmnp(self, X_dev, D_dev):
        '''

        DO NOT CHANGE THIS

        X_dev            a list of input vectors, e.g.,         [[5, 4, 2], [7, 3, 8]]
        D_dev            a list of pair verb forms (plural/singular), e.g.,     [[4, 9], [6, 5]]
        '''
        acc = sum([self.compare_num_pred(X_dev[i], D_dev[i]) for i in range(len(X_dev))]) / len(X_dev)

        return acc


    def compute_mean_loss(self, X, D):
        '''
        compute the mean loss between predictions for corpus X and desired outputs in corpus D.

        X        corpus of sentences x1, x2, x3, [...], each a list of words as indices.
        D        corpus of desired outputs d1, d2, d3 [...], each a list of words as indices.

        return mean_loss        average loss over all words in D
        '''

        mean_loss = 0.

        ##########################
        # --- your code here --- #
        mean_loss = np.fromiter( (self.compute_loss(x, d) for x,d in zip(X,D)), float)
        dim = np.fromiter( (x.size for x in X), float)
        ##########################

        return mean_loss.sum()/dim.sum()


    def train(self, X, D, X_dev, D_dev, epochs=10, learning_rate=0.5, anneal=5, back_steps=0, batch_size=100, min_change=0.0001, log=True):
        '''
        train the RNN on some training set X, D while optimizing the loss on a dev set X_dev, D_dev

        DO NOT CHANGE THIS

        training stops after the first of the following is true:
            * number of epochs reached
            * minimum change observed for more than 2 consecutive epochs

        X                a list of input vectors, e.g.,         [[0, 4, 2], [1, 3, 0]]
        D                a list of desired outputs, e.g.,     [[4, 2, 3], [3, 0, 3]]
        X_dev            a list of input vectors, e.g.,         [[0, 4, 2], [1, 3, 0]]
        D_dev            a list of desired outputs, e.g.,     [[4, 2, 3], [3, 0, 3]]
        epochs            maximum number of epochs (iterations) over the training set. default 10
        learning_rate    initial learning rate for training. default 0.5
        anneal            positive integer. if > 0, lowers the learning rate in a harmonically after each epoch.
                        higher annealing rate means less change per epoch.
                        anneal=0 will not change the learning rate over time.
                        default 5
        back_steps        positive integer. number of timesteps for BPTT. if back_steps < 2, standard BP will be used. default 0
        batch_size        number of training instances to use before updating the RNN's weight matrices.
                        if set to 1, weights will be updated after each instance. if set to len(X), weights are only updated after each epoch.
                        default 100
        min_change        minimum change in loss between 2 epochs. if the change in loss is smaller than min_change, training stops regardless of
                        number of epochs left.
                        default 0.0001
        log                whether or not to print out log messages. (default log=True)
        '''
        if log:
            # stdout.write("\nTraining model for {0} epochs\ntraining set: {1} sentences (batch size {2})".format(epochs, len(X), batch_size))
            # stdout.write("\nOptimizing loss on {0} sentences".format(len(X_dev)))
            # stdout.write("\nVocab size: {0}\nHidden units: {1}".format(self.vocab_size, self.hidden_dims))
            # stdout.write("\nSteps for back propagation: {0}".format(back_steps))
            # stdout.write("\nInitial learning rate set to {0}, annealing set to {1}".format(learning_rate, anneal))
            # stdout.write("\n\ncalculating initial mean loss on dev set")
            # stdout.flush()
            ####################################################
            logging.info("Training model for {0} epochs\ntraining set: {1} sentences (batch size {2})".format(epochs, len(X), batch_size))
            logging.info("Optimizing loss on {0} sentences".format(len(X_dev)))
            logging.info("Vocab size: {0}\nHidden units: {1}".format(self.vocab_size, self.hidden_dims))
            logging.info("Steps for back propagation: {0}".format(back_steps))
            logging.info("Initial learning rate set to {0}, annealing set to {1}".format(learning_rate, anneal))
            logging.info("calculating initial mean loss on dev set")

        t_start = time.time()
        loss_function = self.compute_loss

        loss_sum = sum([len(d) for d in D_dev])
        initial_loss = sum([loss_function(X_dev[i], D_dev[i]) for i in range(len(X_dev))]) / loss_sum

        if log or not log:
            # stdout.write(": {0}\n".format(initial_loss))
            # stdout.flush()
            logging.info(": {0}\n".format(initial_loss))

        prev_loss = initial_loss
        loss_watch_count = -1
        min_change_count = -1

        a0 = learning_rate

        best_loss = initial_loss
        bestU, bestV, bestW = self.U, self.V, self.W
        best_epoch = 0

        for epoch in range(epochs):
            if anneal > 0:
                learning_rate = a0/((epoch+0.0+anneal)/anneal)
            else:
                learning_rate = a0

            if log:
                # stdout.write("\nepoch %d, learning rate %.04f" % (epoch+1, learning_rate))
                # stdout.flush()
                logging.info("epoch %d, learning rate %.04f" % (epoch+1, learning_rate))

            t0 = time.time()
            count = 0

            # use random sequence of instances in the training set (tries to avoid local maxima when training on batches)
            permutation = np.random.permutation(range(len(X)))
            if log:
                stdout.write("\tinstance 1")
            for i in range(len(X)):
                c = i+1
                if log:
                    stdout.write("\b"*len(str(i)))
                    stdout.write("{0}".format(c))
                    stdout.flush()
                p = permutation[i]
                x_p = X[p]
                d_p = D[p]

                y_p, s_p = self.predict(x_p)
                if back_steps == 0:
                    self.acc_deltas(x_p, d_p, y_p, s_p)
                else:
                    self.acc_deltas_bptt(x_p, d_p, y_p, s_p, back_steps)

                if i % batch_size == 0:
                    self.deltaU /= batch_size
                    self.deltaV /= batch_size
                    self.deltaW /= batch_size
                    self.apply_deltas(learning_rate)

            if len(X) % batch_size > 0:
                mod = len(X) % batch_size
                self.deltaU /= mod
                self.deltaV /= mod
                self.deltaW /= mod
                self.apply_deltas(learning_rate)

            loss = sum([loss_function(X_dev[i], D_dev[i]) for i in range(len(X_dev))])/loss_sum

            if log:
                # stdout.write("\tepoch done in %.02f seconds" % (time.time() - t0))
                # stdout.write("\tnew loss: {0}".format(loss))
                # stdout.flush()
                logging.info("\t epoch done in {0:.2f} seconds\tnew loss: {1}".format(
                                                            (time.time() - t0), loss))
                # logging.info("\tnew loss: {0}".format(loss))

            if loss < best_loss:
                best_loss = loss
                bestU, bestV, bestW = self.U.copy(), self.V.copy(), self.W.copy()
                best_epoch = epoch

            # make sure we change the RNN enough
            if abs(prev_loss - loss) < min_change:
                min_change_count += 1
            else:
                min_change_count = 0
            if min_change_count > 2:
                print("\n\ntraining finished after {0} epochs due to minimal change in loss".format(epoch+1))
                logging.info("\n\ntraining finished after {0} epochs due to minimal change in loss".format(epoch+1))
                break

            prev_loss = loss

        t = time.time() - t_start

        if min_change_count <= 2:
            # print("\n\ntraining finished after reaching maximum of {0} epochs".format(epochs))
            logging.info("\n\ntraining finished after reaching maximum of {0} epochs".format(epochs))

        # print("best observed loss was {0}, at epoch {1}".format(best_loss, (best_epoch+1)))
        logging.info("best observed loss was {0}, at epoch {1}".format(best_loss, (best_epoch+1)))

        print("setting U, V, W to matrices from best epoch")
        self.U, self.V, self.W = bestU, bestV, bestW

        return best_loss


    def train_np(self, X, D, X_dev, D_dev, epochs=10, learning_rate=0.5, anneal=5, back_steps=0, batch_size=100, min_change=0.0001, log=True):
        '''
        train the RNN on some training set X, D while optimizing the loss on a dev set X_dev, D_dev

        DO NOT CHANGE THIS

        training stops after the first of the following is true:
            * number of epochs reached
            * minimum change observed for more than 2 consecutive epochs

        X                a list of input vectors, e.g.,         [[5, 4, 2], [7, 3, 8]]
        D                a list of desired outputs, e.g.,     [[0], [1]]
        X_dev            a list of input vectors, e.g.,         [[5, 4, 2], [7, 3, 8]]
        D_dev            a list of desired outputs, e.g.,     [[0], [1]]
        epochs            maximum number of epochs (iterations) over the training set. default 10
        learning_rate    initial learning rate for training. default 0.5
        anneal            positive integer. if > 0, lowers the learning rate in a harmonically after each epoch.
                        higher annealing rate means less change per epoch.
                        anneal=0 will not change the learning rate over time.
                        default 5
        back_steps        positive integer. number of timesteps for BPTT. if back_steps < 2, standard BP will be used. default 0
        batch_size        number of training instances to use before updating the RNN's weight matrices.
                        if set to 1, weights will be updated after each instance. if set to len(X), weights are only updated after each epoch.
                        default 100
        min_change        minimum change in loss between 2 epochs. if the change in loss is smaller than min_change, training stops regardless of
                        number of epochs left.
                        default 0.0001
        log                whether or not to print out log messages. (default log=True)
        '''
        if log:
            # stdout.write("\nTraining model for {0} epochs\ntraining set: {1} sentences (batch size {2})".format(epochs, len(X), batch_size))
            # stdout.write("\nOptimizing loss on {0} sentences".format(len(X_dev)))
            # stdout.write("\nVocab size: {0}\nHidden units: {1}".format(self.vocab_size, self.hidden_dims))
            # stdout.write("\nSteps for back propagation: {0}".format(back_steps))
            # stdout.write("\nInitial learning rate set to {0}, annealing set to {1}".format(learning_rate, anneal))
            # stdout.flush()
            logging.info("Training model for {0} epochs\ntraining set: {1} sentences (batch size {2})".format(epochs, len(X), batch_size))
            logging.info("Optimizing loss on {0} sentences".format(len(X_dev)))
            logging.info("Vocab size: {0}\nHidden units: {1}".format(self.vocab_size, self.hidden_dims))
            logging.info("Steps for back propagation: {0}".format(back_steps))
            logging.info("Initial learning rate set to {0}, annealing set to {1}".format(learning_rate, anneal))

        t_start = time.time()
        loss_function = self.compute_loss_np

        loss_sum = len(D_dev)
        initial_loss = sum([loss_function(X_dev[i], D_dev[i]) for i in range(len(X_dev))]) / loss_sum
        initial_acc = sum([self.compute_acc_np(X_dev[i], D_dev[i]) for i in range(len(X_dev))]) / len(X_dev)

        if log or not log:
            # stdout.write("\n\ncalculating initial mean loss on dev set")
            # stdout.write(": {0}\n".format(initial_loss))
            # stdout.write("calculating initial acc on dev set")
            # stdout.write(": {0}\n".format(initial_acc))
            # stdout.flush()
            logging.info("\n\ncalculating initial mean loss on dev set")
            logging.info(": {0}\n".format(initial_loss))
            logging.info("calculating initial acc on dev set")
            logging.info(": {0}\n".format(initial_acc))

        prev_loss = initial_loss
        loss_watch_count = -1
        min_change_count = -1

        a0 = learning_rate

        best_loss = initial_loss
        bestU, bestV, bestW = self.U, self.V, self.W
        best_epoch = 0

        for epoch in range(epochs):
            if anneal > 0:
                learning_rate = a0/((epoch+0.0+anneal)/anneal)
            else:
                learning_rate = a0

            if log:
                # stdout.write("\nepoch %d, learning rate %.04f" % (epoch+1, learning_rate))
                # stdout.flush()
                logging.info("epoch %d, learning rate %.04f" % (epoch+1, learning_rate))


            t0 = time.time()
            count = 0

            # use random sequence of instances in the training set (tries to avoid local maxima when training on batches)
            permutation = np.random.permutation(range(len(X)))
            if log:
                stdout.write("\tinstance 1")
            for i in range(len(X)):
                c = i+1
                if log:
                    stdout.write("\b"*len(str(i)))
                    stdout.write("{0}".format(c))
                    stdout.flush()
                p = permutation[i]
                x_p = X[p]
                d_p = D[p]

                y_p, s_p = self.predict(x_p)
                if back_steps == 0:
                    self.acc_deltas_np(x_p, d_p, y_p, s_p)
                else:
                    self.acc_deltas_bptt_np(x_p, d_p, y_p, s_p, back_steps)

                if i % batch_size == 0:
                    self.deltaU /= batch_size
                    self.deltaV /= batch_size
                    self.deltaW /= batch_size
                    self.apply_deltas(learning_rate)

            if len(X) % batch_size > 0:
                mod = len(X) % batch_size
                self.deltaU /= mod
                self.deltaV /= mod
                self.deltaW /= mod
                self.apply_deltas(learning_rate)

            loss = sum([loss_function(X_dev[i], D_dev[i]) for i in range(len(X_dev))]) / loss_sum
            acc = sum([self.compute_acc_np(X_dev[i], D_dev[i]) for i in range(len(X_dev))]) / len(X_dev)

            if log:
                # stdout.write("\tepoch done in %.02f seconds" % (time.time() - t0))
                # stdout.write("\tnew loss: {0}".format(loss))
                # stdout.write("\tnew acc: {0}".format(acc))
                # stdout.flush()
                logging.info(
                "\t epoch done in {0:.2f} seconds\tnew loss: {1}\tnew acc: {2}".format(
                                            (time.time() - t0), loss, acc))


            if loss < best_loss:
                best_loss = loss
                best_acc = acc
                bestU, bestV, bestW = self.U.copy(), self.V.copy(), self.W.copy()
                best_epoch = epoch

            # make sure we change the RNN enough
            if abs(prev_loss - loss) < min_change:
                min_change_count += 1
            else:
                min_change_count = 0
            if min_change_count > 2:
                # print("\n\ntraining finished after {0} epochs due to minimal change in loss".format(epoch+1))
                logging.info("\n\ntraining finished after {0} epochs due to minimal change in loss".format(epoch+1))
                break

            prev_loss = loss

        t = time.time() - t_start

        if min_change_count <= 2:
            # print("\n\ntraining finished after reaching maximum of {0} epochs".format(epochs))
            logging.info("\n\ntraining finished after reaching maximum of {0} epochs".format(epochs))

        # print("best observed loss was {0}, acc {1}, at epoch {2}".format(best_loss, best_acc, (best_epoch+1)))
        logging.info("best observed loss was {0}, at epoch {1}".format(best_loss, (best_epoch+1)))

        print("setting U, V, W to matrices from best epoch")
        self.U, self.V, self.W = bestU, bestV, bestW

        return best_loss


if __name__ == "__main__":

    mode = sys.argv[1].lower()
    #data_folder = sys.argv[2]
    data_folder = "data"


    # np.random.seed(2018)

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_handler = logging.FileHandler(pjoin(output_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)
    run_loss = -1

    if mode == "train-lm":
        '''
        code for training language model.
        change this to different values, or use it to get you started with your own testing class
        '''
        epochs = int(sys.argv[2])
        save_models = int(sys.argv[3])
        train_size = 1000
        dev_size = 1000
        vocab_size = 2000

        # hdim = int(sys.argv[3])
        # lookback = int(sys.argv[4])
        # lr = float(sys.argv[5])

        # get the data set vocabulary
        vocab = pd.read_table(data_folder + "/vocab.wiki.txt", header=None, sep="\s+", index_col=0, names=['count', 'freq'], )
        num_to_word = dict(enumerate(vocab.index[:vocab_size]))
        word_to_num = invert_dict(num_to_word)

        # calculate loss vocabulary words due to vocab_size
        fraction_lost = fraq_loss(vocab, word_to_num, vocab_size)
        # print("Retained %d words from %d (%.02f%% of all tokens)\n" % (vocab_size, len(vocab), 100*(1-fraction_lost)))
        logging.info("Retained %d words from %d (%.02f%% of all tokens)\n" % (vocab_size, len(vocab), 100*(1-fraction_lost)))

        # load training data
        docs = load_lm_dataset(data_folder + '/wiki-train.txt')
        S_train = docs_to_indices(docs, word_to_num, 1, 1)
        X_train, D_train = seqs_to_lmXY(S_train)
        X_train = X_train[:train_size]
        D_train = D_train[:train_size]

        # Load the dev set (for tuning hyperparameters)
        docs = load_lm_dataset(data_folder + '/wiki-dev.txt')
        S_dev = docs_to_indices(docs, word_to_num, 1, 1)
        X_dev, D_dev = seqs_to_lmXY(S_dev)
        X_dev = X_dev[:dev_size]
        D_dev = D_dev[:dev_size]

        # q = best unigram frequency from omitted vocab
        # this is the best expected loss out of that set
        q = vocab.freq[vocab_size] / sum(vocab.freq[vocab_size:])

        ##########################
        # --- your code here --- #
        """ Perform parameter tuning
        Use a fixed vocabulary of size 2000, and vary the number of:
            hidden units (at least: 25, 50),
            the look-back in backpropagation (at least: 0, 2, 5),
            and learning rate (at least: 0.5, 0.1, 0.05).
        """
        hyper_params = [[50], [0], [0.5], [50], [0]]
        hyper_params = list(itertools.product(*hyper_params))
        logging.info("Parameter tuning of hidden_dims, lookback, lr, anneal: \n{}".format(
                                        hyper_params))
        logging.info("Total experiments {}".format(len(hyper_params)))
        for experiment, hyper_param in enumerate(hyper_params):
            logging.info("Experiment {}".format(experiment+1))
            np.random.seed(2018)
            hdim = hyper_param[0]
            lookback = hyper_param[1]
            lr = hyper_param[2]
            batch_size = hyper_param[3]
            anneal = hyper_param[4]
            rnn = RNN(vocab_size, hdim, vocab_size)
            run_loss = rnn.train(X_train, D_train, X_dev, D_dev, epochs = epochs,
                                  learning_rate = lr, anneal=anneal, back_steps = lookback,
                                  batch_size = batch_size)

        # Load the test set
        docs = load_lm_dataset(data_folder + '/wiki-test.txt')
        S_test = docs_to_indices(docs, word_to_num, 1, 1)
        X_test, D_test = seqs_to_lmXY(S_test)
        loss = rnn.compute_mean_loss(X_test, D_test)
        logging.info("Mean loss on the full test set: {}".format(loss))

        if save_models:
            np.save('rnn.U.npy', rnn.U)
            np.save('rnn.V.npy', rnn.V)
            np.save('rnn.W.npy', rnn.W)

        logging.info("Unadjusted: %.03f" % np.exp(loss))

        adjusted_loss = adjust_loss(loss, fraction_lost, q, mode='basic')
        logging.info("Adjusted for missing vocab: %.03f" % np.exp(adjusted_loss))
        logging.info("="*10)

        ##########################

    if mode == "train-np":
        '''
        starter code for parameter estimation.
        change this to different values, or use it to get you started with your own testing class
        '''
        epochs = int(sys.argv[2])
        train_size = 1000
        dev_size = 1000
        vocab_size = 2000

        # hdim = int(sys.argv[3])
        # lookback = int(sys.argv[4])
        # lr = float(sys.argv[5])

        # get the data set vocabulary
        vocab = pd.read_table(data_folder + "/vocab.wiki.txt", header=None, sep="\s+", index_col=0, names=['count', 'freq'], )
        num_to_word = dict(enumerate(vocab.index[:vocab_size]))
        word_to_num = invert_dict(num_to_word)

        # calculate loss vocabulary words due to vocab_size
        fraction_lost = fraq_loss(vocab, word_to_num, vocab_size)
        # print("Retained %d words from %d (%.02f%% of all tokens)\n" % (vocab_size, len(vocab), 100*(1-fraction_lost)))
        logging.info("Retained %d words from %d (%.02f%% of all tokens)\n" % (vocab_size, len(vocab), 100*(1-fraction_lost)))


        # load training data
        sents = load_np_dataset(data_folder + '/wiki-train.txt')
        S_train = docs_to_indices(sents, word_to_num, 0, 0)
        X_train, D_train = seqs_to_npXY(S_train)

        X_train = X_train[:train_size]
        Y_train = D_train[:train_size]

        # load development data
        sents = load_np_dataset(data_folder + '/wiki-dev.txt')
        S_dev = docs_to_indices(sents, word_to_num, 0, 0)
        X_dev, D_dev = seqs_to_npXY(S_dev)

        X_dev = X_dev[:dev_size]
        D_dev = D_dev[:dev_size]

        ##########################
        # --- your code here --- #
        hyper_params = [[50], [5], [0.5,], [50]]
        hyper_params = list(itertools.product(*hyper_params))
        logging.info("Parameter tuning of hidden_dims, lookback, lr: \n{}".format(
                                        hyper_params))
        logging.info("Total experiments {}".format(len(hyper_params)))

        min_loss = float("inf")
        for hyper_param in hyper_params:
            np.random.seed(2018)
            hdim = hyper_param[0]
            lookback = hyper_param[1]
            lr = hyper_param[2]
            batch_size = hyper_param[3]
            rnn = RNN(vocab_size, hdim, vocab_size)
            run_loss = rnn.train_np(X_train, D_train, X_dev, D_dev, epochs = epochs,
                                  learning_rate = lr, back_steps = lookback,
                                  batch_size = batch_size)
            if min_loss > run_loss:
                min_loss = run_loss
                best_model = rnn
                best_param = [hdim, lookback, lr]

        # Load the test set
        # sents = load_lm_dataset(data_folder + '/wiki-test.txt')
        # S_test = docs_to_indices(sents, word_to_num, 0, 0)
        # X_test, D_test = seqs_to_lmXY(S_test)
        sents = load_np_dataset(data_folder + '/wiki-train.txt')
        S_test = docs_to_indices(sents, word_to_num, 0, 0)
        X_test, D_test = seqs_to_npXY(S_test)
        # loss_function = best_model.compute_loss_np
        loss = sum([best_model.compute_loss_np(X_test[i], D_test[i]) for i in range(len(X_test))]) / len(D_test)
        acc = sum([best_model.compute_acc_np(X_test[i], D_test[i]) for i in range(len(X_test))]) / len(X_test)
        # acc = best_model.compute_acc_lmnp(X_test, D_test)
        logging.info("Best model with hidden_dims: {}, lookback: {}, lr: {}\n \
                      obatains loss {} and acc {} on test set".format(
                    best_param[0], best_param[1], best_param[2], loss, acc))

        logging.info("Accuracy: %.03f" % acc)
        logging.info("="*10)
        ##########################

        # print("Accuracy: %.03f" % acc)


    if mode == "predict-lm":

        # data_folder = sys.argv[2]
        # rnn_folder = sys.argv[3]
        # get saved RNN matrices and setup RNN
        # U,V,W = np.load(rnn_folder + "/rnn.U.npy"), np.load(rnn_folder + "/rnn.V.npy"), np.load(rnn_folder + "/rnn.W.npy")
        U,V,W = np.load("rnn.U.npy"), np.load("rnn.V.npy"), np.load("rnn.W.npy")

        vocab_size = len(V[0])
        hdim = len(U[0])

        dev_size = 1000

        r = RNN(vocab_size, hdim, vocab_size)
        r.U = U
        r.V = V
        r.W = W

        # get vocabulary
        vocab = pd.read_table(data_folder + "/vocab.wiki.txt", header=None, sep="\s+", index_col=0, names=['count', 'freq'], )
        num_to_word = dict(enumerate(vocab.index[:vocab_size]))
        word_to_num = invert_dict(num_to_word)

        # Load the dev set (for tuning hyperparameters)
        docs, dev_span = load_lm_np_dataset(data_folder + '/wiki-dev.txt')
        S_np_dev = docs_to_indices(docs, word_to_num, 1, 0)
        X_np_dev, D_np_dev = seqs_to_lmnpXY(S_np_dev)

        X_np_dev = X_np_dev[:dev_size]
        D_np_dev = D_np_dev[:dev_size]

        np_acc = r.compute_acc_lmnp(X_np_dev, D_np_dev)
        dev_span = np.concatenate((np.array(r.stats), np.array(dev_span, ndmin=2).T), axis = 1)
        np.savetxt('dev_span.csv', np.array(dev_span), delimiter=',')

        print('Number prediction accuracy:', np_acc)

        # load test data
        sents, test_span = load_lm_np_dataset(data_folder + '/wiki-test.txt')
        S_np_test = docs_to_indices(sents, word_to_num, 1, 0)
        X_np_test, D_np_test = seqs_to_lmnpXY(S_np_test)

        r.stats = [] # reset
        np_acc_test = r.compute_acc_lmnp(X_np_test, D_np_test)
        test_span = np.concatenate((np.array(r.stats), np.array(test_span, ndmin=2).T), axis = 1)
        np.savetxt('test_span.csv', np.array(test_span), delimiter=',')
        print('Number prediction accuracy on test set:', np_acc_test)

        # r.words_embeddings = r.load_embeddings(data_folder + "/glove.6B.50d.txt", word_to_num)
        # acc_embedding = sum([r.compare_num_pred_with_embedding(X_np_dev[i], D_np_dev[i]) for i in range(len(X_np_dev))]) / len(X_np_dev)
        # print('Number prediction accuracy with embedding:', acc_embedding)
