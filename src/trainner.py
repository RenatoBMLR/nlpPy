#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 20:39:35 2017

@author: renatobottermaiolopesrodrigues
"""
import torch
from torch.autograd import Variable
import copy
import torch.optim as optim
import time



class TrainnerNLP():
    
    def __init__(self, params):
        assert (params['model'] is not None) and (params['criterion'] is not None) and (params['optimizer'] is not None)
        self.model = params['model']
        self.criterion = params['criterion']
        self.optimizer = params['optimizer']
        self.words_ix = params['words_ix']
        self.label_ix = params['label_ix']     

    def make_bow_vector(self, sentence, word_to_ix):
        vec = torch.zeros(len(word_to_ix))
        for word in sentence:
            vec[word_to_ix[word]] += 1
        return vec.view(1, -1)

    def make_target(self, label, label_to_ix):
        return torch.LongTensor([label_to_ix[label]])


    def train(self, data, num_epochs=None, train_loader = None):
        
        ii_n = len(data)
        start_time = time.time()
    
        for epoch in range(num_epochs):
            for i, (instance, label) in enumerate(data):
                bow_vec = Variable(self.make_bow_vector(instance.split(), self.words_ix))
                target = Variable(self.make_target(label, self.label_ix))

                log_probs = self.model(bow_vec)

                # Step 4. Compute the loss, gradients, and update the parameters by
                # calling optimizer.step()
                loss = self.criterion(log_probs, target)
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
        
                print('\rTrain: {}/{}'.format(i, ii_n - 1), end='')
            print('-  Epoch: {}/{} ok'.format(epoch, num_epochs - 1))
            
        print('Execution time {0:.2f} s'.format(round(time.time() - start_time), 2))