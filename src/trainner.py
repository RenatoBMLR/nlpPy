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


    def train(self, dset_loader, num_epochs=None, train_loader = None):
        
        ii_n = len(dset_loader)
        start_time = time.time()
        self.loss_lst = []
    
        for epoch in range(num_epochs):
            for i, (instance, label) in enumerate(dset_loader):
                bow_vec, target = Variable(instance), Variable(label)
                 

                log_probs = self.model(bow_vec)

                loss = self.criterion(log_probs, target)
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
        
                print('\rTrain: {}/{}'.format(i, ii_n - 1), end='')
            self.loss_lst.append(loss.data.numpy()[0])
            print(' -  Epoch: {}/{} Loss: {}'.format(epoch, num_epochs - 1, self.loss_lst[-1]))
            
        print('Execution time {0:.2f} s'.format(round(time.time() - start_time), 2))