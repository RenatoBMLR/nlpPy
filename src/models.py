#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 20:20:11 2017

@author: renatobottermaiolopesrodrigues
"""

import torchvision
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import operator


class BoWClassifier(nn.Module):  

    def __init__(self, num_labels, vocab_size):

        super(BoWClassifier, self).__init__()


        self.linear = nn.Linear(vocab_size, num_labels)

        # The non-linearity log softmax does not have parameters

    def forward(self, bow_vec):
        # Pass the input through the linear layer,
        # then pass that through log_softmax.
        return F.log_softmax(self.linear(bow_vec), dim=1)

class Tfid():
    
    def __init__(self, data):
        self.frequency = []
        self.inverse_frequency = {}
        self.data = data

    def get_item(self, index):
        return sorted(self.frequency[index].items(), key=operator.itemgetter(1), reverse=True)
        
    def get_frequency(self):
    
        for i in range(len(self.data)):
            word_count = {}
    
            for word in self.data[i].split():
                if word in word_count:    
                    word_count[word] = word_count[word] + 1
                else:
                    word_count[word] = 1
                    
            for word in word_count:
                if word in self.inverse_frequency:
                    self.inverse_frequency[word] = self.inverse_frequency[word] + 1
                else:
                    self.inverse_frequency[word] = 1            
            self.frequency.append(word_count)
           

            
