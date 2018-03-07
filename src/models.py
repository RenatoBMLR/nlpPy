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
        print('Initializing BoWClassifier')

        self.fc1 = torch.nn.Linear(vocab_size, 2*num_labels)
        self.at1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(2*num_labels, num_labels)
        self.at2 = torch.nn.Sigmoid()

    def forward(self, x):

        x = self.fc1(x)
        x = self.at1(x)
        x = self.fc2(x)
        x=self.at2(x)
        return x


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
           

            
