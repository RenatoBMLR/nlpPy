#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 19:12:46 2017

@author: renatobottermaiolopesrodrigues
"""

import pandas as pd
import numpy as np
from torchvision import transforms
from textDataset import *

from models import BoWClassifier, Tfid

import torch
import torch.nn as nn
import torch.optim as optim

from trainner import TrainnerNLP


if __name__ == '__main__':
        
    '''
    subjects=['biology', 'cooking', 'crypto', 'diy', 'robotics', 'travel']
    path2data = '../data/'
    textdset = TextDataset(subjects, path2data )
    textdset.get_data(col_lst = ['tokens', 'subject'])
    df = textdset.data
    print(df.head)
    '''
    '''
    Net = Net(1000, 10)
    print(Net)
    '''
    '''
    subjects=['biology', 'cooking', 'crypto', 'diy', 'robotics', 'travel']
    path2data = '../data/'
    
    
    df = TextDataset(subjects, path2data + 'train/',  is_train = True)
    df.process_data(lemmalize = True, stem = False)
    
    for i, j in enumerate(df):
        print(i)
        print(j)
        break;
    '''
        
    '''
    
    
    data = df.data
    dsets = {}
    dsets['train'] = df
    
    batch_size = 32
    dset_loaders = create_dataLoader(dsets, batch_size)
    inputs, cls = next(iter(dset_loaders['train']))
    
    
    
    params = {'model' : model.mrnd, 
    'criterion': loss_fn,  
    'optimizer': optimizer, 
    'callbacks': [savebest, ptt.AccuracyMetric()] #ptt.PlotCallback(),
    }
    '''
    
    path2data = '../data/toxic/'

    text = {
            'train': TextProcessing(path2data + 'train/', is_train = True),
            'test':  TextProcessing(path2data + 'test/',  is_test=True)
    }

    text['train'].process_data(col = 'comment_text', lemmalize = False, stem = False)
    
    col = 'comment_text_data'
    
    # index into the Bag of words vector
    words_ix = {}
    words_voc = []
    for index, row in text['train'].data[col].iteritems():
        for word in row.split():
            words_voc.append(word)
            if word not in words_ix:
                words_ix[word] = len(words_ix)
        
    VOCAB_SIZE = len(words_ix)
    NUM_LABELS = 6
    print('VOCAB_SIZE: {} NUM_LABELS: {}'.format(VOCAB_SIZE, NUM_LABELS))


    model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)
    
    loss_fn = torch.nn.MultiLabelMarginLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    params = {'model' : model, 
              'criterion': loss_fn,  
              'optimizer': optimizer, 
    }
    
    y_train = text['train'].data[['toxic', 'severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate']].values
        
    x_train = np.vstack( text['train'].make_bow_vector('comment_text_data', words_ix))

    text['test'].process_data(col = 'comment_text', lemmalize = False, stem = False)
    x_test = np.vstack( text['test'].make_bow_vector('comment_text_data', words_ix))
                      
    
    dsets = {
            'train': TextDataset(x_train, y_train),
            'test': TextDataset(x_train, is_test = True)
            }
    dset_loaders = create_dataLoader(dsets, 10, pin_memory=False, use_shuffle= True)
    
    NLPtrainner = TrainnerNLP(params)
    NLPtrainner.train(dset_loaders['train'], 10)
    
    NLPtrainner.predict(dset_loaders['test'], 10)
    
    

