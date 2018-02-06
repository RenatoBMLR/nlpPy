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
    
    subjects=['biology', 'cooking', 'crypto', 'diy', 'robotics', 'travel']
    path2data = '../data/'
    
    
    df = TextDataset(subjects, path2data + 'train/',  is_train = True)
    df.process_data(lemmalize = True, stem = False)
    
    for i, j in enumerate(df):
        print(i)
        print(j)
        break;
        
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