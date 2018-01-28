o#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 19:12:46 2017

@author: renatobottermaiolopesrodrigues
"""

import pandas as pd
import numpy as np
from utils import TextDataset
from models import Net

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

    tdataset = TextDataset(subjects, path2data + 'train/', col_lst = ['tokens', 'subject'] , is_train = True)
    
