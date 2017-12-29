#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 22:51:44 2017

@author: renatobottermaiolopesrodrigues
"""
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
import glob


class TextDataset(Dataset):

    def __init__(self, subjects, root_dir, col_lst, transform=None, val_size=0.1,
                 is_valid=False, is_test=False, is_train = False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.data = pd.DataFrame()
        self.subjects = subjects
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = TweetTokenizer()
        self.get_data(col_lst)

        if ( (is_train) or (is_valid)):
            self.y = self.data[self.data.columns[:-1]].values

            if is_valid:
                self.split_data(val_size)
                
        if is_test:
            self.y = np.zeros([len(self.data), len(self.data.columns[:-1])])
            
    def split_data(self, val_size = 0.1):

        np.random.seed(4572)
        indices = range(len(self.data))

        ind = np.random.permutation(indices)
        split = np.round(val_size * len(self.data))
        index= np.array(ind[:int(split)])
        
        self.data = self.data.take(index,axis=0)
        self.y = self.y.take(index,axis=0)      

    def removeTagsAndUris(self, x):
    
        uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
        # BeautifulSoup on content
        soup = BeautifulSoup(x, "html.parser")
        # Stripping all <code> tags with their content if any
        if soup.code:
            soup.code.decompose()
        # Get all the text out of the html
        text =  soup.get_text()
        # Returning text stripping out all uris
        return re.sub(uri_re, "", text)

        
    def tokenize_sentence(self, x):
        tokens = self.tokenizer.tokenize(x)
        return tokens

    def get_data(self, col_lst = []):

        for file in glob.glob(self.root_dir + "/*.csv"):
            df_aux = pd.read_csv(file)
            
            if file.split('/')[-1][:-4] in self.subjects:
                sub = file.split('/')[-1][:-4]
            else:
                sub = None
                                
            df_aux['subject'] = sub
            df_aux['content'] = df_aux['content'].apply(lambda x: self.removeTagsAndUris(x) )
            df_aux['tokens'] = df_aux['content'].apply(lambda x: self.tokenize_sentence(x) )
    
            self.data=self.data.append(df_aux)
        if len(col_lst) != 0:
            self.data = self.data[col_lst]
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        textSample = self.data.iloc[idx, 0:-1].as_matrix()
        y = self.data.iloc[idx, -1]
        

        if self.transform:
            textSample = self.transform(textSample)

        return textSample, y


def create_dataLoader(dsets, batch_size, pin_memory =  False):

    dset_loaders = {}
    for key in dsets.keys():
        dset_loaders[key] = DataLoader(dsets[key], batch_size=batch_size, pin_memory=pin_memory)

    return dset_loaders    