#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 22:51:44 2017

@author: renatobottermaiolopesrodrigues
"""
import torch
import pandas as pd
from bs4 import BeautifulSoup
import re
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.


class TextDataset(Dataset):

    def __init__(self, subjects, root_dir, transform=None):
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

    def get_data(self):        
        for sub in self.subjects:
            df_aux = pd.read_csv(self.root_dir + sub + '.csv')
            df_aux['subject'] = sub
            df_aux['content'] = df_aux['content'].apply(lambda x: self.removeTagsAndUris(x) )
            df_aux['tokens'] = df_aux['content'].apply(lambda x: self.tokenize_sentence(x) )

            self.data=self.data.append(df_aux)
            
    def __len__(self):
        return len(self.textData)

    def __getitem__(self, idx):
        
        textSample = self.data.iloc[idx, 0:].as_matrix()

        if self.transform:
            textSample = self.transform(textSample)

        return textSample
    