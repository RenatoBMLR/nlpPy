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
from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
import glob
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import os
import torch
from torch.utils.data import Dataset, DataLoader

def create_dataLoader(dsets, batch_size,  pin_memory=False, use_shuffle=False):

    dset_loaders = {}

    shuffle = False
    for key in dsets.keys():
        if use_shuffle:
            if key != 'test':
                shuffle = True
            else:
                shuffle = False
        dset_loaders[key] = DataLoader(dsets[key], batch_size=batch_size, pin_memory=pin_memory, shuffle = shuffle)

    return dset_loaders


class TextTorchDataset(Dataset):

    def __init__(self, x, y, is_test = False, transform=None):
        self.x = x
        if is_test:
            self.y = np.zeros(len(self.x))
        else:
            self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        x = self.x[idx]
        y = self.y[idx]

        if self.transform is not None:
            x = self.transform(x)

        return x, y


class TextDataset():

    def __init__(self, root_dir, val_size=0.1,
                 extension='.csv', sep=',', is_valid=False, is_test=False, is_train = False, lang = 'english'):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.DataFrame()
        self.root_dir = root_dir

        self.tokenizer = TweetTokenizer()
        self.stop_words = set(stopwords.words(lang))
        self.lemmatizer = WordNetLemmatizer()
        self.ps = PorterStemmer()

        self._read_data(extension, sep)
        #self.data = (self.data if len(col_lst) else self.data[col_lst])

        if ( (is_train) or (is_valid)):
            self.y = self.data[self.data.columns[:-1]].values

            if is_valid:
                self.split_data(val_size)

        elif is_test:
            self.y = np.zeros([len(self.data), len(self.data.columns[:-1])])

    def split_data(self, val_size = 0.1):

        np.random.seed(4572)
        indices = range(len(self.data))

        ind = np.random.permutation(indices)
        split = np.round(val_size * len(self.data))
        index= np.array(ind[:int(split)])

        self.data = self.data.take(index,axis=0)
        self.y = self.y.take(index,axis=0)

    def _removeStopwords(self, words):
        # Removing all the stopwords
        return " ".join([word for word in words.split() if word not in self.stop_words])


    def _removePonctuation(self, words):
        words = words.lower()
        return re.sub(r'[^\w\s]', '', words)

    def _lemmatizing(self, words):
        #Lemmatizing
        return ' '.join(self.lemmatizer.lemmatize(word) for word in words.split())

    def _stemming(self, words):
        #Stemming
        return ' '.join(self.ps(word) for word in words.split())

    def _removeTagsAndUris(self, x):

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

    def process_data(self, col = 'content', remove_pontuation = True,  remove_stopw = False, remove_tags = False, lemmalize = False, stem = False):

        if remove_pontuation:
            self.data[col + '_data'] = self.data[col].apply(lambda x: self._removePonctuation(x) )
        if remove_stopw:
            self.data[col + '_data'] = self.data[col + '_data'].apply(lambda x: self._removeStopwords(x)) 
        if remove_tags:
            self.data[col + '_data'] = self.data[col + '_data'].apply(lambda x: self._removeTagsAndUris(x) )
        if lemmalize:
            self.data[col + '_data'] = self.data[col + '_data'].apply(lambda x: self._lemmatizing(x) )

        if stem:
            self.data[col + '_data'] = self.data[col + '_data'].apply(lambda x: self._stemming(x))

        #self.data[col + '_data'] = self.data[col + '_data'].apply(lambda x: ' '.join(set(x.split())))
        self.data[col + '_data'] = self.data[col + '_data'].apply(lambda x: list(x.split()))

    def _read_data(self, extension, sep):

        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                filename, file_extension = os.path.splitext(os.path.join(root, file))
                if file_extension == extension:
                    df_aux = pd.read_csv(os.path.join(root, file), sep=sep)
                    sub = file.split('/')[-1][:-4]
                    df_aux['subject'] = sub
                    self.data=self.data.append(df_aux)

    def make_bow_vector(self, col, words_ix):

        bow_vector = []
        for i, sentence in self.data[col].iteritems():

            vec = torch.zeros(len(words_ix))
            for word in sentence.split():
                vec[words_ix[word]] += 1
            bow_vector.append(vec.view(1, -1))
        return bow_vector

    def get_data(self, x_col, words_ix, y_col = []):

        x = self.make_bow_vector(x_col, words_ix)
        y=self.data[y_col]
        return (x,y)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        textSample = self.data['data'].iloc[idx]
        y = self.data['subject'].iloc[idx]

        return textSample, y
