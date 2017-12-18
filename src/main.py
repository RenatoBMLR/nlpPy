#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 19:12:46 2017

@author: renatobottermaiolopesrodrigues
"""

import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.

def removeTagsAndUris(x):
    
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
 
    
def get_data(subjects):
    data ={}
    for sub in subjects:
        df_aux = pd.read_csv('../data/' + sub + '.csv')
        data[sub]=df_aux
    return data


def tokenize_sentence(x):
    tokens = tokenizer.tokenize(x)
    return tokens
        
    

subjects=['biology', 'cooking', 'crypto', 'diy', 'robotics', 'travel']
data = get_data(subjects)
tokenizer = TweetTokenizer()


for key in data.keys():
    data[key]['content'] = data[key]['content'].apply(lambda x: stripTagsAndUris(x) )
    data[key]['tags'] = data[key]['tags'].apply(lambda x: x.split(' ') )
    data[key]['tokens'] = data[key]['content'].apply(lambda x: tokenize_sentence(x) )

    
