#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 19:26:19 2018

@author: renatobottermaiolopesrodrigues
"""

import itertools 
import collections
import pandas as pd
import networkx as nx



class WordsGraph():
    
    def __init__(self, df, col = 'tags', weight = 10):
            self.df = df
            self.col = col
            self.weight = weight
            self._getGraph()

    def _getGraph(self):

        self._words_count()
        self.d = self.words_df.unstack().reset_index()
        self.d.columns = ['source', 'target', 'weight']
        self.d = self.d[self.d.weight > self.weight]
        
        #create graph
        self.g = nx.from_pandas_dataframe(self.d, 'source', 'target', ['weight'])
     
    def _words_count(self):
                # Created DataFrame indexed on col
        words_lists = [t.strip().split() for t in self.df[self.col].values]
        self.words_df = pd.DataFrame(index=set(itertools.chain(*words_lists)))
    
        # For each category create a column and update the flag to col count
        for i, (name, group) in enumerate(self.df.groupby('subject')):
            self.words_df[name] = 0
            tmp_index, count = self._get_top_words(group)
            tmp = pd.Series(count, index=tmp_index)
            self.words_df[name].update(tmp)
    
        self.words_df['categories_appears'] = self.words_df.apply(lambda x: x.astype(bool).sum(), axis=1)
    
    
    def _get_top_words(self, df, n=None):
        words = list(itertools.chain(*[t.strip().split() for t in df[self.col].values]))
        top_words = collections.Counter(list(words)).most_common(n)
        words, count = zip(*top_words)
        return words, count
  