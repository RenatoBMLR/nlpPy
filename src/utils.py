#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 22:51:44 2017

@author: renatobottermaiolopesrodrigues
"""
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

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

    def get_data(self):        
        for sub in self.subjects:
            df_aux = pd.read_csv(self.root_dir + sub + '.csv')
            df_aux['subject'] = sub
            self.data=self.data.append(df_aux)
            
    def __len__(self):
        return len(self.textData)

    def __getitem__(self, idx):
        
        textSample = self.data.iloc[idx, 0:].as_matrix()

        if self.transform:
            textSample = self.transform(textSample)

        return textSample
    