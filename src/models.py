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


class Net(nn.Module):
    def __init__(self, nb_input, nb_output):
        super(Net, self).__init__()
        # Dense
        self.fc1 = nn.Linear(nb_input, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, nb_output)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)
