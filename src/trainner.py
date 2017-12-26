#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 20:39:35 2017

@author: renatobottermaiolopesrodrigues
"""
import torch
from torch.autograd import Variable
import copy

class Trainner():
    
    def __init__(self, model = None, train_loader= None, test_loader= None ):
        
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader

    def fit(self, X, Y):
        Y_hat_oh = self.model(Variable(X) )
        values, Y_hat = torch.max(Y_hat_oh, 1)
        return Y_hat
    
    
    def getAccuracy(self, loader):
        correct = 0
        total = 0
        for data in loader:
            images, labels = data
            outputs = self.model(Variable(images))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.numpy() == labels.numpy()).sum()
        return 100 * correct / total
    
    
    def train(self, num_epochs= None, loss_fn= None, optimizer= None, patience= None):
        
        history_loss = []
        accuracy_train_history = []
        accuracy_test_history = []
        best_test_acc =  np.inf
        patience_count= 0
        for epoch in range(num_epochs):
            for i, data in enumerate(self.train_loader):
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                predict = self.model(inputs)
                print(predict)
                print('*****')
                predict(labels)
                loss = loss_fn(predict, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            accuracy_train_history.append(self.getAccuracy(self.train_loader) )
            accuracy_test_history.append( self.getAccuracy(self.test_loader) )
            history_loss.append(loss.data[0])
            print('Epoch:', epoch, 'train loss:', history_loss[-1], ' train acc: ', accuracy_train_history[-1],  ' test acc: ', accuracy_test_history[-1])
    
            
            #Early stopping
            if(best_test_acc < accuracy_test_history[-1]):
                patience_count = 0
                best_test_acc = accuracy_test_history[-1]
                best_model = copy.deepcopy(self.model)
    
            if(patience_count > patience):
                break;
    
            patience_count += 1
    
        print('Done!')
        return history_loss, accuracy_train_history, accuracy_test_history, best_model 