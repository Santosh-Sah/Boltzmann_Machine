# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 18:05:21 2020

@author: Santosh Sah
"""
import torch
from BoltzmannMachineUtils import (importBoltzmannMachineTrainingDataset, importBoltzmannMachineTestingDataset, convertUsersInLineMoviesInColumns,
                                   convertRatingIntoBinaryOneAndZero, saveTrainingAndTestingDataset, saveNumberOfusers)

def preprocess():
    
    boltzmannMachineTrainingDataset =importBoltzmannMachineTrainingDataset("ml-100k/u1.base")
    boltzmannMachineTestingDataset =importBoltzmannMachineTestingDataset("ml-100k/u1.test")
    
    # Getting the number of users and movies
    numberOfUsers = int(max(max(boltzmannMachineTrainingDataset[:,0]), max(boltzmannMachineTestingDataset[:,0])))
    numberOfMovies = int(max(max(boltzmannMachineTrainingDataset[:,1]), max(boltzmannMachineTestingDataset[:,1])))
    
    boltzmannMachineTrainingSet = convertUsersInLineMoviesInColumns(boltzmannMachineTrainingDataset, numberOfUsers, numberOfMovies)
    boltzmannMachineTestinggSet = convertUsersInLineMoviesInColumns(boltzmannMachineTestingDataset, numberOfUsers, numberOfMovies)
    
    #converting training data into torch tensors
    boltzmannMachineTrainingSet = torch.FloatTensor(boltzmannMachineTrainingSet)
    
    #converting testing data into torch tensors
    boltzmannMachineTestinggSet = torch.FloatTensor(boltzmannMachineTestinggSet)
    
    #convert rating in training dataset to binary 0 and 1
    boltzmannMachineTrainingSet = convertRatingIntoBinaryOneAndZero(boltzmannMachineTrainingSet)
    
    #convert rating in testing dataset to binary 0 and 1
    boltzmannMachineTestinggSet = convertRatingIntoBinaryOneAndZero(boltzmannMachineTestinggSet)
    
    saveTrainingAndTestingDataset(boltzmannMachineTrainingSet, boltzmannMachineTestinggSet)
    
    saveNumberOfusers(numberOfUsers)

    
if __name__ == "__main__":
    preprocess()