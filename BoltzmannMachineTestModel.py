# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 18:38:00 2020

@author: Santosh Sah
"""
import torch
from BoltzmannMachineTrainModel import RestrictedBoltzmannMachineArchitecture
from BoltzmannMachineUtils import (readBoltzmannMachineXTrain, readNumberOfusers, readBoltzmannMachineXTest)

def testRestrictedBoltzmannMachine():
    
    boltzmannMachineTrainingSet = readBoltzmannMachineXTrain()
    boltzmannMachineTestingSet = readBoltzmannMachineXTest()
    
    numberOfusers = readNumberOfusers()
    
    #loss function to measure the error
    testingLoss = 0
        
    #normalize the training loss requred to divide the training loss with counter.
    #initialize the counter
    counter = 0.
    
    #get the number of visible nodes. In our case the number of training obervation will be the number of visible nodes.
    numberOfVisibleNodes = len(boltzmannMachineTrainingSet[0])
    
    #get the number of hidden nodes. Hidden nodes will be the number of features detected by the models.
    #If we talk about the features of the movies would be actors, type of movies, directors, award given to those movies.
    #Lets we want to detect 100 features
    #The number of hidden nodes can be tunnable. We can try to create different model based on the different hidden nodes.
    numberOfHiddenNodes = 100
    
    restrictedBoltzmannMachine = RestrictedBoltzmannMachineArchitecture(numberOfVisibleNodes, numberOfHiddenNodes)
    
    for id_user in range(numberOfusers):
        
        input_set = boltzmannMachineTrainingSet[id_user:id_user+1]
        target_set = boltzmannMachineTestingSet[id_user:id_user+1]
        
        if len(target_set[target_set >= 0]) > 0:
            
            #updating the hidden nodes. Updation of the hidden nodes starts after the first iteration.
            _, samplingOfHiddenNuerons = restrictedBoltzmannMachine.samplingTheHiddenNodes(input_set)
                
            #updating the visible nodes
            _, input_set = restrictedBoltzmannMachine.samplingTheVisibleNodes(samplingOfHiddenNuerons)
            
            testingLoss += torch.mean(torch.abs(target_set[target_set >= 0] - 
                                                 input_set[target_set >= 0]))
            
            counter += 1.
            
    print("training loss: " + str(testingLoss/counter))

if __name__ == "__main__":
    testRestrictedBoltzmannMachine()
