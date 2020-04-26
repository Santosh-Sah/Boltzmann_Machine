# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 18:05:41 2020

@author: Santosh Sah
"""
import torch
import torch.nn.parallel
import torch.utils.data
from BoltzmannMachineUtils import (readBoltzmannMachineXTrain, readNumberOfusers)

#creating the architecture of restricted boltzmann machine
class RestrictedBoltzmannMachineArchitecture():
    
    def __init__(self, numberOfVisisbleNodes, numberOfHiddenNodes):
        
        #initialize weights and biases
        self.weights = torch.randn(numberOfHiddenNodes, numberOfVisisbleNodes)
        self.biasForHiddenNodes = torch.randn(1, numberOfHiddenNodes)
        self.biasForVisibleNodes = torch.randn(1, numberOfVisisbleNodes)
    
    #We are going to do sampling of the hidden nodes based on the concept of probability of hidden nodes given visible nodes.
    #probability of hidden nodes given visible nodes is nothing but the sigmoid activation function
    #We will be doing sampling based on the Gibb's sampling concept.
    #To apply Gibb's sampling we need to know the probability of hidden nodes given visible nodes
    #If the hidden nodes are 100 then based on the Gibb's sampling it will sample some nodes and activate it.
    def samplingTheHiddenNodes(self, visibleNodes):
        
        #calculate the probability of hidden nodes given visible nodes. It is nothing but the sigmoid activation function.
        #How sigmoid function will applyt. It will be the product of the weights and the visibleNodes
        #plus the the biases of the hidden nodes
        productOfBiasesOfHiddenNodesAndvisibleNodes = torch.mm(visibleNodes, self.weights.t())
        
        #calculate the activation function. In this case the activation function will of sigmoid type.
        #Each input vector is not treated independently but it will treated as a batch.
        #When we applied the bias then we should make sure that the bias is applied to each mini batch.
        #For this we need to apply a function expand_as to achieve this.
        activation = productOfBiasesOfHiddenNodesAndvisibleNodes + self.biasForHiddenNodes.expand_as(productOfBiasesOfHiddenNodesAndvisibleNodes)
        
        #calculate the sigmoid activation function
        probabilityOfHiddenNodesGivenVisibleNodes = torch.sigmoid(activation)
        
        #We need to return the sample of hidden nodes based on this sigmoid function and this sigmoid activation function is based on the 
        #probability of hidden nodes given visible nodes. The sampling will be dones based on the Bernui sampling.
        #Based on this sampling some of the hidden nodes will activated and some of the nodes will not activated. Activation and not activation is based on the
        #probabilityof hidden nodes given visible nodes vectors.
        samplingOfHiddenNuerons = torch.bernoulli(probabilityOfHiddenNodesGivenVisibleNodes)
        
        return probabilityOfHiddenNodesGivenVisibleNodes, samplingOfHiddenNuerons
    
    #We are going to do sampling of the visible nodes based on the concept of probability of visible nodes given hidden nodes.
    #probability of visible nodes given hidden nodes is nothing but the sigmoid activation function
    #We will be doing sampling based on the Gibb's sampling concept.
    #To apply Gibb's sampling we need to know the probability of visible nodes given hidden nodes
    #If the visible nodes are 100 then based on the Gibb's sampling it will sample some nodes and activate it.
    def samplingTheVisibleNodes(self, hiddenNodes):
        
        #calculate the probability of visible nodes given hidden nodes. It is nothing but the sigmoid activation function.
        #How sigmoid function will applyt. It will be the product of the weights and the hiddenNodes
        #plus the the biases of the visible  nodes
        productOfBiasesOfVisibleNodesAndHiddenNodes = torch.mm(hiddenNodes, self.weights)
        
        #calculate the activation function. In this case the activation function will of sigmoid type.
        #Each input vector is not treated independently but it will treated as a batch.
        #When we applied the bias then we should make sure that the bias is applied to each mini batch.
        #For this we need to apply a function expand_as to achieve this.
        activation = productOfBiasesOfVisibleNodesAndHiddenNodes + self.biasForVisibleNodes.expand_as(productOfBiasesOfVisibleNodesAndHiddenNodes)
        
        #calculate the sigmoid activation function
        probabilityOfVisibleNodesGivenHiddenNodes = torch.sigmoid(activation)
        
        #We need to return the sample of visible nodes based on this sigmoid function and this sigmoid activation function is based on the 
        #probability of visible nodes given hidden nodes. The sampling will be dones based on the Bernui sampling.
        #Based on this sampling some of the visible nodes will activated and some of the nodes will not activated. Activation and not activation is based on the
        #probabilityof visible nodes given hidden nodes vectors.
        samplingOfVisibleNuerons = torch.bernoulli(probabilityOfVisibleNodesGivenHiddenNodes)
        
        return probabilityOfVisibleNodesGivenHiddenNodes, samplingOfVisibleNuerons
    
    #calculate the Constrastive Divergence function which is used to approximate the restricted boltzmann machine log-likelihood gradient.
    #restricted boltzmann machine is energy based model. Thisis the energy function we are trying to minimise. Since this energyfunction depends on the weights
    #of the model. We need to optimize these weights to mininise the energy. The goal would be alsoto maximize the log likelihood of the training set.
    #To minimise the enegry or to maximize the log likelihood we need to calculate the gradient. The algorithm whichwill use to achieve this is called
    #Constrastive Divergence.
    #inputVector contains all the rating of the movies given by a user.
    def RestrictedBoltzmannMachine(self, inputVectorAsVisibleNodes, visibleNodesAfterKSampling, vectorOfProbabilityOfHiddenNodesGivenVectorVisibleNodes, 
                                   probabilityOfHiddenNodesAfterKSamplingGivenvisibleNodesAfterKSampling):
        #update the tensor weight.
        self.weights += (torch.mm(inputVectorAsVisibleNodes.t(), vectorOfProbabilityOfHiddenNodesGivenVectorVisibleNodes) - 
                         torch.mm(visibleNodesAfterKSampling.t(), probabilityOfHiddenNodesAfterKSamplingGivenvisibleNodesAfterKSampling)).t()
        
        #update the biases of hidden nodes.
        self.biasForHiddenNodes += torch.sum((vectorOfProbabilityOfHiddenNodesGivenVectorVisibleNodes - probabilityOfHiddenNodesAfterKSamplingGivenvisibleNodesAfterKSampling), 0)
        
        #updates thebiases of visible nodes
        self.biasForVisibleNodes += torch.sum((inputVectorAsVisibleNodes - visibleNodesAfterKSampling), 0)

def trainRestrictedBoltzmannMachine():
    
    boltzmannMachineTrainingSet = readBoltzmannMachineXTrain()
    
    #get the number of visible nodes. In our case the number of training obervation will be the number of visible nodes.
    numberOfVisibleNodes = len(boltzmannMachineTrainingSet[0])
    
    #get the number of hidden nodes. Hidden nodes will be the number of features detected by the models.
    #If we talk about the features of the movies would be actors, type of movies, directors, award given to those movies.
    #Lets we want to detect 100 features
    #The number of hidden nodes can be tunnable. We can try to create different model based on the different hidden nodes.
    numberOfHiddenNodes = 100
    
    #Batch size. When the model will updates the weights then it will not updates the weights for a sungle obeservation.
    #It will update theweights for a batch and each batch contains many obeservation.
    #The number of batch size can be tunnable. We can try to create different model based on the different batch size.
    batch_size = 100
    
    restrictedBoltzmannMachine = RestrictedBoltzmannMachineArchitecture(numberOfVisibleNodes, numberOfHiddenNodes)
    
    numberOfEpoch = 10
    
    numberOfusers = readNumberOfusers()
    
    for epoch in range(1, numberOfEpoch + 1):
        
        #loss function to measure the error
        trainingLoss = 0
        
        #normalize the training loss requred to divide the training loss with counter.
        #initialize the counter
        counter = 0.
        
        for id_user in range(0, numberOfusers - batch_size, batch_size):
            
            visibleNodesAfterKSampling = boltzmannMachineTrainingSet[id_user:id_user+batch_size]
            inputVectorAsVisibleNodes = boltzmannMachineTrainingSet[id_user:id_user+batch_size]
            vectorOfProbabilityOfHiddenNodesGivenVectorVisibleNodes, _ = restrictedBoltzmannMachine.samplingTheHiddenNodes(inputVectorAsVisibleNodes)
            
            #Gibb's sampling is based on the visible nodes to hidden nodes and hidden nodes to visible nodes.
            #each round of the Gibb's chain the visible nodes gets updated.
            for k in range(10):
                
                #updating the hidden nodes. Updation of the hidden nodes starts after the first iteration.
                _, samplingOfHiddenNuerons = restrictedBoltzmannMachine.samplingTheHiddenNodes(visibleNodesAfterKSampling)
                
                #updating the visible nodes
                _, visibleNodesAfterKSampling = restrictedBoltzmannMachine.samplingTheVisibleNodes(samplingOfHiddenNuerons)
                
                #freeze the visible nodes which contains -1 as a rating. It is not possible to update the visible nodes which has -1 as rating.
                visibleNodesAfterKSampling[inputVectorAsVisibleNodes < 0] = inputVectorAsVisibleNodes[inputVectorAsVisibleNodes < 0]
            
            
            probabilityOfHiddenNodesAfterKSamplingGivenvisibleNodesAfterKSampling, _ = restrictedBoltzmannMachine.samplingTheHiddenNodes(visibleNodesAfterKSampling)
            
            #train the model
            restrictedBoltzmannMachine.RestrictedBoltzmannMachine(inputVectorAsVisibleNodes, visibleNodesAfterKSampling, vectorOfProbabilityOfHiddenNodesGivenVectorVisibleNodes, 
                                   probabilityOfHiddenNodesAfterKSamplingGivenvisibleNodesAfterKSampling)
            
            trainingLoss += torch.mean(torch.abs(inputVectorAsVisibleNodes[inputVectorAsVisibleNodes >= 0] - 
                                                 visibleNodesAfterKSampling[inputVectorAsVisibleNodes >= 0]))
                        
            counter += 1.
        
        print("epoch: "  + str(epoch) + " training loss: " + str(trainingLoss/counter))
        """
        epoch: 1 training loss: tensor(0.3858)
        epoch: 2 training loss: tensor(0.2610)
        epoch: 3 training loss: tensor(0.2521)
        epoch: 4 training loss: tensor(0.2521)
        epoch: 5 training loss: tensor(0.2479)
        epoch: 6 training loss: tensor(0.2491)
        epoch: 7 training loss: tensor(0.2483)
        epoch: 8 training loss: tensor(0.2455)
        epoch: 9 training loss: tensor(0.2463)
        epoch: 10 training loss: tensor(0.2476)
        """

if __name__ == "__main__":
    trainRestrictedBoltzmannMachine()
        
    