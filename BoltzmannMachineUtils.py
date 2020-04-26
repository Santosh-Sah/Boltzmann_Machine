# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 18:03:51 2020

@author: Santosh Sah
"""

import pandas as pd
import numpy as np
import pickle

"""
Import training data set
"""
def importBoltzmannMachineTrainingDataset(boltzmannMachineTrainingDatasetFileName):
    
    boltzmannMachineTrainingDataset = pd.read_csv("ml-100k/u1.base", delimiter = '\t')
    
    boltzmannMachineTrainingDataset = np.array(boltzmannMachineTrainingDataset, dtype = 'int')
    
    return boltzmannMachineTrainingDataset

"""
Import testing data set
"""
def importBoltzmannMachineTestingDataset(boltzmannMachineTestingDatasetFileName):
    
    boltzmannMachineTestingDataset = pd.read_csv("ml-100k/u1.base", delimiter = '\t')
    
    boltzmannMachineTestingDataset = np.array(boltzmannMachineTestingDataset, dtype = 'int')
    
    return boltzmannMachineTestingDataset

"""
Converting the dataset into an array with users in lines and movies in columns. Rating will be in each cell
"""

def convertUsersInLineMoviesInColumns(dataset, maximnumNumberOfUsers, maximumNumberOfMovies):
    
    usersInLineMoviesInColumns = []
    
    for id_users in range(1, maximnumNumberOfUsers + 1):
        
        #creating a list of ratings given by the users to all the movies.
        #In the dataset the information available for rating given by all the users.
        #We need to get the list of rating for each user for every movie and create a list.
        id_rating = dataset[:, 2][dataset[:, 0] == id_users]
        
        #creating a list of movies to which rating given by users.
        #In the dataset the information available for movies given by all the users.
        #We need to get the list of movies.
        id_movies = dataset[:, 1][dataset[:, 0] == id_users]
        
        #There are many movies present in the list to which user did not give any rating.
        #We need to put zero if the user did not give any rating to the movies
        
        #Initialize the list of rating with zero
        ratings = np.zeros(maximumNumberOfMovies)
        
        #Fill the rating of the users given to movies. If rating not given then the value will be zero.
        #Rating start with 0 but numpy starts with 0 hence we need do -1
        ratings[id_movies - 1] = id_rating
        
        usersInLineMoviesInColumns.append(list(ratings))
    
    return usersInLineMoviesInColumns

"""
Converting rating into binary rating 0 and 1
"""
def convertRatingIntoBinaryOneAndZero(dataset):
    
    dataset[dataset == 0] = -1
    dataset[dataset == 1] = 0
    dataset[dataset == 2] = 0
    dataset[dataset >= 3] = 1
    
    return dataset
    
"""
Save training and testing dataset
"""
def saveTrainingAndTestingDataset(X_train, X_test):
    
    #Write X_train in a picke file
    with open("X_train.pkl",'wb') as X_train_Pickle:
        pickle.dump(X_train, X_train_Pickle, protocol = 2)
    
    #Write X_test in a picke file
    with open("X_test.pkl",'wb') as X_test_Pickle:
        pickle.dump(X_test, X_test_Pickle, protocol = 2)
    

"""
Save SupportVectorMachineModel as a pickle file.
"""
def saveSupportVectorMachineModel(supportVectorMachineModel):
    
    #Write SupportVectorMachineModel as a picke file
    with open("SupportVectorMachineModel.pkl",'wb') as SupportVectorMachineModel_Pickle:
        pickle.dump(supportVectorMachineModel, SupportVectorMachineModel_Pickle, protocol = 2)

"""
read supportVectorMachineModel from pickle file
"""
def readSupportVectorMachineModel():
    
    #load SupportVectorMachineModel model
    with open("SupportVectorMachineModel.pkl","rb") as SupportVectorMachineModel:
        supportVectorMachineModel = pickle.load(SupportVectorMachineModel)
    
    return supportVectorMachineModel

"""
read X_train from pickle file
"""
def readBoltzmannMachineXTrain():
    
    #load X_train
    with open("X_train.pkl","rb") as X_train_pickle:
        X_train = pickle.load(X_train_pickle)
    
    return X_train

"""
read X_test from pickle file
"""
def readBoltzmannMachineXTest():
    
    #load X_test
    with open("X_test.pkl","rb") as X_test_pickle:
        X_test = pickle.load(X_test_pickle)
    
    return X_test

"""
Save NumberOfusers as a pickle file.
"""
def saveNumberOfusers(numberOfusers):
    
    #Write NumberOfusers as a picke file
    with open("NumberOfusers.pkl",'wb') as NumberOfusers_Pickle:
        pickle.dump(numberOfusers, NumberOfusers_Pickle, protocol = 2)

"""
read NumberOfusers from pickle file
"""
def readNumberOfusers():
    
    #load NumberOfusers model
    with open("NumberOfusers.pkl","rb") as NumberOfusers_pickle:
        numberOfusers = pickle.load(NumberOfusers_pickle)
    
    return numberOfusers
