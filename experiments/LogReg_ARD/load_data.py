"""
Load UCI datasets and preprocess

"""

import numpy as np
import pandas as pd
from scipy.io import arff
import random
import pickle
from sklearn.model_selection import train_test_split

def arcene_load(seed):
    np.random.seed(seed)

    x_train = pd.read_csv('./data/arcene_train.data', delim_whitespace= True, header = None)
    y_train = pd.read_csv('./data/arcene_train.labels', delim_whitespace= True, header = None)

    x_test = pd.read_csv('./data/arcene_valid.data', delim_whitespace= True, header = None)
    y_test = pd.read_csv('./data/arcene_valid.labels', delim_whitespace= True,header = None)

    #set y to 0 and 1
    y_train[y_train == -1] =0
    y_test[y_test == -1] =0


    #concatenate and resplit
    x = np.concatenate((x_train,x_test),axis = 0)
    y = np.concatenate((y_train,y_test))
    x_train,x_test, y_train,y_test = train_test_split(x,y[:,0],test_size = 0.5, stratify = y)
    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)



    #normalize data by mean and std
    x_train = (x_train - x_train.mean(axis = 0))/x_train.std(axis = 0)
    x_test = (x_test - x_test.mean(axis = 0))/x_test.std(axis = 0)

    #impute Nans with means
    x_train.fillna(0,inplace = True)
    x_test.fillna(0,inplace = True)

    N_train = np.shape(x_train)[0]
    N_test = np.shape(x_test)[0]
    D = np.shape(x_train)[1]


    #convert to dictionary and save
    ar_data_train= {'y': y_train, 'x': x_train, 'N':N_train,'D':D}
    ar_data_test= {'y': y_test, 'x': x_test, 'N':N_test,'D':D}

    with open('./data/ar_train_seed{}'.format(seed), 'wb') as handle:
        pickle.dump(ar_data_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./data/ar_test_seed{}'.format(seed), 'wb') as handle:
        pickle.dump(ar_data_test, handle, protocol=pickle.HIGHEST_PROTOCOL)


def adult_load(seed):
    np.random.seed(seed)

    #import
    ad_train = pd.read_csv('./data/adult.data',header = None)
    ad_test = pd.read_csv('./data/adult.test', header = None)

    #convert missing data to Nans
    ad_train[ad_train == ' ?']= np.nan
    ad_test[ad_test == ' ?']= np.nan

    #drop missing categorical data
    ad_train.dropna(axis = 0,inplace = True)
    ad_test.dropna(axis = 0,inplace = True)

    #separate covariates from classes
    N_train = np.shape(ad_train)[0]
    y_train = np.zeros(N_train)
    y_train[ad_train.iloc[:,14] == ' >50K']=1
    x_train = ad_train.iloc[:,0:14]

    N_test = np.shape(ad_test)[0]
    y_test = np.zeros(N_test)
    y_test[ad_test.iloc[:,14] == ' >50K.']=1
    x_test = ad_test.iloc[:,0:14]

    #setup dummy
    x_train =pd.get_dummies(x_train,drop_first = True,columns = [1,3,5,6,7,8,9,13])
    x_test = pd.get_dummies(x_test,drop_first = True,columns = [1,3,5,6,7,8,9,13])    

    #fix dummy difference
    missing_cols = set( x_train.columns ) - set( x_test.columns)
    for c in missing_cols:
        x_test[c] = 0
    x_test = x_test[x_train.columns]

    D = np.shape(x_train)[1]
    colnames = list(x_train.columns.values)

    #concatenate and resplit
    x = np.concatenate((x_train,x_test),axis = 0)
    y = np.concatenate((y_train,y_test))
    x_train,x_test, y_train,y_test = train_test_split(x,y,test_size = 0.2, stratify = y)
    x_train = pd.DataFrame(x_train,columns = colnames)
    x_test = pd.DataFrame(x_test, columns = colnames)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    N_train = np.shape(x_train)[0]
    N_test = np.shape(x_test)[0]

    #normalize by mean and std for non dummy variables
    mean_train = x_train.mean(axis = 0)
    std_train = x_train.std(axis = 0)
    x_train[[0,2,4,10,11,12]] -= mean_train
    x_train[[0,2,4,10,11,12]] /= std_train

    mean_test = x_test[[0,2,4,10,11,12]].mean(axis = 0)
    std_test = x_test[[0,2,4,10,11,12]].std(axis = 0)
    x_test[[0,2,4,10,11,12]] -= mean_test
    x_test[[0,2,4,10,11,12]] /= std_test

    #convert binarys to uint8 to save space
    y_train = y_train.astype('uint8')    
    y_test = y_test.astype('uint8')

    colnames2 = set(colnames) - set([0,2,4,10,11,12])
    x_train[list(colnames2)] = x_train[list(colnames2)].astype('uint8') 
    x_test[list(colnames2)] = x_test[list(colnames2)].astype('uint8') 

    #Put into dictionary and save
    ad_data_train= {'y': y_train, 'x': x_train, 'N':N_train,'D':D}
    ad_data_test= {'y': y_test, 'x': x_test, 'N':N_test,'D':D}

    with open('./data/ad_train_seed{}'.format(seed), 'wb') as handle:
        pickle.dump(ad_data_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./data/ad_test_seed{}'.format(seed), 'wb') as handle:
        pickle.dump(ad_data_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

def polish_load(seed):
    np.random.seed(seed)

    #training data percentage
    year =3
    train_perc = 0.8

    #import
    ar,met = arff.loadarff('./data/{}year.arff'.format(year))
    pc = pd.DataFrame(ar)

    N = np.shape(pc)[0]
    D = np.shape(pc)[1]-1

    #resplit
    x = pc.iloc[:,0:D]
    y = pc.iloc[:,D]
    x_train,x_test, y_train,y_test = train_test_split(x,y,test_size = 0.2, stratify = y)
    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    #Calculate training and test N
    N_train = np.shape(x_train)[0]
    N_test = np.shape(x_test)[0]

    #normalize x values by mean and std
    x_train = (x_train - x_train.mean(axis = 0))/x_train.std(axis = 0)
    x_test = (x_test - x_test.mean(axis = 0))/x_test.std(axis = 0)

    #impute means as numbers are real financial after split
    x_train.fillna(0,inplace = True)  
    x_test.fillna(0,inplace= True)

    #convert to dict and save
    pc_data_train= {'y': y_train, 'x': x_train, 'N':N_train,'D':D}
    pc_data_test= {'y': y_test, 'x': x_test, 'N':N_test,'D':D}

    with open('./data/pc_train_y{}_seed{}'.format(year,seed), 'wb') as handle:
        pickle.dump(pc_data_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./data/pc_test_y{}_seed{}'.format(year,seed), 'wb') as handle:
        pickle.dump(pc_data_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

for i in range(30):
    arcene_load(100+i)
    adult_load(100+i)
    polish_load(100+i)
  