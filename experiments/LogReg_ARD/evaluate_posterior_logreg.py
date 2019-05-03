"""
Evaluate posterior samples predictive performance/time

"""

import numpy as np
import scipy as sp
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import copy
import importlib
from npl.evaluate import logreg_ll as lrl

def load_data(dataset,seed):
    #Load test data
    if dataset == 'Polish':
        pref = 'pol'
        year = 3
        with open('./data/pc_train_y{}_seed{}'.format(year,seed), 'rb') as handle:
            pc_train = pickle.load(handle)
            
        y = pd.to_numeric(pc_train['y'].values[:,0])
        x = pc_train['x'].values
        D = pc_train['D']
        N = pc_train['N']

        year = 3
        with open('./data/pc_test_y{}_seed{}'.format(year,seed), 'rb') as handle:
            pc_test = pickle.load(handle)
            
        y_test = pd.to_numeric(pc_test['y'].values[:,0])
        x_test = pc_test['x'].values
        N_test = pc_test['N']
        c = 0
        eps = 1e-1

    if dataset == 'Adult':
        pref = 'ad'

        with open('./data/ad_train_seed{}'.format(seed), 'rb') as handle:
            ad_train = pickle.load(handle)

        #Move into vectors
        y =  np.uint8(ad_train['y'])[:,0]
        x = ad_train['x'].values
        D = ad_train['D']
        N = ad_train['N']

        with open('./data/ad_test_seed{}'.format(seed), 'rb') as handle:
            ad_test = pickle.load(handle)

        #Move into vectors
        y_test =  np.uint8(ad_test['y'])[:,0]
        x_test = ad_test['x'].values
        N_test = ad_test['N']
        c=0
        eps = 1e-1

    if dataset == 'Arcene':
        pref = 'ar'

        with open('./data/ar_train_seed{}'.format(seed), 'rb') as handle:
            ar_train = pickle.load(handle)

        N = ar_train['N']
        D = ar_train['D']
        y = np.int8(ar_train['y'].values.reshape(N,))
        x = ar_train['x'].values

        with open('./data/ar_test_seed{}'.format(seed), 'rb') as handle:
            ar_test = pickle.load(handle)

        N_test = ar_test['N']
        D = ar_test['D']
        y_test = np.int8(ar_test['y'].values.reshape(N,))
        x_test = ar_test['x'].values
        c = 1
        eps = 1e-2
    return y_test,x_test,N_test,D,c,pref,eps

def load_posterior(method,pref,seed,c,D):
    if method == 'NPL':
        a= 1
        b= 1
        par = pd.read_pickle('./parameters/par_bb_logreg_c{}_a{}_b{}_gN_{}_B2000_seed{}'.format(c,a,b,pref,seed))
        beta = par['beta'][:,0:D]
        alpha = par['beta'][:,D]
        time = par['time']
    elif method == 'NUTS':
        par = pd.read_pickle('./parameters/par_nuts_logreg_{}_ARD_seed{}'.format(pref,seed))
        beta = par.iloc[:,3:D+3].values
        alpha = par.iloc[:,D+3].values
        time = np.load('./parameters/time_nuts_{}_seed{}.npy'.format(pref,seed))

    elif method == 'ADVI':
        par = pd.read_pickle('./parameters/par_advi_logreg_{}_ARD_seed{}'.format(pref,seed))
        beta = par.iloc[:,0:D].values
        alpha = par.iloc[:,D].values
        time = np.load('./parameters/time_advi_{}_seed{}.npy'.format(pref,seed))

    return beta,alpha,time

def eval(dataset,method):
    #Load test data
    lppd = np.zeros(30)
    mse = np.zeros(30)
    predcor = np.zeros(30)
    time = np.zeros(30)
    card = np.zeros(30)

    for i in range(30):
        #Run through each seed to calculate predictive performance/times
        seed = 100+i
        y_test,x_test,N_test,D,c,pref,eps = load_data(dataset,seed)
        beta , alpha,time[i] = load_posterior(method,pref,seed,c,D)

        lppd[i]=lrl.lppd(y_test,x_test,beta,alpha)
        mse[i]=lrl.MSE(y_test,x_test,beta,alpha)
        predcor[i]= lrl.predcorrect(y_test,x_test,beta,alpha)
        mean_beta = np.mean(beta,axis = 0)
        card[i] = lrl.checkcard([mean_beta],eps)

    dict = {'lppd':lppd/N_test, 'mse':mse, 'predcor': 100*predcor, 'card': ((D-card)/D)*100, 'time': time }

    #Print mean with standard error for each measure/time 
    print('For {}, dataset {}'.format(method,dataset))
    print('lppd')
    print(np.mean(lppd/N_test))
    print(np.std(lppd/N_test))

    print('mse')
    print(np.mean(mse))
    print(np.std(mse))

    print('pa')
    print(np.mean(100*predcor))
    print(np.std(100*predcor))

    print('sparse')
    print(np.mean(((D-card)/D)*100))
    print(np.std(((D-card)/D)*100))
    
    print('time')
    print(np.mean(time))
    print(np.std(time))


#Run for 3 datasets from paper
def main():
    eval('Adult','NPL')
    eval('Adult','NUTS')
    eval('Adult','ADVI')
    eval('Polish','NPL')
    eval('Polish','NUTS')
    eval('Polish','ADVI')
    eval('Arcene','NPL')
    eval('Arcene','NUTS')
    eval('Arcene','ADVI')

if __name__=='__main__':
    main()
