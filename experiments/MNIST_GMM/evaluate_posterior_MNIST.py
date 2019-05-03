"""
Evaluate NPL for MNIST, uncomment for FI-NPL

"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy as sp
import load_MNIST as lm
import importlib
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from npl.evaluate import gmm_ll as gll


def load_posterior(method,seed):
    if method =='NPL':
        c = 1
        N = 10000
        par =pd.read_pickle('./parameters/par_bb_MNIST_c{}_{}_B2000_seed{}_VBinit'.format(c,N,seed))
        pi =np.array(par['pi'])
        mu =np.array(par['mu'])
        sigma = np.array(par['sigma'])
        time = par['time']
    elif method == 'ADVI':
        K = 10
        D = 784
        N = 10000
        par = pd.read_pickle('./parameters/par_advi_MNIST_{}_alpha1000_seed{}'.format(N,seed))
        pi =par.iloc[:,0:K].values
        mu =par.iloc[:,K: (K*(D+1))].values.reshape(2000,D,K).transpose(0,2,1)
        sigma = par.iloc[:,K*(D+1) : K*(2*D+1)].values.reshape(2000,D,K).transpose(0,2,1)
        #time = np.load('./parameters/time_advi_MNIST_10000_seed{}.npy'.format(seed))
        time = 0
    return pi,mu,sigma,time

def eval(method):
    #Load Data
    N_test = 2500  #so test is 20%/80% split
    gmm_data_test = lm.MNIST_test_load(N_test)

    #Extract parameters from data
    K = gmm_data_test['K']
    D = gmm_data_test['D']
    z_test = gmm_data_test['z']
    y_test = gmm_data_test['y']

    if method =='NPL':
        ll_test = np.zeros(30)
        time = np.zeros(30)
        for i in range(30):
            seed = 100+i
            pi,mu,sigma,time[i] = load_posterior(method,seed)

            ll_test[i] = gll.lppd(y_test,pi,mu, sigma,K)

    else:
        pi,mu,sigma,time = load_posterior(method,104)
        ll_test = gll.lppd(y_test,pi,mu, sigma,K)

    print(np.mean(ll_test/N_test))
    print(np.std(ll_test/N_test))

    print(np.mean(time))
    print(np.std(time))

def main():
    eval('NPL')
    eval('ADVI')

if __name__=='__main__':
    main()