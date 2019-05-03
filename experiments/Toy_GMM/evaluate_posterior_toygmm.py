"""
Evaluate posterior samples predictive performance/time

"""


import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy as sp
import importlib
from npl.evaluate import gmm_ll as gll

def load_data(type,seed):
    #load test data
    gmm_data_test = np.load('./sim_data/gmm_data_test_{}_seed{}.npy'.format(type,seed),allow_pickle = True).item()

    #Extract parameters from data
    N_test = gmm_data_test['N']
    K = gmm_data_test['K']
    y_test = gmm_data_test['y'].reshape(N_test)
    
    return y_test,N_test,K


def load_posterior(method,type,seed,K):
    if method == 'RRNPL':
        par = pd.read_pickle('./parameters/par_bb_{}_rr_rep{}_B{}_seed{}'.format(type,10,2000,seed))
        pi =np.array(par['pi'])
        mu =np.array(par['mu'])
        sigma = np.array(par[['sigma']][0])
        time = par['time']

    elif method =='FINPL':
        par = pd.read_pickle('./parameters/par_bb_{}_fi__B{}_seed{}'.format(type,2000,seed))
        pi =np.array(par['pi'])
        mu =np.array(par['mu'])
        sigma = np.array(par[['sigma']][0])
        time = par['time']

    elif method == 'NUTS':
        D = 1
        par = pd.read_pickle('./parameters/par_nuts_{}_seed{}'.format(type,seed))
        pi =par.iloc[:,3:K+3].values
        mu =par.iloc[:,3+K: 3+(K*(D+1))].values.reshape(2000,D,K).transpose(0,2,1)
        sigma = par.iloc[:,3+K*(D+1) :3+ K*(2*D+1)].values.reshape(2000,D,K).transpose(0,2,1)
        time = np.load('./parameters/time_nuts_{}_seed{}.npy'.format(type,seed),allow_pickle = True)

    elif method == 'ADVI':
        D = 1
        par = pd.read_pickle('./parameters/par_advi_{}_seed{}'.format(type,seed))
        pi =par.iloc[:,0:K].values
        mu =par.iloc[:,K: (K*(D+1))].values.reshape(2000,D,K).transpose(0,2,1)
        sigma = par.iloc[:,K*(D+1) : K*(2*D+1)].values.reshape(2000,D,K).transpose(0,2,1)
        time = np.load('./parameters/time_advi_{}_seed{}.npy'.format(type,seed),allow_pickle = True)


    return pi,mu,sigma,time

def eval(method,type):
    ll_test = np.zeros(30)
    time = np.zeros(30)
    for i in range(30):
        seed = 100+i

        #Extract parameters from data
        y_test,N_test,K = load_data(type,seed)
        pi,mu,sigma,time[i]  = load_posterior(method,type,seed,K)
        ll_test[i] = gll.lppd(y_test.reshape(-1,1),pi,mu, sigma,K)
        

    print('For {}, dataset {}'.format(method,type))
    print(np.mean(ll_test/N_test))
    print(np.std(ll_test/N_test))

    print(np.mean(time))
    print(np.std(time))

def main():
    eval('RRNPL','sep')
    eval('FINPL','sep')
    eval('NUTS','sep')
    eval('ADVI','sep')

if __name__=='__main__':
    main()