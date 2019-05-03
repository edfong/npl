"""
Evaluate NPL posterior samples predictive performance/time

"""


import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import scipy as sp
import importlib
import npl.sk_gaussian_mixture as skgm
from npl.evaluate import gmm_ll as gll
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky
from joblib import Parallel, delayed


def IS_lppd(y,pi,mu,sigma,K,logweights_is):  #calculate posterior predictive of test
    model = skgm.GaussianMixture(K, covariance_type = 'diag')
    B = np.shape(mu)[0]
    N_test = np.shape(y)[0]
    ll_test = np.zeros(N_test)
    model.fit(y,np.ones(N_test))

    for i in tqdm(range(B)):
        model.means_ = mu[i,:,:]
        model.covariances_ = sigma[i,:]**2
        model.precisions_ = 1/(sigma[i,:]**2)
        model.weights_ = pi[i,:]
        model.precisions_cholesky_ = _compute_precision_cholesky(model.covariances_, model.covariance_type)
        if i ==0:
            ll_test =  model.score_lppd(y) + logweights_is[i]
        else:
            ll_test = np.logaddexp(ll_test, model.score_lppd(y) + logweights_is[i]) 
    lppd_test = np.sum(ll_test)
    return lppd_test

def eval_IS(y_test,N_test,K,i):
    par_is = pd.read_pickle('./parameters/par_IS_B{}_r{}'.format(10000000,i))
    pi_is = np.array(par_is['pi'])
    mu_is = np.array(par_is['mu'])
    sigma_is = np.array(par_is['sigma'])
    logweights_is = par_is['logweights_is']
    #Normalize weights
    logweights_is -= sp.special.logsumexp(logweights_is)
    lppd = IS_lppd(y_test,pi_is,mu_is,sigma_is,K,logweights_is)

    ESS = 1/np.exp(sp.special.logsumexp(2*logweights_is))

    time = par_is['time']
    return np.sum(lppd),ESS,time

def load_data(method,seed):
    if method == 'RRNPL_IS' or method =='IS':
        gmm_data_test = np.load('./sim_data_plot/gmm_data_test_sep.npy',allow_pickle=True).item()
        N_test = gmm_data_test['N']
        K = gmm_data_test['K']
        y_test = gmm_data_test['y'].reshape(N_test)

    else:
        #load test data
        gmm_data_test = np.load('./sim_data/gmm_data_test_insep_seed{}.npy'.format(seed),allow_pickle = True).item()

        #Extract parameters from data
        N_test = gmm_data_test['N']
        K = gmm_data_test['K']
        y_test = gmm_data_test['y'].reshape(N_test)
    
    return y_test,N_test,K


def load_posterior(method,type,seed,K):
    logweights_is = 0
    if method == 'RRNPL_IS':
        par = pd.read_pickle('./parameters/par_bb_{}_random_repeat_parallel_rep{}_B{}_plot{}'.format(type,10,2000,seed-100))
        pi =np.array(par['pi'])
        mu =np.array(par['mu'])
        sigma = np.array(par[['sigma']][0])
        time = par['time']

    elif method == 'DPNPL':
        par = pd.read_pickle('./parameters/par_bb_{}_random_repeat_parallel_alpha10_rep{}_B{}_seed{}'.format(type,10,2000,seed))
        pi =np.array(par['pi'])
        mu =np.array(par['mu'])
        sigma = np.array(par[['sigma']][0])
        time = par['time']

    if method == 'MDPNPL':
        par = pd.read_pickle('./parameters/par_bb_{}_random_repeat_parallel_alpha1000_rep{}_B{}_seed{}_MDP'.format(type,10,2000,seed))
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
        
    return pi,mu,sigma,time, logweights_is 



def eval(method):
    if method == 'RRNPL_IS' or method =='IS':
        rep = 10
        type = 'sep'
    else:
        rep = 30
        type = 'insep'
    ll_test = np.zeros(rep)
    time = np.zeros(rep)

    if method != 'IS':
        for i in range(rep):
            seed = 100+i
            #Extract parameters from data
            y_test,N_test,K = load_data(method,seed)
            pi,mu,sigma,time[i],logweights  = load_posterior(method,type,seed,K)
            ll_test[i] = gll.lppd(y_test.reshape(-1,1),pi,mu, sigma,K)
    else:
        y_test,N_test,K = load_data(method,0)
        temp = Parallel(n_jobs=-1, backend= 'loky')(delayed(eval_IS)(y_test.reshape(-1,1),N_test,K,i) for i in tqdm(range(rep)))
        ESS  = np.zeros(rep)
        for i in range(rep):
            ll_test[i] = temp[i][0]
            ESS[i]= temp[i][1]
            time[i] = temp[i][2]
        print(np.mean(ESS))
        print(np.std(ESS))

    print('For {}, dataset {}'.format(method,type))
    print(np.mean(ll_test/N_test))
    print(np.std(ll_test/N_test))

    print(np.mean(time))
    print(np.std(time))

if __name__=='__main__':
    # eval('RRNPL_IS')
    eval('IS')
    # eval('DPNPL')
    # eval('MDPNPL')
    # eval('NUTS')
    # eval('ADVI')


