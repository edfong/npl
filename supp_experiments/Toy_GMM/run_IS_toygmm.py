""" 
Running Importance Sampling for Toy GMM

"""

import numpy as np
import pandas as pd
import time
import copy
from joblib import Parallel, delayed
from tqdm import tqdm
import npl.sk_gaussian_mixture as skgm
from scipy.stats import dirichlet
from scipy.stats import norm
from scipy.stats import lognorm
from sklearn.mixture import GaussianMixture
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky


def sampleproposal(B,D,K):
    #sample pis from dirichlet(1,1...1,1)
    pi_alpha = 0.1
    pi_pr = np.random.dirichlet(pi_alpha* np.ones(K),size = B)
    logpidensity =dirichlet.logpdf(np.transpose(pi_pr),pi_alpha*np.ones(K))

    #sample mus from normal(0,25)
    mu_std = 5
    mu_pr = mu_std*np.random.randn(B,K,D)
    logmudensity = np.sum(norm.logpdf(mu_pr, scale = mu_std),axis = (1,2))

    #sample sigmas from lognormal(0,1)
    sigma_pr = np.exp(np.random.randn(B,K,D)) 
    logsigmadensity = np.sum(lognorm.logpdf(sigma_pr, s = 1),axis = (1,2))
 
    logpropdensity = logpidensity + logmudensity + logsigmadensity

    logpipriordensity = dirichlet.logpdf(np.transpose(pi_pr),np.ones(K))
    logpriordensity = logpipriordensity  + np.sum(norm.logpdf(mu_pr, scale = 1),axis = (1,2)) +logsigmadensity

    return pi_pr,mu_pr,sigma_pr, logpriordensity, logpropdensity


def gmm_loglik(y,pi,mu,sigma,K):
    model = GaussianMixture(K, covariance_type = 'diag')
    model.fit(y)
    N = np.shape(mu)[0]
    N_test = np.shape(y)[0]
    ll_test = np.zeros(N)
    for i in (range(N)):
        model.means_ = mu[i,:]
        model.covariances_ = sigma[i,:]**2
        model.precisions_ = 1/(sigma[i,:]**2)
        model.weights_ = pi[i,:]
        model.precisions_cholesky_ = _compute_precision_cholesky(model.covariances_, model.covariance_type)
        ll_test[i] = model.score(y) 
    return ll_test*N_test


def sample(B,y,N,D,K):
    pi_pr,mu_pr,sigma_pr,logpriordensity, logpropdensity = sampleproposal(B,D,K)
    loglik = Parallel(n_jobs=-1, backend= 'loky')(delayed(gmm_loglik)(y,pi_pr[i*100:(i+1)*100],mu_pr[i*100:(i+1)*100],sigma_pr[i*100:(i+1)*100],K) for i in tqdm(range(np.int(B/100))))
    loglik = np.array(loglik).flatten()
    logweights_is = loglik + logpriordensity - logpropdensity
    
    return pi_pr,mu_pr,sigma_pr,logweights_is


def main(B,r): #B is number of IS samples
    gmm_data = np.load('./sim_data_plot/gmm_data_sep.npy',allow_pickle = True).item()

    #Extract parameters from data
    N = gmm_data['N']
    K = gmm_data['K']
    D = gmm_data['D']
    zvalues = gmm_data['zvals']
    y = gmm_data['y']


    start = time.time()
    pi_is,mu_is,sigma_is,logweights_is = sample(B,y,N,D,K)
    end = time.time()

    print(end-start)

    #save file
    dict_is = {'pi': pi_is.tolist(),'sigma': sigma_is.tolist(), 'mu': mu_is.tolist(), 'logweights_is' : logweights_is.tolist(),'time': end-start}

    par_is = pd.Series(data = dict_is)

   
    par_is.to_pickle('./parameters/par_IS_B{}_r{}'.format(B,r))


if __name__=='__main__':
    np.random.seed(100)
    for i in tqdm(range(10)):
        main(10000000,i)


