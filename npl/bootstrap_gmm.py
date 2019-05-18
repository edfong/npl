"""function for NPL posterior bootstrap sampler for GMM

Parameters
----------
B_postsamples : int 
    Number of posterior samples to generate

alph_conc : float
    Concentration parameter for DP prior

T_trunc: int > 0 
    Number of prior pseudo-samples from DP base measure for truncated sampling

y: array
    Observed datapoints

N_data: int
    Number of data points

D_data: int
    Dimension of observables

K_clusters: int
    Number of clusters in GMM model

R_restarts: int 
    Number of random restarts per posterior bootstrap sample

tol: float
    Stopping criterion for weighted EM

max_iter: int
    Maximum number of iterations for weighted EM

init: function
    Returns initial parameters for random restart maximizations 

sampleprior: function
    To generate prior pseudo-samples for DP prior

postsamples: array
    Centering posterior samples for MDP-NPL
    
n_cores: int
    Number of cores Joblib can parallelize over; set to -1 to use all cores
"""


import numpy as np
import pandas as pd
import time
import copy
from joblib import Parallel, delayed
from tqdm import tqdm
from npl import maximise_gmm as mgmm


def bootstrap_gmm(B_postsamples,alph_conc,T_trunc,y,N_data,D_data,K_clusters,R_restarts,tol,max_iter,init,sampleprior,postsamples= None,n_cores = -1):
    #Declare parameters
    pi_bb = np.zeros((B_postsamples,K_clusters))              #mixing weights 
    mu_bb = np.zeros((B_postsamples,K_clusters,D_data))       #means
    sigma_bb = np.zeros((B_postsamples,K_clusters,D_data))    #covariances 
  

    #Generate prior pseudo-samples and concatenate y_tots
    if alph_conc!=0:
        alphas = np.concatenate((np.ones(N_data), (alph_conc/T_trunc)*np.ones(T_trunc)))
        weights = np.random.dirichlet(alphas,B_postsamples) 
        y_prior = sampleprior(D_data,T_trunc,K_clusters,B_postsamples, postsamples)
    else:
        weights = np.random.dirichlet(np.ones(N_data), B_postsamples)
        y_prior = np.zeros(B_postsamples)

    #Initialize parameters randomly for RR-NPL
    pi_init,mu_init,sigma_init = init(R_restarts, K_clusters,B_postsamples, D_data)


    #Parallelize bootstrap
    if R_restarts == 0: #FI-NPL (with MLE initialization to select single mode)
        pi_init_mle,mu_init_mle,sigma_init_mle = mgmm.init_params(y,N_data,K_clusters,D_data,tol,max_iter)
        temp = Parallel(n_jobs=n_cores, backend= 'loky')(delayed(mgmm.maximise_mle)(y,weights[i],pi_init_mle,\
            mu_init_mle,sigma_init_mle,K_clusters,tol,max_iter,N_data) for i in tqdm(range(B_postsamples))) 
    else:
        temp = Parallel(n_jobs=n_cores, backend= 'loky')(delayed(mgmm.maximise)(y,y_prior[i],weights[i],\
            pi_init[i*R_restarts:(i+1)*R_restarts],mu_init[i*R_restarts:(i+1)*R_restarts],sigma_init[i*R_restarts:(i+1)*R_restarts],\
            alph_conc, T_trunc,K_clusters,tol,max_iter,R_restarts,N_data,D_data, postsamples = postsamples) for i in tqdm(range(B_postsamples)))
    

    for i in range(B_postsamples):
        pi_bb[i] = temp[i][0]
        mu_bb[i] = temp[i][1]
        sigma_bb[i]= temp[i][2]

    return pi_bb,mu_bb,sigma_bb
