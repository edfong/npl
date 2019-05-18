"""function for NPL posterior bootstrap sampler for logistic regression

Parameters
----------
B_postsamples : int 
    Number of posterior samples to generate

alph_conc : float
    Concentration parameter for DP prior

T_trunc: int > 0 
    Number of prior pseudo-samples from DP base measure for truncated sampling

y: array
    Observed classes

x: array
    Covariates

N_data: int
    Number of data points

D_covariate: int
    Dimension of covariates

a,b: floats
    Hyperparameter terms in Student t prior

gamma: float
    Loss scaling term for loss-NPL

n_cores: int
    Number of cores Joblib can parallelize over; set to -1 to use all cores
"""
import numpy as np
import scipy as sp
import pandas as pd
import copy
import time
import npl.maximise_logreg as mlr
from joblib import Parallel, delayed,dump, load
from tqdm import tqdm
import os


def bootstrap_logreg(B_postsamples,alph_conc,T_trunc,y,x,N_data,D_covariate,a,b,gamma,n_cores = -1):   #Bootstrap posterior
    if alph_conc!=0:
        alphas = np.concatenate((np.ones(N_data), (alph_conc/T_trunc)*np.ones(T_trunc)))
        weights = np.random.dirichlet(alphas,B_postsamples)
        y_prior,x_prior = mlr.sampleprior(x,N_data,D_covariate,T_trunc,B_postsamples) 
    else:
        alphas = np.ones(N_data)
        weights = np.random.dirichlet(alphas,B_postsamples)   
        y_prior = np.zeros(B_postsamples)
        x_prior = np.zeros(B_postsamples)

    #Initialize parameters 
    ll_bb = np.zeros(B_postsamples)                     #weighted log loss
    beta_bb = np.zeros((B_postsamples,D_covariate+1))   #regression coefficient + intercept as the last value

    #Initialize RR-NPL with normal(0,1)
    R_restarts = 1
    beta_init = np.random.randn(R_restarts*B_postsamples,D_covariate+1)

    #Carry out weighted loss maximizations in parallel with joblib, pass different set of R_restarts inits for each bootstrap
    temp = Parallel(n_jobs=n_cores, backend= 'loky')(delayed(mlr.maximise)(y,x,y_prior[i],x_prior[i],N_data,D_covariate,alph_conc,T_trunc,weights[i],beta_init[i*R_restarts:(i+1)*R_restarts],a,b,gamma,R_restarts) for i in tqdm(range(B_postsamples)))

    #Convert to numpy array
    for i in range(B_postsamples):
        beta_bb[i],ll_bb[i] = temp[i] 


    return beta_bb, ll_bb        
