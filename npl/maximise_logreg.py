"""
Contains functions for generating prior pseudo-samples and maximizing weighted log likelihood in logistic regression examples
Uses scipy optimize for gradient descent
"""

import numpy as np
import copy
import scipy as sp
from scipy.stats import bernoulli


def sampleprior(x,N_data,D_covariate,T_trunc,B_postsamples): #sample prior pseudo-samples
    ind_x = np.random.randint(low = 0, high = N_data, size = (B_postsamples,T_trunc))  #sample x indices with replacement
    x_prior = x[ind_x]
    y_prior = bernoulli.rvs(0.5, size = (B_postsamples,T_trunc))
    return y_prior, x_prior

def func(beta,weights,y,x,a,b,gamma): #calculate weighted loss
    N_data = np.shape(y)[0]
    D_covariate = np.shape(x)[1]
    z = np.dot(x,beta[0:D_covariate])+ beta[D_covariate]
    logeta = -np.logaddexp(0,-z)   
    lognegeta = -np.logaddexp(0,z)
    loglik_i = y*logeta + (1-y)*lognegeta
    loglik = np.sum(loglik_i * weights)
    k = -gamma * ((2*a + 1)/2)*np.sum(np.log(1+ (1/(2*b))*beta[0:D_covariate]**2))
    return -(loglik +k)

def grad(beta,weights,y,x,a,b,gamma): #calculate weighted loss gradient
    N_data = np.shape(y)[0]
    D_covariate = np.shape(x)[1]
    z = np.dot(x,beta[0:D_covariate])+ beta[D_covariate]
    err = ((y - sp.special.expit(z))*weights)
    gradient = np.zeros(D_covariate+1)
    gradient[0:D_covariate] = np.dot(err,x) - gamma * ((2*a + 1)/2)*(beta[0:D_covariate])/(b + 0.5* beta[0:D_covariate]**2)
    gradient[D_covariate] = np.sum(err,axis = 0)   #intercept gradient (no prior weighting)
    return -gradient


def maximise(y,x,y_prior,x_prior,N_data,D_covariate,alph_conc,T_trunc,weights,beta_init,a,b,gamma,R_restarts): #maximize weighted loss
    if alph_conc!= 0:  #concatenate prior pseudo-samples
        y_tot = np.concatenate((y,y_prior))
        x_tot = np.concatenate((x,x_prior))
    else:
        y_tot = y
        x_tot = x

    beta_bb = np.zeros((R_restarts,D_covariate+1))
    ll = np.zeros(R_restarts)
    for r in range(R_restarts):
        #optimize using scipy's L-BFGS-B
        optimizeresult = sp.optimize.minimize(func,beta_init[r],(weights,y_tot,x_tot,a,b,gamma), method = 'L-BFGS-B', jac = grad, tol = None)
        beta_bb[r] = optimizeresult.x
        ll[r] = optimizeresult.fun

    #select value with smallest negative log-likelihood (i.e. largest log likelihood)
    ind = np.argmin(ll)
    
    return beta_bb[ind], ll[ind]