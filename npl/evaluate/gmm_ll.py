"""
Script for calculating GMM predictive

"""

import numpy as np
from scipy.stats import norm
import copy
import scipy as sp
from sklearn.mixture import GaussianMixture
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky
import npl.sk_gaussian_mixture as skgm


def lppd(y,pi,mu,sigma,K):  #calculate posterior predictive of test
    model = skgm.GaussianMixture(K, covariance_type = 'diag')
    B = np.shape(mu)[0]
    N_test = np.shape(y)[0]
    ll_test = np.zeros((B,N_test))
    model.fit(y,np.ones(N_test))
    for i in range(B):
        model.means_ = mu[i,:]
        model.covariances_ = sigma[i,:]**2
        model.precisions_ = 1/(sigma[i,:]**2)
        model.weights_ = pi[i,:]
        model.precisions_cholesky_ = _compute_precision_cholesky(model.covariances_, model.covariance_type)
        ll_test[i] = model.score_lppd(y) 
    lppd_test = np.sum(sp.special.logsumexp(ll_test,axis = 0)- np.log(B))
    return lppd_test
