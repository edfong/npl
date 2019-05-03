import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy as sp
import sk_gaussian_mixture as skgm
from tqdm import tqdm
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky
from joblib import Parallel, delayed

#Evaluate on test data
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

B= 10000000

gmm_data = np.load('./sim_data_plot/gmm_data_test_sep.npy',allow_pickle = True).item()

#Extract parameters from data
N_test = gmm_data['N']
K = gmm_data['K']
D = gmm_data['D']
zvalues = gmm_data['zvals']
y_test = gmm_data['y']

rep = 10
lppd = np.zeros(rep)
ESS = np.zeros(rep)
def eval(i):
    par_is = pd.read_pickle('./parameters/par_IS_B{}_r{}'.format(B,i))
    pi_is = np.array(par_is['pi'])
    mu_is = np.array(par_is['mu'])
    sigma_is = np.array(par_is['sigma'])
    logweights_is = par_is['logweights_is']
    #Normalize weights
    logweights_is -= sp.special.logsumexp(logweights_is)
    lppd = IS_lppd(y_test,pi_is,mu_is,sigma_is,K,logweights_is)/N_test

    ESS = 1/np.exp(sp.special.logsumexp(2*logweights_is))

    return np.sum(lppd),ESS

temp = Parallel(n_jobs=-1, backend= 'loky')(delayed(eval)(i) for i in tqdm(range(rep)))

for i in range(rep):
    lppd[i] = temp[i][0]
    ESS[i]= temp[i][1]

# for i in range(rep):
#     lppd[i],ESS[i] = eval(i)

print(np.mean(lppd))
print(np.std(lppd))
print(np.mean(ESS))
print(np.std(ESS))
