"""
Contains functions for generating prior pseudo-samples, initializing for RR-NPL and maximizing weighted log likelihood in GMM examples
Uses modified sklearn sk_base and sk_gaussian_mixture classes to carry out weighted EM
"""
import numpy as np
import npl.sk_gaussian_mixture as skgm
from scipy.stats import norm
from scipy.stats import invgamma

def sampleprior_MNIST(y,D_data,T_trunc,K_clusters, postsamples = None):  #generate prior data points
    y_prior = norm.rvs(loc = 0,scale = 0.1, size = (T_trunc,D_data))   #approximately the marginal y from GMM, centre at empirical mean
    return y_prior

def sampleprior_toy(y,D_data,T_trunc,K_clusters, postsamples = None):  #generate prior data points
    y_prior = norm.rvs(loc = 0,scale = 1, size = (T_trunc,D_data))   #approximately the marginal y from GMM, centre at empirical mean
    return y_prior

def sampleprior_toyMDP(y,D_data,T_trunc,K_clusters, postsamples):  #generate prior data points
    par_nuts = postsamples
    pi_nuts =par_nuts.iloc[:,3:K_clusters+3]
    mu_nuts =par_nuts.iloc[:,3+K_clusters: 3+(K_clusters*(D_data+1))]
    sigma_nuts = par_nuts.iloc[:,3+K_clusters*(D_data+1) :3+ K_clusters*(2*D_data+1)]

    B_postsamples = np.shape(pi_nuts)[0]

    ind_postsample = np.random.choice(B_postsamples)

    ind_cluster = np.random.choice(K_clusters, p = pi_nuts.iloc[ind_postsample])

    y_prior = norm.rvs(loc = mu_nuts.iloc[ind_postsample,ind_cluster], scale = sigma_nuts.iloc[ind_postsample,ind_cluster] , size = (T_trunc,D_data))

    return y_prior

def init_toy(R_restarts, K_clusters,B_postsamples, D_data):
    pi_init = np.random.dirichlet(np.ones(K_clusters),R_restarts*B_postsamples)
    mu_init = 8*np.random.rand(R_restarts*B_postsamples,K_clusters,D_data) - 2
    sigma_init = invgamma.rvs(1, size = (R_restarts*B_postsamples,K_clusters,D_data))  #covariances 
    return pi_init,mu_init,sigma_init

def init_MNIST(R_restarts, K_cluster,B_postsamples, D_data):
    import pkg_resources
    pi_init = np.tile([np.load(pkg_resources.resource_filename('npl','init_parameters/pi_init_VB.npy'))],(B_postsamples,1))
    mu_init = np.tile([np.load(pkg_resources.resource_filename('npl','init_parameters/mu_init_VB.npy'))],(B_postsamples,1,1))
    sigma_init = np.tile([np.load(pkg_resources.resource_filename('npl','init_parameters/sigma_init_VB.npy'))],(B_postsamples,1,1))
    return pi_init,mu_init,sigma_init

def init_params(y,N_data,K_clusters,D_data,tol,max_iter): #initialize parameters for FI-NPL by picking MLE
    R_restarts = 10

    pi_bb = np.zeros((R_restarts,K_clusters))       #mixing weights (randomly)
    mu_bb = np.zeros((R_restarts,K_clusters,D_data))     #means
    sigma_bb = np.zeros((R_restarts,K_clusters,D_data))  #covariances 
    ll_bb = np.zeros(R_restarts)

    #Initialize parameters randomly
    pi_init = np.random.dirichlet(np.ones(K_clusters),R_restarts)
    mu_init = 8*np.random.rand(R_restarts,K_clusters,D_data) - 2
    sigma_init = invgamma.rvs(1, size = (R_restarts,K_clusters,D_data)) 

    for i in range(R_restarts):
        model = skgm.GaussianMixture(K_clusters, covariance_type = 'diag',means_init = mu_init[i],weights_init = pi_init[i],precisions_init = 1/sigma_init[i], tol = tol,max_iter = max_iter)
        model.fit(y,np.ones(N_data))
        pi_bb[i] = model.weights_
        mu_bb[i]= model.means_
        sigma_bb[i] = np.sqrt(model.covariances_)
        ll_bb[i] = model.score(y,np.ones(N_data))*N_data
    ind = np.argmax(ll_bb)
    return pi_bb[ind],mu_bb[ind],sigma_bb[ind]

def maximise_mle(y,weights,pi_init,mu_init, sigma_init,K_clusters,tol,max_iter,N_data):  #maximization for FI-NPL
    model = skgm.GaussianMixture(K_clusters, covariance_type = 'diag',means_init = mu_init,weights_init = pi_init,precisions_init = 1/sigma_init, tol = tol,max_iter = max_iter)
    model.fit(y,N_data*weights)
    pi_bb = model.weights_
    mu_bb= model.means_
    sigma_bb = np.sqrt(model.covariances_)
    return pi_bb,mu_bb,sigma_bb


def maximise(y,weights,pi_init,mu_init,sigma_init,alph_conc, T_trunc,K_clusters,tol,max_iter,R_restarts,N_data,D_data,sampleprior, postsamples = None): #maximization when c = 0 for RR-NPL
    if alph_conc !=0:
        y_prior = sampleprior(y,D_data,T_trunc,K_clusters,postsamples)
        y_tot = np.concatenate((y,y_prior))
    else:
        y_tot = y
    pi_bb = np.zeros((R_restarts,K_clusters))      #mixing weights (randomly)
    mu_bb = np.zeros((R_restarts,K_clusters,D_data))         #means
    sigma_bb = np.zeros((R_restarts,K_clusters,D_data))   #covariances 
    ll_bb = np.zeros(R_restarts)
    n_tot = np.shape(y_tot)[0]
    for i in range(R_restarts):
        model = skgm.GaussianMixture(K_clusters, covariance_type = 'diag',means_init = mu_init[i],weights_init = pi_init[i],precisions_init = 1/sigma_init[i], tol = tol,max_iter = max_iter)
        model.fit(y_tot,n_tot*weights)
        pi_bb[i] = model.weights_
        mu_bb[i]= model.means_
        sigma_bb[i] = np.sqrt(model.covariances_)
        ll_bb[i] = model.score(y_tot,weights)*n_tot
    ind = np.argmax(ll_bb)
    return pi_bb[ind],mu_bb[ind],sigma_bb[ind]
