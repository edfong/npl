""" 
Running RR-NPL for Toy GMM  (set R_restarts = 0 for FI-NPL)

"""

import numpy as np
import npl.sk_gaussian_mixture as skgm
import pandas as pd
import time
import copy
from npl import bootstrap_gmm as bgmm
from npl.maximise_gmm import init_toy
from npl.maximise_gmm import sampleprior_toy
from npl.maximise_gmm import sampleprior_toyMDP


def load_data(seed):
    #load data and parameters
    gmm_data = np.load('./sim_data/gmm_data_insep_seed{}.npy'.format(seed),allow_pickle = True).item()

    #Extract parameters from data
    N_data = gmm_data['N']
    K_clusters = gmm_data['K']
    D_data = gmm_data['D']
    y = gmm_data['y']
    
    return y,N_data,K_clusters,D_data


def main_IS(B_postsamples,R_restarts): #B_postsamples is number of bootstrap samples, R_restarts is number of repeats in RR-NPL (set to 0 for FI-NPL)
    np.random.seed(100)
    gmm_data = np.load('./sim_data_plot/gmm_data_sep.npy',allow_pickle = True).item()

    #Extract parameters from data
    N_data = gmm_data['N']
    K_clusters = gmm_data['K']
    D_data = gmm_data['D']
    y = gmm_data['y']

    #prior settings
    alph_conc=0         #alph_concentration
    T_trunc = 500     #DP truncation
    tol = 1e-7
    max_iter = 6000

    rep = 10
    for r in range(rep):
        start = time.time()
        pi_bb,mu_bb,sigma_bb= bgmm.bootstrap_gmm(B_postsamples,alph_conc,T_trunc,y,N_data,D_data,K_clusters,R_restarts,tol,max_iter,init_toy,None)
        end = time.time()

        print(end-start)

        #save file
        dict_bb = {'pi': pi_bb.tolist(),'sigma': sigma_bb.tolist(), 'mu': mu_bb.tolist(),'time': end-start}

        par_bb = pd.Series(data = dict_bb)

        par_bb.to_pickle('./parameters/par_bb_sep_random_repeat_parallel_rep{}_B{}_plot{}'.format(R_restarts,B_postsamples,r))

def main_DP(B_postsamples,R_restarts): #B_postsamples is number of bootstrap samples, R_restarts is number of repeats in RR-NPL (set to 0 for FI-NPL)
    for n in range(30):
        seed = 100+n

        np.random.seed(seed)
        y,N_data,K_clusters,D_data = load_data(seed)
        #prior settings
        alph_conc=10         #alph_concentration
        T_trunc = 500     #DP truncation
        tol = 1e-7
        max_iter = 6000
  

        start = time.time()
        pi_bb,mu_bb,sigma_bb= bgmm.bootstrap_gmm(B_postsamples,alph_conc,T_trunc,y,N_data,D_data,K_clusters,R_restarts,tol,max_iter,init_toy,sampleprior_toy)
        end = time.time()

        print(end-start)

        #save file
        dict_bb = {'pi': pi_bb.tolist(),'sigma': sigma_bb.tolist(), 'mu': mu_bb.tolist(),'time': end-start}

        par_bb = pd.Series(data = dict_bb)

        if R_restarts ==0: 
            par_bb.to_pickle('./parameters/par_bb_insep_parallel_mle_rep_B{}_seed{}'.format(B_postsamples,seed)) #uncomment for FI-NPL
        else:
            par_bb.to_pickle('./parameters/par_bb_insep_random_repeat_parallel_alpha{}_rep{}_B{}_seed{}'.format(alph_conc,R_restarts,B_postsamples,seed))


def main_MDP(B_postsamples,R_restarts): #B_postsamples is number of bootstrap samples, R_restarts is number of repeats in RR-NPL (set to 0 for FI-NPL)
    for n in range(30):
        seed = 100+n

        alph_conc = 1000
        np.random.seed(seed)
        y,N_data,K_clusters,D_data = load_data(seed)

        T_trunc = 500     #DP truncation
        tol = 1e-7
        max_iter = 6000
  
        par_nuts = pd.read_pickle('./parameters/par_nuts_insep_seed{}'.format(seed))

        start = time.time()
        pi_bb,mu_bb,sigma_bb= bgmm.bootstrap_gmm(B_postsamples,alph_conc,T_trunc,y,N_data,D_data,K_clusters,R_restarts,tol,\
            max_iter,init_toy,sampleprior_toyMDP,postsamples = par_nuts)
        end = time.time()

        print(end-start)

        #save file
        dict_bb = {'pi': pi_bb.tolist(),'sigma': sigma_bb.tolist(), 'mu': mu_bb.tolist(),'time': end-start}

        par_bb = pd.Series(data = dict_bb)

        
        par_bb.to_pickle('./parameters/par_bb_insep_random_repeat_parallel_alpha{}_rep{}_B{}_seed{}_MDP'.format(alph_conc, R_restarts,B_postsamples,seed))

if __name__ == '__main__':
    main_IS(2000,10)
    main_DP(2000,10)
    main_MDP(2000,10)

