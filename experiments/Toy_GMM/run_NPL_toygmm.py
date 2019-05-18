""" 
Running RR-NPL for Toy GMM  (set R_restarts = 0 for FI-NPL)

"""

import numpy as np
import pandas as pd
import time
import copy
from npl import bootstrap_gmm as bgmm
from npl.maximise_gmm import init_toy
from npl.maximise_gmm import sampleprior_toy

def load_data(seed):
    #load data and parameters
    gmm_data = np.load('./sim_data/gmm_data_sep_seed{}.npy'.format(seed),allow_pickle = True).item()

    #Extract parameters from data
    N_data = gmm_data['N']
    K_clusters = gmm_data['K']
    D_data = gmm_data['D']
    y = gmm_data['y']
    
    return y,N_data,K_clusters,D_data

def main(B_postsamples,R_restarts): #B_postsamples is number of bootstrap samples, R_restarts is number of repeats in RR-NPL (set to 0 for FI-NPL)
    for n in range(30):
        seed = 100+n

        np.random.seed(seed)
        y,N_data,K_clusters,D_data = load_data(seed)
        #prior settings
        alph_conc=0         #alph_concentration
        T_trunc = 500       #DP truncation
        tol = 1e-7
        max_iter = 6000
  

        start = time.time()
        pi_bb,mu_bb,sigma_bb= bgmm.bootstrap_gmm(B_postsamples,alph_conc,T_trunc,y,N_data,D_data,K_clusters,R_restarts,tol,max_iter,init_toy,None)
        end = time.time()

        print(end-start)

        #save file
        dict_bb = {'pi': pi_bb.tolist(),'sigma': sigma_bb.tolist(), 'mu': mu_bb.tolist(),'time': end-start}

        par_bb = pd.Series(data = dict_bb)

        if R_restarts ==0: 
            par_bb.to_pickle('./parameters/par_bb_sep_fi__B{}_seed{}'.format(B_postsamples,seed)) #uncomment for FI-NPL
        else:
            par_bb.to_pickle('./parameters/par_bb_sep_rr_rep{}_B{}_seed{}'.format(R_restarts,B_postsamples,seed))

def main_plot(B_postsamples,R_restarts):
    seed = 100 
    np.random.seed(seed)

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

    start = time.time()
    pi_bb,mu_bb,sigma_bb= bgmm.bootstrap_gmm(B_postsamples,alph_conc,T_trunc,y,N_data,D_data,K_clusters,R_restarts,tol,max_iter,init_toy,None)
    end = time.time()

    print(end-start)

    #save file
    dict_bb = {'pi': pi_bb.tolist(),'sigma': sigma_bb.tolist(), 'mu': mu_bb.tolist(),'time': end-start}

    par_bb = pd.Series(data = dict_bb)

    if R_restarts ==0: 
        par_bb.to_pickle('./parameters/par_bb_sep_fi_B{}_plot'.format(B_postsamples))
    else:
        par_bb.to_pickle('./parameters/par_bb_sep_rr_rep{}_B{}_plot_tol'.format(R_restarts,B_postsamples))

if __name__=='__main__': 
    #RR-NPL and FI-NPL experiments
    main(2000,10)
    main(2000,0)

    #Posterior samples for plots
    main_plot(2000,0)
    main_plot(2000,1)
    main_plot(2000,2)
    main_plot(2000,5)
    main_plot(2000,10)
