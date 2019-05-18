"""
main script for running NPL

"""

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import copy
import time
from npl import bootstrap_logreg as bbl
import pickle

def load_data(dataset,seed):
        #load polish
    if dataset == 'Polish':
        year = 3
        with open('./data/pc_train_y{}_seed{}'.format(year,seed), 'rb') as handle:
            pc_train = pickle.load(handle)

        #Move into vectors
        y = pd.to_numeric(pc_train['y'].values[:,0])
        x = pc_train['x'].values
        D_data = pc_train['D']
        N_data = pc_train['N']

        #prior and loss settings from paper
        alph_conc = 0           #prior strength
        gamma = 1/N_data        #loss scaling relative to log-likelihood 


    #load adult
    if dataset == 'Adult':
        with open('./data/ad_train_seed{}'.format(seed), 'rb') as handle:
            ad_train = pickle.load(handle)

        #Move into vectors
        y = np.uint8(ad_train['y'])[:,0]
        x = ad_train['x'].values
        D_data = ad_train['D']
        N_data = ad_train['N']
  

        #prior and loss settings from paper
        alph_conc = 0 
        gamma = 1/N_data

    #load arcene
    if dataset == 'Arcene':
        with open('./data/ar_train_seed{}'.format(seed), 'rb') as handle:
            ar_train = pickle.load(handle)

        N_data = ar_train['N']
        D_data = ar_train['D']
        y = np.int8(ar_train['y'].values.reshape(N_data,))
        x = ar_train['x'].values

        #prior and loss settings from paper
        alph_conc = 1
        gamma  = 1/N_data

    return y,x,alph_conc,gamma,N_data,D_data



def main(dataset, B_postsamples):
    #same parameters between datasets
    T_trunc = 100
    a=1
    b = 1 #rate of gamma hyperprior
    for i in range(30):

        seed = 100+i
        np.random.seed(seed)
        y,x,alph_conc,gamma,N_data,D_data = load_data(dataset,seed)

        start= time.time()
        #carry out posterior bootstrap
        beta_bb, ll_b = bbl.bootstrap_logreg(B_postsamples,alph_conc,T_trunc,y,x,N_data,D_data,a,b,gamma)
        end = time.time()
        print ('Time elapsed = {}'.format(end - start))

        #convert to dataframe and save
        dict_bb = {'beta': beta_bb, 'll_b': ll_b, 'time': end-start}
        par_bb = pd.Series(data = dict_bb)

        #Polish
        if dataset == 'Polish':
            par_bb.to_pickle('./parameters/par_bb_logreg_c{}_a{}_b{}_gN_pol_B{}_seed{}'.format(alph_conc,a,b,B_postsamples,seed))

        #Adult
        if dataset == 'Adult':
            par_bb.to_pickle('./parameters/par_bb_logreg_c{}_a{}_b{}_gN_ad_B{}_seed{}'.format(alph_conc,a,b,B_postsamples,seed))

        #Arcene
        if dataset == 'Arcene':
            par_bb.to_pickle('./parameters/par_bb_logreg_c{}_a{}_b{}_gN_ar_B{}_seed{}'.format(alph_conc,a,b,B_postsamples,seed))

if __name__=='__main__':
    main('Polish',2000)
    main('Adult',2000)
    main('Arcene',2000)




