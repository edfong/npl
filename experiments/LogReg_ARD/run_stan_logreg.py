""" 
Running NUTS and ADVI on Pystan for Logistic Regression

"""

import pystan
import numpy as np
import pickle
import pandas as pd
import time
from joblib import Parallel, delayed
from tqdm import tqdm

def setup_model():
    #define model
    sm = pystan.StanModel(file = 'logreg.stan')

    #save model so no need for recompile
    with open('model_ARD_pr.pkl', 'wb') as f:
        pickle.dump(sm, f)

def main(dataset,seed):
    #load stan model
    sm = pickle.load(open('model_ARD_pr.pkl','rb'))


    #hyperparams for gamma hyperprior
    a = 1
    b = 1


    #load Polish data
    if dataset =='Polish':
        year = 3
        with open('./data/pc_train_y{}_seed{}'.format(year,seed), 'rb') as handle:
            pc_train = pickle.load(handle)

        y = pd.to_numeric(pc_train['y'].values[:,0])
        x = pc_train['x'].values
        D = pc_train['D']
        N = pc_train['N']

    #load Adult data
    if dataset =='Adult':
        with open('./data/ad_train_seed{}'.format(seed), 'rb') as handle:
            ad_train = pickle.load(handle)

        y = np.int8(ad_train['y'])[:,0]
        x = ad_train['x'].values
        D = ad_train['D']
        N = ad_train['N']

    #load Arcene data
    if dataset =='Arcene':
        with open('./data/ar_train_seed{}'.format(seed), 'rb') as handle:
            ar_train = pickle.load(handle)

        N = ar_train['N']
        D = ar_train['D']
        y = np.int8(ar_train['y'].values.reshape(N,))
        x = ar_train['x'].values

    #put into dict for stan
    smdata = {'N': N,'D': D, 'y':y, 'x': x, 'a' : a, 'b' :b}

    #fit NUTS 
    start = time.time()
    fit_nuts = sm.sampling(data = smdata, warmup = 1000, iter = 3000, chains = 1, n_jobs = -1, seed = seed)
    end = time.time()

    time_nuts = end-start

    #process MCMC data
    par_nuts = pd.DataFrame(fit_nuts.to_dataframe())

    #save parameters for NUTS
    #polish
    if dataset == 'Polish':
        par_nuts.to_pickle('./parameters/par_nuts_logreg_pol_ARD_seed{}'.format(seed))
        np.save('./parameters/time_nuts_pol_seed{}'.format(seed),time_nuts)
    #adult
    if dataset =='Adult':
        par_nuts.to_pickle('./parameters/par_nuts_logreg_ad_ARD_seed{}'.format(seed))
        np.save('./parameters/time_nuts_ad_seed{}'.format(seed),time_nuts)
    #arcene
    if dataset =='Arcene':
        par_nuts.to_pickle('./parameters/par_nuts_logreg_ar_ARD_seed{}'.format(seed))
        np.save('./parameters/time_nuts_ar_seed{}'.format(seed),time_nuts)

    #fit ADVI, separate file for each repeat
    start = time.time()
    fit_advi = sm.vb(data = smdata, algorithm = 'meanfield',output_samples = 2000, seed = seed)
    end = time.time()
    print('ADVI required {}'.format(end-start))

    time_advi = end-start

    #process VB data
    param_names_advi = fit_advi['sampler_param_names']
    params_advi = fit_advi['sampler_params']
    par_advi = pd.DataFrame(dict(zip(param_names_advi,params_advi)))

    #save ADVI data
    #polish
    if dataset == 'Polish':
        par_advi.to_pickle('./parameters/par_advi_logreg_pol_ARD_seed{}'.format(seed))
        np.save('./parameters/time_advi_pol_seed{}'.format(seed),time_advi)

    #adult
    if dataset =='Adult':
        par_advi.to_pickle('./parameters/par_advi_logreg_ad_ARD_seed{}'.format(seed))
        np.save('./parameters/time_advi_ad_seed{}'.format(seed),time_advi)

    # #arcene
    if dataset =='Arcene':
        par_advi.to_pickle('./parameters/par_advi_logreg_ar_ARD_seed{}'.format(seed))
        np.save('./parameters/time_advi_ar_seed{}'.format(seed),time_advi)

if __name__=='__main__':
    setup_model()
    seed = np.arange(100,130,1)
    Parallel(n_jobs=-1, backend= 'loky')(delayed(main)('Polish',seed[i]) for i in tqdm(range(30)))
    Parallel(n_jobs=-1, backend= 'loky')(delayed(main)('Adult',seed[i]) for i in tqdm(range(30)))
    Parallel(n_jobs=-1, backend= 'loky')(delayed(main)('Arcene',seed[i]) for i in tqdm(range(30)))