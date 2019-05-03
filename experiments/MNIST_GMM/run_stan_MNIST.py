"""
Run ADVI for MNIST

"""

import pystan
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import load_MNIST as lm
from joblib import Parallel, delayed
from tqdm import tqdm

def setup_model():
    #define model
    sm = pystan.StanModel(file = 'gmm.stan')

    #save model so no need for recompile
    with open('model.pkl', 'wb') as f:
        pickle.dump(sm, f)

def main(seed):
    #load stan model
    sm = pickle.load(open('model.pkl','rb'))

    #load data
    N=10000
    gmm_data = lm.MNIST_load(N)

    smdata = {'N': gmm_data['N'],'K': gmm_data['K'],'D': gmm_data['D'], 'y': gmm_data['y'],
    'alpha0' : 1000, 'mu_sigma0': 1, 'sigma_sigma0': 1}

    #fit nuts ADVI and time
    start = time.time()
    fit_advi = sm.vb(data = smdata,algorithm = 'meanfield',output_samples = 2000, seed = seed)
    end = time.time()
    print('ADVI required {}'.format(end-start))

    #process ADVI data
    param_names_advi = fit_advi['sampler_param_names']
    params_advi = fit_advi['sampler_params']
    par_advi = pd.DataFrame(dict(zip(param_names_advi,params_advi)))


    par_advi.to_pickle('./parameters/par_advi_MNIST_{}_alpha1000_seed{}'.format(N,seed))
    np.save('./parameters/time_advi_MNIST_{}_seed{}'.format(N,seed), end - start)

if __name__=='__main__':
    setup_model()
    main(104)
