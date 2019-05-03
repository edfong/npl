""" 
Running NUTS and ADVI on Pystan for Toy GMM problem

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
    sm = pystan.StanModel(file = 'gmm.stan')

    #save model so no need for recompile
    with open('model.pkl', 'wb') as f:
        pickle.dump(sm, f)

setup_model()
#load Stan model
sm = pickle.load(open('model.pkl','rb'))



#Loop over seeds
def main(seed):
    #load Toy GMM data, put in hyperparameters
    gmm_data = np.load('./sim_data/gmm_data_insep_seed{}.npy'.format(seed),allow_pickle = True).item()
    smdata = {'N': gmm_data['N'],'K': gmm_data['K'],'D': gmm_data['D'], 'y': gmm_data['y'],'alpha0' : 1, 'mu_sigma0': 1, 'sigma_sigma0': 1}
    #fit nuts 
    start_nuts = time.time()
    fit_nuts = sm.sampling(data = smdata, warmup = 500, iter = 2500, chains = 1, n_jobs = -1,seed = seed)
    end_nuts = time.time()

    #process NUTS data and save
    par_nuts = pd.DataFrame(fit_nuts.to_dataframe())
    par_nuts.to_pickle('./parameters/par_nuts_insep_seed{}'.format(seed))
    np.save('./parameters/time_nuts_insep_seed{}'.format(seed),end_nuts-start_nuts)

    #fit ADVI
    start_advi = time.time()
    fit_advi = sm.vb(data = smdata, algorithm = 'meanfield',output_samples = 2000,seed = seed)
    end_advi = time.time()

    #process ADVI data and save
    param_names_advi = fit_advi['sampler_param_names']
    params_advi = fit_advi['sampler_params']
    par_advi = pd.DataFrame(dict(zip(param_names_advi,params_advi)))
    par_advi.to_pickle('./parameters/par_advi_insep_seed{}'.format(seed))
    np.save('./parameters/time_advi_insep_seed{}'.format(seed),end_advi-start_advi)

if __name__=='__main__':
    setup_model()
    seed = np.arange(100,130,1)
    Parallel(n_jobs=-1, backend= 'loky')(delayed(main)(seed[i]) for i in tqdm(range(30)))

