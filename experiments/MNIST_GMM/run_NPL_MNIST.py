import load_MNIST as lm
import numpy as np
import pandas as pd
import time
from npl import bootstrap_gmm as bgmm
from npl.maximise_gmm import init_MNIST
from npl.maximise_gmm import sampleprior_MNIST

def main(B_postsamples,alph_conc):  #alph_conc is the prior concentration, B_postsamples is number of bootstrap samples
    N_data = 10000
    gmm_data = lm.MNIST_load(N_data)
    y = gmm_data['y']
    z = gmm_data['z']
    D_data = gmm_data['D']
    K_clusters = gmm_data['K']

        
    for i in range(30):
        seed = 100+i
        np.random.seed(seed)

        R_restarts = 1

        #Initialize EM parameters
        tol = 1e-2
        max_iter = 100

        #Initialize DP truncation term
        T_trunc = 500

        start = time.time()
        pi_bb,mu_bb,sigma_bb = bgmm.bootstrap_gmm(B_postsamples,alph_conc,T_trunc,y,N_data,D_data,K_clusters,R_restarts,tol,max_iter,init_MNIST,sampleprior_MNIST)
        end = time.time()
        print(end-start)


        #save file
        dict_bb = {'pi': pi_bb.tolist(),'sigma': sigma_bb.tolist(), 'mu': mu_bb.tolist(),'time':end-start}
        par_bb = pd.Series(data = dict_bb)
        par_bb.to_pickle('./parameters/par_bb_MNIST_c{}_{}_B{}_seed{}_VBinit'.format(alph_conc,N_data,B_postsamples,seed))

if __name__=='__main__':
    main(2000,1)

