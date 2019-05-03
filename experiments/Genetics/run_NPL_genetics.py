"""
main script for running NPL

"""

import numpy as np
import scipy as sp
import pandas as pd
import copy
import time
from npl import bootstrap_logreg as bbl
import pickle
from tqdm import tqdm


np.random.seed(25)

#load genetics dataset
with open('./data/gen_data', 'rb') as handle:
    gen_train = pickle.load(handle)

N_data = gen_train['N']
D_data = gen_train['D']
y = np.int8(gen_train['y'].reshape(N_data,))
x = gen_train['x'].values
gamma = 1/N_data

#Set parameters
B_postsamples =4000		#number of bootstrap samples
alph_conc = 0			#prior strength
T_trunc = 500				#DP truncation

P_steps = 450		#number of different b values

#Initialize
beta_samps = np.zeros((P_steps,B_postsamples,D_data+1))   #regression coefficient + intercept as the last value
a = np.zeros(P_steps)
b = np.zeros(P_steps)

start= time.time()

for p in tqdm(range(P_steps)):
    #ARD parameters
    a[p]=1
    b[p]=0.98**p

    #carry out posterior bootstrap for setting of a,b
    beta_samps[p], ll_b = bbl.bootstrap_logreg(B_postsamples,alph_conc,T_trunc,y,x,N_data,D_data,a[p],b[p],gamma)

end = time.time()
print ('Time elapsed = {}'.format(end - start))

#convert to dataframe and save
dict_bb = {'beta': beta_samps, 'a': a,'b':b}
par_bb = pd.Series(data = dict_bb)

#save file
par_bb.to_pickle('./parameters/par_bb_logreg_gen_T{}_a{}_ARD_B{}_small'.format(P_steps,a[0],B_postsamples))


