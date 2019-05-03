"""
Load Genetics dataset and preprocess

"""

import numpy as np
import pandas as pd
import scipy as sp
from scipy import special
import random
import pickle

x_train = pd.DataFrame(np.random.randn(500,50))

#take first 500 points
x_train = x_train[0:500]
#take first 50 covariates
x_train = x_train.iloc[:,0:50]


N_train = np.shape(x_train)[0]
D = np.shape(x_train)[1]

beta = np.zeros(D)
#Follow values of beta from Lee's paper
ind = np.array([10,14,24,31,37])-1
val = np.array([-0.2538,0.4578,-0.1873,-0.1498,0.0996])
beta[ind]= val

#Generate ys from bernoulli
eta = special.expit(np.dot(x_train,beta))
y_rng = np.random.rand(N_train)
y = np.zeros(N_train)

y[eta > y_rng]= 1

gen_data = {'y': y, 'x': x_train, 'N':N_train,'D':D, 'beta': beta}
with open('./data/gen_data', 'wb') as handle:
    pickle.dump(gen_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
