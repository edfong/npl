import numpy as np
import scipy as sp
import copy as copy

def main(seed, modes):
    np.random.seed(seed)

    #Define GMM parameters
    K = 3 
    D = 1
    Pi = np.array([0.1,0.3,0.6])
    zvalues = np.array([0,1,2])
  
    if modes == 'sep':
        meanvalues = np.array([0,2,4])
    elif modes == 'insep':
        meanvalues = np.array([0,1,2]) 
    sigvalues = np.array([1,1,1])


    #Sample training z and y
    N = 1000
    z = np.zeros(N,dtype = 'int')
    y = np.zeros(N)
    zrng = np.random.rand(N)
    yrng = np.random.randn(N)

    for i in range(N):
        ind = np.argmax(np.cumsum(Pi) > zrng[i])
        z[i]= zvalues[ind]
        y[i] = yrng[i]*sigvalues[ind] + meanvalues[ind]

    y = y.reshape(N,D)

    #Concatenate and save training
    gmm_data = {'y': y, 'z': z, 'N':N,'K': K,'D':D, 'Pi' : Pi, 'zvals': zvalues, 'meanvals': meanvalues, 'sigvals':sigvalues}
    np.save('./sim_data/gmm_data_{}_seed{}'.format(modes,seed),gmm_data)


    #Sample test z and y
    N_test = 250
    z_test = np.zeros(N_test,dtype = 'int')
    y_test = np.zeros(N_test)
    zrng = np.random.rand(N_test)
    yrng = np.random.randn(N_test)

    for i in range(N_test):
        ind = np.argmax(np.cumsum(Pi) > zrng[i])
        z_test[i]= zvalues[ind]
        y_test[i] = yrng[i]*sigvalues[ind] + meanvalues[ind]

    y_test = y_test.reshape(N_test,D)

    #Concatenate and save test data
    gmm_data_test = {'y': y_test, 'z': z_test, 'N':N_test,'K': K,'D':D, 'Pi' : Pi, 'zvals': zvalues, 'meanvals': meanvalues, 'sigvals':sigvalues}
    np.save('./sim_data/gmm_data_test_{}_seed{}'.format(modes,seed),gmm_data_test)

for i in range(30):
    seed = 100+i
    main(seed,'sep')