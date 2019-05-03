"""
Loading MNIST data

"""

from mnist import MNIST
import numpy as np

def MNIST_load(N):
    mndata = MNIST('samples')
    y,z = mndata.load_training()

    z = np.array(z[0:N])
    y = np.array(y[0:N])/256
    D = np.shape(y)[1]
    zvalues = np.arange(0,10)
    K = 10
    gmm_data= {'y': y, 'z': z, 'N':N,'K': K,'D':D, 'zvals': zvalues}

    return gmm_data

def MNIST_test_load(N):
    mndata = MNIST('samples')
    y,z = mndata.load_testing()

    z = np.array(z[0:N])
    y = np.array(y[0:N])/256
    D = np.shape(y)[1]
    zvalues = np.arange(0,10)
    K = 10
    gmm_data= {'y': y, 'z': z, 'N':N,'K': K,'D':D, 'zvals': zvalues}

    return gmm_data