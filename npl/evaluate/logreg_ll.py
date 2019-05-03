"""
File used to evaluate predictive performance on test data of posterior samples

"""

import numpy as np
import scipy as sp

#For all:
#beta = posterior coefficient samples with shape (B,D)
#alpha = intercept coefficient samples with shape (D)
#y = test data classification with shape (N) 
#x = test data covariates with shape (N,D)

#evaluate log posterior predictive
def logpp(y,x,beta, alpha):
    Ns = np.shape(beta)[0]
    logpp = np.zeros(Ns)
    pred =np.zeros(Ns)
    for n in range(Ns):
        z = np.dot(x,beta[n]) + alpha[n]
        logeta = -np.logaddexp(0,-z)
        logneta = -np.logaddexp(0,z)
        logpp[n] = np.sum(y * logeta + (1-y)*logneta)
    logpp_mean = (sp.special.logsumexp(logpp)) - np.log(Ns)
    return logpp_mean

#evaluate LPPD
def lppd(y,x,beta,alpha):
    Ns = np.shape(beta)[0]
    N =np.shape(y)[0]
    lppd = np.zeros((Ns,N))
    pred =np.zeros(Ns)
    for n in range(Ns):
        z = np.dot(x,beta[n]) + alpha[n]
        logeta = -np.logaddexp(0,-z)
        logneta = -np.logaddexp(0,z)
        lppd[n] = y * logeta + (1-y)*logneta
    lppd = sp.special.logsumexp(lppd,axis = 0) - np.log(Ns)
    lppd_sum = np.sum(lppd)
    return lppd_sum

#evaluate classification percentage correct
def predcorrect(y,x,beta,alpha):
    Ns = np.shape(beta)[0]
    N =np.shape(y)[0]
    pred = np.zeros(N)
    N_error = np.zeros(Ns)
    logeta = np.zeros((Ns,N))
    for n in range(Ns):
        z = np.dot(x,beta[n]) + alpha[n]
        logeta[n] = -np.logaddexp(0,-z)
    logeta_mean = sp.special.logsumexp(logeta,axis = 0) - np.log(Ns)
    pred[np.exp(logeta_mean) >= 0.5] = 1
    N_error  = np.sum(np.abs(pred-y))
    return (N-N_error)/N

#evaluate MSE 
def MSE(y,x,beta,alpha):
    Ns = np.shape(beta)[0]
    N =np.shape(y)[0]
    pred = np.zeros(N)
    MSE = np.zeros(Ns)
    logeta = np.zeros((Ns,N))
    for n in range(Ns):
        z = np.dot(x,beta[n]) + alpha[n]
        logeta[n] = -np.logaddexp(0,-z)

    #average p(ytest | beta) then re-log
    logeta_mean = sp.special.logsumexp(logeta,axis = 0) - np.log(Ns)
    MSE = np.mean((np.exp(logeta_mean) - y)**2)
    return MSE

#check cardinality of beta
def checkcard(beta,epsilon):
    Ns = np.shape(beta)[0]
    card = np.count_nonzero(np.abs(beta)> epsilon,axis = 1)
    card_mean = np.mean(card)
    return card_mean

