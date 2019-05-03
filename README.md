# npl
This repository contains Python code for Bayesian Nonparametric Learning with a Dirichlet process prior. More details can be found in the paper below: 

Fong, E., Lyddon, S. and Holmes, C. **Scalable Nonparametric Sampling from Multimodal Posteriors with the Posterior Bootstrap.** In *Proceedings of the Thirty-sixth International Conference on Machine Learning (ICML) 2019.*
https://arxiv.org/abs/1902.03175

## Getting Started
To install the npl package, clone the repository and run
```
python3 setup.py develop
```
## Python Dependencies
Although the setup installs the packages automatically, you may need to install `pystan` separately using `pip` if `setuptools` isn't working correctly. Please make sure the version of `pystan` is newer than v2.19.0.0 or the evaluate scripts may not work properly. The code has been tested on Python 3.6.7. 

## Core Usage and Parallel Processing
* Current implementation will use all cores available on the local computer. If this is undesired, pass the number of cores as `n_cores` to the function `bootstrap_gmm` or `bootstrap_logreg`  in the run scripts,.
* If running on multi-core computer, make sure to restrict `numpy` to use 1 thread per process for `joblib` to parallelize without CPU oversubscription, with the bash command:
`export OPENBLAS_NUM_THREADS=1`

## Overview
A directory overview is given below:
* `npl` - Contains main functions for the posterior bootstrap and evaluating posterior samples on test data
* `experiments` - Contains scripts for running main experiments
* `supp_experiments` - Contains scripts for running supplementary experiments

## Structure of  `npl` 
A overview of the structure of `npl` is given below:
* `bootstrap_logreg.py` and `bootstrap_gmm.py` contain the main posterior bootstrap sampling functions
* `maximise_logreg.py` and `maximise_gmm.py` contain functions for sampling the prior pseudo-samples, initialising random restarts and maximising the weighted log likelihood. These functions can be edited to use NPL with different models and priors.
* `./evaluate` contains functions for calculating log posterior predictives of the different posteriors

## Datasets
### __Example 3.1.2__ - MNIST GMM (in `./experiments/MNIST_GMM`)
1. Download MNIST files from http://yann.lecun.com/exdb/mnist/. 
2. Extract and place in `./samples`, so the folder contains the files:
```
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
train-images-idx3-ubyte
train-labels-idx1-ubyte
```

### __Example 3.2__ - Logistic Regression with ARD priors (in `./experiments/LogReg_ARD`)
1. Download the __Adult__, __Polish companies bankruptcy 3rd year__, and __Arcene__ datasets from UCI Machine Learning Repository, links below: 
* __Adult__ - https://archive.ics.uci.edu/ml/datasets/adult
* __Polish__- https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data
*  __Arcene__- https://archive.ics.uci.edu/ml/datasets/Arcene

2. Extract and place all data files in  `./data`, so the folder contains the files:
```
3year.arff
adult.data
adult.test
arcene_train.data
arcene_train.labels
arcene_valid.data
arcene_valid.labels
```

## Experiments
### __Example 3.1.1__ - Toy GMM (in `./experiments/Toy_GMM`)

1. Run `generate_gmm.py` to generate toy data. The files in `./sim_data_plot` are the train/test data used for the plots in the paper, and the files in `./sim_data` are the datasets for the tabular results.
2. Run `run_NPL_toygmm.py` for the NPL example and `run_stan_toygmm.py` for the NUTS and ADVI examples.
3. Run `evaluate_posterior_toygmm.py` to evaluate posterior samples. The Jupyter notebook `Plot bivariate KDEs for GMM.ipynb` can be used to produce posterior plots.

### __Example 3.1.2__ - MNIST GMM (in `./experiments/MNIST_GMM`)

1. Run `run_NPL_MNIST.py` for the NPL example and `run_stan_MNIST.py` for the NUTS and ADVI examples.
2. Run `evaluate_posterior_MNIST.py` to evaluate posterior samples. The Jupyter notebook `Plot MNIST KDE.ipynb` can be used to produce posterior plots.
 

### __Example 3.2__ - Logistic Regression with ARD priors (in `./experiments/LogReg_ARD`)

1. Run `load_data.py` to preprocess data and generate different train-test splits.
2. Run `run_NPL_logreg.py` for the NPL example and `run_stan_logreg.py` for the NUTS and ADVI examples.
3. Run `evaluate_posterior_logreg.py` to evaluate posterior samples. The Jupyter notebook `Plot marginal KDE (for Adult).ipynb` can be used to produce posterior plots.


### __Example 3.3__ - Bayesian Sparsity Path Analysis (in .`/experiments/Genetics`)
 
1. Covariate data is not included for privacy reasons. Run `load_data.py` to generate simulated covariates from Normal(0,1) (uncorrelated unlike real data) and pseudo-phenotypes. 
2. Run `run_NPL_genetics.py` for the NPL example.
3. The Jupyter notebook `Plotting Sparsity Plots.ipynb` can be used to produce sparsity plots.


## Supplementary Material Experiments
### __Example E.1__ - Normal Location Model (in `./supp_experiments/Normal`)

1. The Jupyter notebook `Normal location model.ipynb` contains all experiments and plots.

### __Example E.2.3__ - Comparison to Importance Sampling (in `./supp_experiments/Toy_GMM`)
1. Run `generate_gmm.py` to generate toy data. The files in `./sim_data_plot` are the train/test data used for the plots in the paper.
2. Run `main_IS` function in `run_NPL_toygmm.py` for the NPL example; run `run_IS_toygmm.py` for the importance sampling example.
3. Run `evaluate_posterior_toygmm.py` to evaluate posterior samples on test data. The Jupyter notebook `Plot bivariate KDEs for GMM.ipynb` can be used to produce posterior plots.


### __Example E.2.4__ - Comparison to MDP-NPL (in `./supp_experiments/Toy_GMM`)
1. Run `generate_gmm.py` to generate toy data. The files in `./sim_data_plot` are the train/test data used for the plots in the paper, and the files in `./sim_data` are for the tabular results.
2. First run `run_stan_toygmm` to generate the NUTS (required for MDP-NPL) and ADVI samples, then run `main_MDP` and `main_DP` in `run_NPL_toygmm.py` for MDP-NPL and DP-NPL respectively.
3. Run `evaluate_posterior_toygmm.py` to evaluate posterior samples on test data. The Jupyter notebook `Plot bivariate KDEs for GMM.ipynb` can be used to produce posterior plots.
