# npl
Python3 code for Bayesian Nonparametric Learning. 

Paper:



## Getting Started
To install npl package, clone the repo and run
```
python3 setup.py develop
```
## Python Dependencies
Although the setup installs the packages automatically, you may need to install pystan separately using pip.


## Core Usage and Parallel Processing
* Current implementation will use all cores available on the local computer. If this is undesired, in the run scripts, pass the number of cores `n_cores` to `bootstrap_gmm` or `bootstrap_logreg`
* If running on multi-core computer, make sure to restrict numpy to use 1 thread per process for joblib to parallelize without CPU oversubscription, with the bash command:
`export OPENBLAS_NUM_THREADS=1`

## Overview
A directory overview is given below:
*`npl` - Contains main functions for the posterior bootstrap and evaluating posterior samples
*`experiments` - Contains scripts for running main experiments
*`supp_experiments` - Contains scripts for running supplementary experiments

## `npl` Structure
* `bootstrap_logreg.py` and `bootstrap_gmm.py` contain the main posterior bootstrap sampling functions
* `maximise_logreg.py` and `maximise_gmm.py` contain functions for sampling the prior pseudo-samples, initialising random restarts and maximising the weighted log likelihood. These functions can be edited to use NPL with different models and priors.
* `./evaluate` contains functions for calculating log posterior predictives of the different posteriors

## Experiments
### __Example 3.1__ - Toy GMM (in `./experiments/Toy_GMM`)

1. Run `generate_gmm.py` to generate toy data. The files in `./sim_data_plot` are the train/test data used for the plots in the paper.

2. 

3. 

### __Example 3.1__ - MNIST GMM (in `./experiments/MNIST_GMM`)

Download MNIST files from http://yann.lecun.com/exdb/mnist/, place in `./samples`.

### __Example 3.2__ - Logistic Regression with ARD priors (in `./experiments/LogReg_ARD`)

Place all data files downloaded off UCI ML repo in  `./data`, then run `load_data.py`.

### __Example 3.3__ - Bayesian Sparsity Path Analysis (in .`/experiments/Genetics`)
 
Covariate data is not included for privacy reasons; run `load_data.py` to generate simulated covariates from N(0,1) (uncorrelated unlike real data, and not tested) and pseudo-phenotypes. 

### __Example E.1__ - Normal Location Model (in `./supp_experiments/Normal`)

Generated in Jupyter notebook `Normal location model.ipynb`

### __Example E.2.3 - E.2.4__ - Comparison to Importance Sampling and MDP-NPL (in `./supp_experiments/Toy_GMM`)
Run `generate_gmm.py` to generate toy data. The files in `./sim_data_plot` are the train/test data used for the plots in the paper.

Jupyter notebook `Normal location model.ipynb`

### __Example E.2.3 - E.2.4__ - Comparison to Importance Sampling and MDP-NPL (in `./supp_experiments/Toy_GMM`)
Run `generate_gmm.py` to generate toy data. The files in `./sim_data_plot` are the train/test data used for the plots in the paper.
