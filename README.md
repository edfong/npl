# npl
Python3 code for Bayesian Nonparametric Learning. 

Paper:



## Getting Started
To install npl package, clone the repo and run
```
python3 setup.py develop
```
## List of Python Dependencies
numpy
scipy
scikit-learn
pandas
matplotlib
seaborn
pystan
joblib
tqdm
python-mnist

Although the setup installs the packages automatically, you may need to install pystan separately using pip.


## Core Usage and Parallel Processing
* Current implementation will use all cores available on the local computer. If this is undesired, in the run scripts, pass `n_cores` to `bootstrap_gmm` or `bootstrap_logreg`
* If running on multi-core computer, make sure to restrict numpy to use 1 thread per process for joblib to parallelize without CPU oversubscription, with the bash command:
`export OPENBLAS_NUM_THREADS=1`

## Overview
A directory overview is given below:
*`npl` - Contains core files for the posterior bootstrap and evaluating posterior samples
*`experiments` - Contains scripts for running main experiments
*`supp_experiments` - Contains scripts for running supplementary experiments

## Datasets
### __Example 3.1__ - Toy GMM (in `./experiments/Toy_GMM`)

Run `generate_gmm.py` to generate toy data. The files in `./sim_data_plot` are the train/test data used for the plots in the paper

### __Example 3.1__ - MNIST GMM (in `./experiments/MNIST_GMM`)

Download MNIST files from http://yann.lecun.com/exdb/mnist/, place in `./samples`.

### __Example 3.2__ - Logistic Regression with ARD priors (in `./experiments/LogReg_ARD`)

Place all data files downloaded off UCI ML repo in  `./data`, then run `load_data.py`.

### __Example 3.3__ - Bayesian Sparsity Path Analysis (in .`/experiments/Genetics`)
 
Covariate data is not included for privacy reasons; run `load_data.py` to generate simulated covariates from N(0,1) (uncorrelated unlike real data, and not tested) and pseudo-phenotypes. 

### __Example E.1__ - Normal Location Model (in `./supp_experiments/Normal`)

Generated in Jupyter notebook `Normal location model.ipynb`

### __Example E.2.3__ - Comparison to Importance Sampling

### __Example E.2.4__ - Comparison to MDP-NPL 

## `npl` 
