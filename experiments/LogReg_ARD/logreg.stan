data {
    int<lower = 0> N; //  number of data points
    int<lower = 0> D; //  number of dimensions
    int<lower=0,upper=1> y[N]; // observables
    matrix[N,D]    x;      //input matrix

    #hyper prior parameters
    real<lower = 0> a; // Gamma prior for lambda
    real<lower = 0> b; // Gamma prior for lambda
}


parameters {
    vector[D] beta;             //regression weights
    real alpha;                 //intercept
    vector<lower = 0>[D] lambda; //local precision (1/variance)
}

model {
    //priors
    for (i in 1:D) {
        lambda[i] ~ gamma(a,b);
        beta[i] ~ normal(0, 1/sqrt(lambda[i]));
    }
    //likelihood
    y ~ bernoulli_logit(alpha + x * beta);
    }