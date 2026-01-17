import numpy as np

# Continuous
def mgf_normal(t, mu, sigma):
    return np.exp(mu*t + 0.5*sigma**2*t**2)

def mgf_exponential(t, lam):
    return lam / (lam - t)

def mgf_gamma(t, alpha, beta):
    return (beta / (beta - t))**alpha

# Discrete
def mgf_bernoulli(t, p):
    return (1 - p) + p*np.exp(t)

def mgf_binomial(t, n, p):
    return (1 - p + p*np.exp(t))**n

def mgf_poisson(t, lam):
    return np.exp(lam * (np.exp(t) - 1))

