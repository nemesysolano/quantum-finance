import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.special import gammaln
from scipy.optimize import minimize
from scipy.special import gammaln

lambda_epsilon = 1e-9

def scaled_φ(ξ, μ, σ, ν):
    """
    Fast Student-T PDF implementation using log-gamma to avoid scipy.stats overhead.
    ξ is the point at which to evaluate.
    """
    # Using the log-space formula for numerical stability and speed
    term1 = gammaln((ν + 1) / 2) - gammaln(ν / 2)
    term2 = 0.5 * np.log(ν * np.pi) + np.log(σ)
    term3 = -((ν + 1) / 2) * np.log(1 + (ξ - μ)**2 / (ν * σ**2))
    
    return np.exp(term1 - term2 + term3)

def t_nll(params, data):
    """
    Calculates the Negative Log-Likelihood of the Student-t distribution.
    Minimizing this value yields the Maximum Likelihood Estimate.
    """
    mu, sigma, nu = params
    
    # Constraints: sigma > 0, nu > 0
    # We return infinity if constraints are violated to guide the optimizer away
    if sigma <= 0 or nu <= 0:
        return np.inf

    n = len(data)
    
    # Log-Likelihood Formula (vectorized)
    # Term 1: Log of the Gamma/normalization constants
    term1 = n * gammaln((nu + 1) / 2) - n * gammaln(nu / 2)
    term2 = -0.5 * n * np.log(nu * np.pi) - n * np.log(sigma)
    
    # Term 3: The data-dependent part
    residuals = (data - mu) / sigma
    term3 = -0.5 * (nu + 1) * np.sum(np.log1p((residuals**2) / nu))
    
    log_likelihood = term1 + term2 + term3
    
    return -log_likelihood  # Return negative for minimization

def estimate_studen_t(time_series):
    """
    Estimates loc (μ), scale (σ), and df (ν) without scipy.stats.
    """
    data = np.asarray(time_series)
    data = data[~np.isnan(data)] # Filter NaNs
    
    # 1. Initial Guesses (Method of Moments)
    # These provide a "warm start" for the optimizer, speeding up convergence.
    mu_init = np.mean(data)
    mean = mu_init
    sigma_init = np.std(data)
    std = sigma_init
    nu_init = 4.0 # 4 is a safe starting point for financial returns (fat tails)
    
    initial_params = [mu_init, sigma_init, nu_init]
    
    # 2. Optimization
    # L-BFGS-B is fast and handles bounds (sigma > 0, nu > 0) efficiently.
    # We set lower bounds to slight epsilon above 0 to avoid div/0 errors.
    bounds = [(-np.inf, np.inf), (1e-6, np.inf), (1e-6, np.inf)]
    
    result = minimize(
        t_nll, 
        initial_params, 
        args=(data,), 
        method='L-BFGS-B', 
        bounds=bounds
    )
    
    mu, sigma, nu = result.x
    return mu, sigma, nu, mean, std

def quantum_lambda(return_p):    
    μ, σ, ν, _, std = estimate_studen_t(return_p)
    s = std if np.isnan(σ) else std
    L0  = s
    f0 = scaled_φ(L0, μ, s, ν)
    L1  = -s
    f1 = scaled_φ(L1, μ, s, ν)    
    L = np.abs(L0**2 * f0 - L1**2 * f1)/(1e-9 + np.abs(L1**4 * f1 - L0**4 * f0))
    return L

    
def quantum_energy_level(l, n): 
    k_n = np.cbrt((1.1924 + 33.2383*n +  56.2169 * np.power(n,2))/ (1 + 43.6196*n))
    p = -np.power(2*n + 1, 2)
    q = -l * np.power(2*n + 1, 3) * np.power(k_n, 3)
    sqrt_arg = np.power(np.abs(q), 2) / 4 + np.power(np.abs(p),3) / 27    #TODO: Handle negative numbers here.
    sub_sqrt = np.sqrt(sqrt_arg)
    if np.isnan(sub_sqrt):
        raise ValueError("sub_sqrt is nan. Aborting calculation.")
    half_q = q / 2
    E = np.cbrt(-half_q + sub_sqrt) + np.cbrt(-half_q - sub_sqrt)
    return E

def quantum_energy_levels(l, minimum, maximum):
    k = 1
    E1 = quantum_energy_level(l, k)    
    E0 = E1

    while E1 < minimum:        
        k += 1
        E0 = E1
        E1 = quantum_energy_level(l, k)

    E = [E0]        
    while E1 < maximum:        
        k += 1
        E1 = quantum_energy_level(l, k)
        E.append(E1)

    return E

def maximum_energy_level(x, l):
    max_n=100000
    """
    Finds the highest energy level E^(n) that is strictly less than x.
    """
    last_E = -np.inf
    # Start from n=0 to calculate energy levels
    for n in range(max_n):
        E_n = quantum_energy_level(l, n)
        if E_n < x:
            last_E = E_n
        else:
            # Energy levels are monotonically increasing, so we can stop
            # once we find a level greater than or equal to x.
            break
    # Return the last found level that was less than x, or nan if none were.
    return last_E if last_E > -np.inf else np.nan

def minimum_energy_level(x, l):
    max_n=100000
    """
    Finds the lowest energy level E^(n) that is strictly greater than x.
    """
    # Start from n=0 to calculate energy levels
    for n in range(max_n):
        E_n = quantum_energy_level(l, n)
        if E_n > x:
            return E_n
    # If no level is found greater than x within max_n, return nan.
    return np.nan
