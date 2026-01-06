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
    Estimates loc (μ), scale (σ), and df (ν) with standardization for numerical stability.
    """
    data = np.asarray(time_series)
    data = data[~np.isnan(data)] # Filter NaNs
    
    # 1. STANDARDIZATION (Crucial for FOREX)
    # We store these to rescale the results later
    data_mean = np.mean(data)
    data_std = np.std(data)
    
    # Prevent division by zero if data is flat
    if data_std < 1e-9:
        return data_mean, 0.0, 4.0, data_mean, 0.0

    # Transform data to be roughly N(0, 1)
    standardized_data = (data - data_mean) / data_std
    
    # 2. Optimization on Standardized Data
    # Initial guesses are now simple because data is standardized
    initial_params = [0.0, 1.0, 4.0] # mu=0, sigma=1, nu=4
    
    bounds = [(-np.inf, np.inf), (1e-6, np.inf), (1e-6, np.inf)]
    
    try:
        result = minimize(
            t_nll, 
            initial_params, 
            args=(standardized_data,), 
            method='L-BFGS-B', 
            bounds=bounds
        )
        
        mu_std, sigma_std, nu = result.x
        
        # 3. Rescale Parameters back to Original Space
        # mu_real = (mu_std * scale) + center
        # sigma_real = sigma_std * scale
        mu = (mu_std * data_std) + data_mean
        sigma = sigma_std * data_std
        
        return mu, sigma, nu, data_mean, data_std

    except Exception:
        # Fallback if optimization still fails
        return data_mean, data_std, 4.0, data_mean, data_std
    
def quantum_lambda(return_p):    
    # Your current estimate_studen_t is good because it standardizes
    μ, σ, ν, _, std = estimate_studen_t(return_p)
    
    # We use σ from the MLE fit; it represents the 'width' of the well
    s = σ if (not np.isnan(σ) and σ > 0) else std
    
    # The math for lambda (L) should remain dimensionless
    L0 = s
    f0 = scaled_φ(L0, μ, s, ν)
    L1 = -s
    f1 = scaled_φ(L1, μ, s, ν)    
    
    L = np.log(np.abs(L0**2 * f0 - L1**2 * f1) / (1e-9 + np.abs(L1**4 * f1 - L0**4 * f0)))
    
    return L
    
def quantum_energy_level(l, n): 
    l = l if l > 10 else l
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

def maximum_energy_level(x, l, market_type="STOCK"):
    max_n = 2000 
    
    # Define the reference scale based on the market
    # Stocks move in units of 1.0+, FOREX moves in units of 0.0001
    base_scale = 1.0 if market_type == "STOCK" else 0.0001
    
    # Normalize inputs for the physics search
    x_scaled = x / base_scale
    l_scaled = l / base_scale

    last_E = -np.inf
    for n in range(max_n):
        E_n = quantum_energy_level(l_scaled, n)
        if E_n < x_scaled:
            last_E = E_n
        else:
            break
            
    # Rescale the result back to the actual market price
    return last_E * base_scale if last_E > -np.inf else np.nan

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
