import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

def get_binomial_weights(d, k):
    """
    Calculates weights w_i for the prediction model.
    Formula: w_i = w_{i-1} * (i - 1 - d) / i | w_1 = -d
    """
    if k < 1: return np.array([])
    # Based on the recurrence, if w_0 = 1, then w_1 = -d
    weights = [-float(d)]
    for i in range(2, k + 1):
        # Recursive step from README.md
        weights.append(weights[-1] * (i - 1 - d) / i)
    return np.array(weights)

def perform_ols_and_fit(past_diffs, target_diff, k):
    """
    Internal helper that maps OLS coefficients to the fractional parameter d.
    """
    # 1. Generate a local training set using a sliding window for stability
    X_rows, y_vals = [], []
    for i in range(k, len(past_diffs)):
        X_rows.append(past_diffs[i-k:i][::-1]) # Reverse to align lags (t-1, t-2...)
        y_vals.append(past_diffs[i])
    
    # Append the most recent window and the current target difference
    X_rows.append(past_diffs[-k:][::-1])
    y_vals.append(target_diff)
    
    X_mat, y_vec = np.array(X_rows), np.array(y_vals)
    
    # 2. Solve for Empirical Weights using OLS
    ols = LinearRegression(fit_intercept=False).fit(X_mat, y_vec)
    empirical_weights = ols.coef_

    # 3. Map Empirical Weights to the parameter 'd' via Curve Fitting
    def binomial_model(indices, d_val):
        return get_binomial_weights(d_val, len(indices))

    indices = np.arange(1, len(empirical_weights) + 1)
    try:
        # Fit the binomial weights curve to the OLS coefficients
        d_refined, _ = curve_fit(
            binomial_model, 
            indices, 
            empirical_weights, 
            p0=0.5, 
            bounds=(0.01, 0.99) # Constraint for stationary fractional range
        )
        return d_refined[0]
    except Exception:
        # Fallback: Since w_1 = -d, the first weight is the most direct proxy
        return np.clip(-empirical_weights[0], 0.01, 0.99)

def predict_next_price(window_diffs, d, k, last_raw_price):
    """Reverses the Δ% formula to project absolute price x(t+1)."""
    weights = get_binomial_weights(d, k)
    pred_delta = np.clip(np.dot(weights, window_diffs), -0.99, 0.99)
    # x(t+1) = x(t) * (1 + Δ%) / (1 - Δ%)
    return last_raw_price * (1 + pred_delta) / (1 - pred_delta)