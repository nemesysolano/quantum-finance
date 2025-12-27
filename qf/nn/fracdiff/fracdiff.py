import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

def get_binomial_weights(d, k):
    """
    Calculates weights w_i: w_i = w_{i-1} * (i - 1 - d) / i | w_1 = -d
    """
    if k < 1: return np.array([])
    weights = [-float(d)]
    for i in range(2, k + 1):
        weights.append(weights[-1] * (i - 1 - d) / i)
    return np.array(weights)

def perform_ols_and_fit(past_diffs, target_diff, k):
    """
    Estimates 'd' by fitting the binomial model to empirical OLS coefficients.
    """
    if len(past_diffs) < k: return 0.5
    X_rows, y_vals = [], []
    
    # 1. Build historical OLS matrix
    for i in range(k, len(past_diffs)):
        X_rows.append(past_diffs[i-k:i][::-1])
        y_vals.append(past_diffs[i])
    
    # 2. Add the most recent known window and its resulting diff (target_diff)
    X_rows.append(past_diffs[-k:][::-1])
    y_vals.append(target_diff)
    
    X_mat, y_vec = np.array(X_rows), np.array(y_vals)
    # Solve for weights that best explain current price dynamics
    ols = LinearRegression(fit_intercept=False).fit(X_mat, y_vec)
    
    def binomial_model(indices, d_val):
        return get_binomial_weights(d_val, len(indices))

    try:
        d_refined, _ = curve_fit(binomial_model, np.arange(1, k+1), ols.coef_, p0=0.5, bounds=(0.01, 0.99))
        return d_refined[0]
    except:
        return np.clip(-ols.coef_[0], 0.01, 0.99)

def predict_next_price(window_diffs, d, k, last_raw_price):
    """
    Predicts x(t+1) by projecting the fractional innovation.
    """
    weights = get_binomial_weights(d, k)
    pred_delta = np.clip(np.dot(weights, window_diffs), -0.99, 0.99)
    # Reversing the Bounded Percentage Difference formula
    return last_raw_price * (1 + pred_delta) / (1 - pred_delta)