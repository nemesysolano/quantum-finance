import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

def get_shannon_entropy(prices, window=20, bins=10):
    """
    Calculates Shannon Entropy of discretized returns. 
    High values (> 0.8 normalized) indicate a 'Memoryless' Random Walk.
    """
    returns = np.diff(prices)
    if len(returns) < window:
        return 1.0 # Assume maximum chaos for insufficient data
    
    # Calculate entropy for the rolling window
    hist, _ = np.histogram(returns[-window:], bins=bins, density=True)
    # Filter out zero probabilities to avoid log(0)
    p = hist[hist > 0]
    entropy = -np.sum(p * np.log2(p))
    
    # Normalize by max possible entropy for the given bin count
    max_entropy = np.log2(bins)
    return entropy / max_entropy

def is_binary_event(atr_history, window=20, threshold=2.5):
    """
    Detects if current volatility is a statistical outlier.
    """
    if len(atr_history) < window:
        return False
    
    recent_vols = atr_history[-window:]
    mean_vol = np.mean(recent_vols)
    std_vol = np.std(recent_vols)
    
    z_score = (atr_history[-1] - mean_vol) / (std_vol + 1e-9)
    return z_score > threshold

def get_atr(y_actual, window=14, use_percent=True):
    """
    Calculates a Normalized ATR (ATR%) or standard ATR with a noise buffer.
    ATR% = ((Mean_Range + 1.5*Std_Range) / Price) * 100
    """
    if len(y_actual) < window + 1:
        return np.zeros(len(y_actual))
    
    diffs = np.abs(np.diff(y_actual))
    atr_values = np.zeros(len(y_actual))
    
    for i in range(window, len(y_actual)):
        window_diffs = diffs[i-window:i]
        vol_mean = np.mean(window_diffs)
        vol_std = np.std(window_diffs)
        
        # Robust Noise Band
        raw_vol = vol_mean + (1.5 * vol_std)
        
        if use_percent:
            # Normalize by current price to get ATR%
            atr_values[i] = (raw_vol / y_actual[i]) * 100
        else:
            atr_values[i] = raw_vol
    
    atr_values[:window] = atr_values[window]
    return atr_values

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
    pred_delta_scaler =  (1 + pred_delta) / (1 - pred_delta)
    next_price = last_raw_price * pred_delta_scaler
    return next_price, pred_delta