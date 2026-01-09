import numpy as np
import pandas as pd
from nn import fracdiff as frac
from qf.quantum import quantum_lambda
from qf.quantum import maximum_energy_level, minimum_energy_level

def add_breaking_gap(historical_data, Q=0.0001):
    """
    Calculates the breaking gap G(t) based on structural breaches and decay.
    
    Args:
        historical_data (pd.DataFrame): Must contain 'open', 'High', 'Low', 'Close'.
        Q (float): Per-bar decay rate.
        
    Returns:
        pd.DataFrame: Dataframe with 'G' (Gap magnitude) and 'Gd' (Gap direction/type).
    """
    df = historical_data
    n = len(df)
    
    # Initialize output columns
    g_values = np.zeros(n)
    gd_values = [None] * n # Renamed from breach_type to Gd
    
    # State variables for the "last structural breach"
    gb = 0.0      # Initial magnitude of the last breach
    k = 0         # Bars elapsed since breach
    last_type = None
    
    # Calculations require at least 4 bars to check t-3 indices
    for t in range(3, n):
        current_gb = 0.0
        current_type = None
        
        # 1. Check for Support Breach (Ascending Trend Violation)
        if (df['Close'].iloc[t-3] < df['Close'].iloc[t-2] and 
            df['Low'].iloc[t] < df['Low'].iloc[t-2] and 
            df['Low'].iloc[t-2] < df['Low'].iloc[t-1]):
            
            current_gb = df['Low'].iloc[t-2] - df['Low'].iloc[t]
            current_type = -1
            
        # 2. Check for Resistance Breach (Descending Trend Violation)
        elif (df['Close'].iloc[t-3] > df['Close'].iloc[t-2] and 
              df['High'].iloc[t] > df['High'].iloc[t-2] and 
              df['High'].iloc[t-2] > df['High'].iloc[t-1]):
              
            current_gb = df['High'].iloc[t] - df['High'].iloc[t-2]
            current_type = 1
            
        # 3. Update Breach State if a new violation occurs
        if current_gb > 0:
            gb = current_gb
            k = 1 
            last_type = current_type
        else:
            k += 1 # Increment time elapsed since the last stored Gb
            
        # 4. Apply Decay Formula: G(t) = max(Gb - k*Q, 0)
        g_values[t] = max(gb - (k * Q), 0)
        gd_values[t] = last_type

    df['G'] = g_values
    df['Gd'] = gd_values # Directional label
    df.dropna(inplace=True)
    return df

def add_swing_ratio(historical_data): # Fixed Critical Leakage, Jan 8 2026
    """
    Calculates the swing ratio S(t) based on the breaking gap and local range.
    
    Formula: S(t) = |G(t)| / A(t)
    Where:
        A(t) = max(|G(t)|, R(t))
        R(t) = max(h_{t-i}) - min(l_{t-i}) for i in 0...3
    
    Args:
        historical_data (pd.DataFrame): Dataframe already containing 'G' (Breaking Gap).
                                      Must contain 'High' and 'Low'.
    Returns:
        pd.DataFrame: Dataframe with added 'S' (Swing Ratio) column.
    """
    df = historical_data
    n = len(df)
    s_values = np.zeros(n)
    
    # Requirement: Sequence must be longer than 2 bars. 
    # R(t) uses a lookback of 4 bars (i=0 to 3).
    for t in range(3, n):
        # 1. Calculate Local Range R(t) over the last 4 bars (0, 1, 2, 3)
        # Using .iloc[t-3:t+1] to get the slice of 4 elements ending at t
        local_highs = df['High'].iloc[t-3:t+1]
        local_lows = df['Low'].iloc[t-3:t+1]        
        rt = local_highs.max() - local_lows.min()
        
        # 2. Calculate Absolute Reference A(t)
        # G(t) magnitude is stored in column 'G'
        at = max(abs(df['G'].iloc[t]), rt)
        
        # 3. Calculate Swing Ratio S(t)
        if at > 0:
            s_values[t] = abs(df['G'].iloc[t]) / at
        else:
            s_values[t] = 0.0
            
    df['S'] = s_values
    df.dropna(inplace=True)
    return df

def add_directional_probabilities(historical_data, epsilon=9**-5):
    """
    Calculates directional probabilities P↑(t) and P↓(t) based on structural breaches.
    
    This implementation follows the requirements from the README:
    - Uses the Serial Bounded Ratio δ(x(t)) instead of Squared Serial Difference.
    - Applies the Logarithmic Filter ρ(x) to the price ratio.
    - Ensures probabilities are complementary where P↑(t) + P↓(t) = 1 (when δ=1).
    """
    df = historical_data
    n = len(df)
    
    # Initialize output arrays
    p_up = np.zeros(n)
    p_down = np.zeros(n)
    
    # 1. Define Logarithmic Filter ρ(x)
    # ρ(x) = (2 / log 2) * (log(1+x) / (1+x))
    def rho(x):
        return (2 / np.log(2)) * (np.log(1 + x) / (1 + x))

    # 2. Calculate Serial Bounded Ratio δ(c(t))
    # δ(c(t)) = ρ(max(c(t)/c(t-1), ε))
    c = df['Close'].values
    c_prev = df['Close'].shift(1).values
    
    # The README specifies a strictly positive time series for the ratio
    # np.maximum ensures the ratio is at least ε to prevent log(0) errors
    raw_ratio = c / (c_prev + 1e-9) 
    bounded_input = np.maximum(raw_ratio, epsilon)
    
    # Apply the logarithmic filter to calculate δ(c(t))
    # We clip the input to 1.0 as the filter domain is x ∈ (ε, 1]
    delta_ct = rho(np.minimum(bounded_input, 1.0)) 

    # 3. Apply Probability Mapping based on Breach Case
    for t in range(1, n):
        st = df['S'].iloc[t]   # Swing Ratio (Bounded [0, 1])
        gd = df['Gd'].iloc[t]  # Breach Direction (1: Resistance, -1: Support)
        d_val = delta_ct[t]    # Serial Bounded Ratio
        
        # Resistance Breach Case (Descending Trend Violation)
        # Current gap exerts downward pressure
        if gd == 1:
            p_down[t] = st * d_val
            p_up[t] = 1.0 - st
            
        # Support Breach Case (Ascending Trend Violation)
        # Current gap exerts upward pressure
        elif gd == -1:
            p_up[t] = st * d_val
            p_down[t] = 1.0 - st
            
        # Default/Equilibrium state (No breach history)
        else:
            p_up[t] = 0.5
            p_down[t] = 0.5
            
    df['P↑'] = p_up
    df['P↓'] = p_down
    df.dropna(inplace=True)
    return df


def add_price_volume_oscillator(historical_data):
    """
    Calculates the price-volume oscillator Y(t).
    
    Formula: Y(t) = Δp(t) * Δ²v(t)
    Where:
        Δp(t) = Bounded percentage difference of Close price (k=1)
        Δ²v(t) = Squared bounded percentage difference of Volume (k=1)
        
    Args:
        historical_data (pd.DataFrame): Dataframe containing 'Close' and 'Volume'.
        
    Returns:
        pd.DataFrame: Dataframe with added 'Y' column.
    """
    df = historical_data
    
    p = df['Close'].values

    # Calculate shifts (t-1)
    p_prev = df['Close'].shift(1).values

    # 1. Calculate Δp(t): Bounded Percentage Difference of Price
    # Formula: (p(t) - p(t-1)) / (|p(t)| + |p(t-1)|)
    delta_p = (p - p_prev) / (np.abs(p) + np.abs(p_prev))


    if df.loc[df.index[0], 'Volume'] > 0:
        v = df['Volume'].values
        
        # Calculate shifts (t-1)
        v_prev = df['Volume'].shift(1).values
                
        # 2. Calculate Δv(t): Bounded Percentage Difference of Volume
        delta_v = 1 if v[0] != 0 else (v - v_prev) / (np.abs(v) + np.abs(v_prev))
        
        # 3. Calculate Δ²v(t): Squared Serial Difference of Volume
        delta_v_sq = np.square(delta_v)
        
        # 4. Calculate Y(t)
        df['Y'] = delta_p * delta_v_sq
        
    else:
        # 4. Calculate Y(t)
        df['Y'] = delta_p
        
    df.dropna(inplace=True)
    return df

def add_price_time_angles(historical_data, epsilon=1e-9):
    """
    Calculates the four Price-Time Angles (Θ1...Θ4) based on structural geometry.
    REVISED: Uses t-1 values as reference to ensure zero lookahead bias.
    """
    df = historical_data
    n = len(df)
    
    # Initialize result arrays
    theta = {f'ϴ{k}': np.zeros(n) for k in range(1, 5)}
    
    # Start from index 2 because we need at least one lookback bar for reference
    for t in range(2, n):
        # REFERENCE: We use values from t-1 to define the "current" state
        # This ensures the angle is known at the OPEN of bar t.
        h_ref = df['High'].iloc[t-1]
        l_ref = df['Low'].iloc[t-1]
        
        # 1. Find Closest Extremes relative to the reference bar (t-1)
        # We search from j=2 (which is t-2) backward to the start
        try:
            # Closest Higher High relative to h_ref
            i_h_up = next((j for j in range(2, t + 1) if df['High'].iloc[t-j] > h_ref), 2)
            h_up = df['High'].iloc[t - i_h_up]
            
            # Closest Lower High relative to h_ref
            i_h_down = next((j for j in range(2, t + 1) if df['High'].iloc[t-j] < h_ref), 2)
            h_down = df['High'].iloc[t - i_h_down]
            
            # Closest Higher Low relative to l_ref
            i_l_up = next((j for j in range(2, t + 1) if df['Low'].iloc[t-j] > l_ref), 2)
            l_up = df['Low'].iloc[t - i_l_up]
            
            # Closest Lower Low relative to l_ref
            i_l_down = next((j for j in range(2, t + 1) if df['Low'].iloc[t-j] < l_ref), 2)
            l_down = df['Low'].iloc[t - i_l_down]
        except StopIteration:
            # Fallback if no prior extremes are found
            continue

        # 2. Normalization Factors (using distances relative to t-1)
        bt_factor = max(i_h_up-1, i_h_down-1, i_l_up-1, i_l_down-1)
        ct_factor = max(h_up - h_ref, h_ref - h_down, l_up - l_ref, l_ref - l_down)
        
        # 3. Normalized Vectors
        # Adjusted indices to be relative to the reference point (t-1)
        b = [(i_h_up-1)/bt_factor, (i_h_down-1)/bt_factor, (i_l_up-1)/bt_factor, (i_l_down-1)/bt_factor]
        
        c = [
            (h_up - h_ref) / (ct_factor + epsilon),
            (h_ref - h_down) / (ct_factor + epsilon),
            (l_up - l_ref) / (ct_factor + epsilon),
            (l_ref - l_down) / (ct_factor + epsilon)
        ]
        
        # 4. Calculate Theta Angles
        for k in range(4):
            theta[f'ϴ{k+1}'][t] = np.arctan(b[k] / (c[k] + epsilon))
            
    # Add to dataframe
    for col, values in theta.items():
        df[col] = values
        
    df.dropna(inplace=True)
    return df

def add_wavelets(historical_data, k):
    """
    REVISED: Implements strictly causal Wavelet Gain Control.
    Uses t-1 data for all volatility and momentum inputs to prevent lookahead bias.
    """
    df = historical_data
    epsilon = 1e-9
    
    # --- 1. Strictly Lagging Volatility & Momentum ---
    # We use .shift(1) on all raw inputs to ensure we only know 
    # what happened UP TO the previous close.
    high_lag = df['High'].shift(1)
    low_lag = df['Low'].shift(1)
    close_lag = df['Close'].shift(1)
    prev_close_lag = df['Close'].shift(2)
    
    # Calculate ATR% based on lagging data
    tr_lag = pd.concat([
        high_lag - low_lag, 
        (high_lag - prev_close_lag).abs(), 
        (low_lag - prev_close_lag).abs()
    ], axis=1).max(axis=1)
    
    atr_lag = tr_lag.rolling(window=k).mean()
    # ATR% known at the start of bar t
    atr_pct_lag = (atr_lag / close_lag).replace(0, np.nan)
    
    # Delta C: Bounded momentum of the PREVIOUS bar
    denom_lag = close_lag.abs() + prev_close_lag.abs()
    delta_c_lag = (close_lag - prev_close_lag) / denom_lag.replace(0, epsilon)

    # --- 2. Interference & Amplitude (Physics Engine) ---
    # Theta angles are already shifted to t-1 in add_price_time_angles
    t1, t2, t3, t4 = df['ϴ1'], df['ϴ2'], df['ϴ3'], df['ϴ4']
    
    interference = (np.cos(t1) + np.sin(t1)) + (np.cos(t2) + np.sin(t2)) + \
                   (np.cos(t3) + np.sin(t3)) + (np.cos(t4) + np.sin(t4))
    direction_sign = np.sign(interference)
    
    # Sigma calculation: uses the ATR% known at the start of the bar
    e_terms = [(4 * (np.cos(tx) + np.sin(tx)))**2 for tx in [t1, t2, t3, t4]]
    A = np.maximum.reduce(e_terms)
    sigma_lag = (A / 32.0) * atr_pct_lag

    # --- 3. Baseline Sensitivity Beta_0 (Period: 3k) ---
    # SNR calculation now uses the lagging momentum and lagging sigma
    raw_snr_lag = (delta_c_lag.abs() / (sigma_lag + epsilon))
    snr_t_lag = raw_snr_lag.rolling(window=3*k).median()
    
    beta_0 = (1.0 / (snr_t_lag + epsilon)).clip(0.8, 1.5)

    # --- 4. Relative Volatility Scaling (Period: 2k) ---
    baseline_vol_lag = atr_pct_lag.rolling(window=2*k).mean()
    rel_vol_lag = (atr_pct_lag / (baseline_vol_lag + epsilon)).fillna(1.0)
    
    # --- 5. Final Beta & Wavelet State ---
    beta_final = (beta_0 / rel_vol_lag).clip(0.5, 2.5)

    # W(t) calculation is now purely a function of T-1 and T-2 data
    argument = beta_final * (delta_c_lag / (sigma_lag + epsilon))
    df['W'] = np.tanh(argument) * direction_sign
    
    # Wd calculation (Serial Difference of the causal signal)
    df['Wd'] = (df['W'] - df['W'].shift(1)) / 2.0
    
    return df

def add_bar_inbalance(historical_data):
    """
    Price-Neutral Bar Inbalance logic.
    Addresses missed breakouts in ABT/NSC by using price velocity (delta) 
    instead of price level (c).
    """
    df = historical_data
    prices = df['Close'].values
    n = len(prices)
    # --- 1. Calculate Balance Rule b(t) ---
    b = np.zeros(n)
    b[0] = np.sign(prices[0]) if prices[0] != 0 else 1.0
    for t in range(1, n):
        diff = prices[t] - prices[t-1]
        b[t] = np.sign(diff) if diff != 0 else b[t-1]
            
    # --- 2. Calculate Time Inbalance I(t) ---
    df['I'] = np.cumsum(b)
    
    # Utility: Serial Difference Δ
    def get_serial_diff(series):
        curr = series
        prev = series.shift(1)
        return (curr - prev) / (np.abs(curr) + np.abs(prev) + 1e-9)

    # Utility: Logarithmic filter ρ
    def rho(x):
        return (2.0 / np.log(2.0)) * (np.log(1.0 + x) / (1.0 + x))
    
    # Utility: Serial Bounded Ratio δ
    def get_delta(series):
        eps = 9**-5
        s_prev = series.shift(1).replace(0, eps)
        ratio = series / s_prev
        ratio = np.maximum(ratio, eps)
        return rho(ratio)

    # --- 3. Calculate Bar Inbalance Momentum Im(t) ---
    # Price Conviction δ(c(t-1), 1) available at start of bar t
    price_conviction = get_delta(df['Close']).shift(1)
    
    # Structural Change Δ(I(t-1), 1) available at start of bar t
    structural_change = get_serial_diff(df['I']).shift(1)
    
    # Momentum is the conviction-weighted structural shift
    df['Im'] = price_conviction * structural_change
    
    # --- 4. Calculate Bar Inbalance Difference Id(t) ---
    # Id(τ) = [Im(τ-1) - Im(τ-2)] / 2
    im_prev1 = df['Im'].shift(1)
    im_prev2 = df['Im'].shift(2)
    
    df['Id'] = (im_prev1 - im_prev2) / 2
    
    # Cleanup to prevent NaN errors in Neural Network
    df[['Im', 'Id']] = df[['Im', 'Id']].fillna(0.0)
    df.dropna(inplace=True)
    return df

def add_probability_differences(historical_data):
    """
    Calculates the Probability Difference (Pd) between upward and downward pressures.
    
    Formula: Pd(t) = P↑(t) - P↓(t)
    
    Args:
        historical_data (pd.DataFrame): Dataframe containing 'P↑' and 'P↓'.
        
    Returns:
        pd.DataFrame: Dataframe with the 'Pd' column added.
    """
    df = historical_data
    
    # Calculate the net difference
    # P↑: Likelihood of upward move
    # P↓: Likelihood of downward move
    df['Pd'] = df['P↑'] - df['P↓']
    df.dropna(inplace=True)
    return df


def add_price_volume_differences(historical_data):
    """
    Calculates the Price-Volume Difference (Yd) using the oscillator Y.
    
    Formula: Yd(t) = Δ(Y(t)) = Bounded percentage difference of Y at k=1.
    
    Args:
        historical_data (pd.DataFrame): Dataframe containing 'Y'.
        
    Returns:
        pd.DataFrame: Dataframe with added 'Yd' column.
    """
    df = historical_data
    
    # Extract current and previous values of the oscillator Y
    y = df['Y'].values
    y_prev = df['Y'].shift(1).values
    
    # Calculate Bounded Percentage Difference: Δ%(a, b) = (b-a) / (|a| + |b|)
    # This represents the Serial Difference Δ(Y(t), 1)
    # Epsilon added to denominator to ensure stability
    numerator = y - y_prev
    denominator = np.abs(y) + np.abs(y_prev) + 1e-9
    
    df['Yd'] = numerator / denominator
    
    # Fill the initial NaN resulting from the lookback
    df.dropna(inplace=True)
    
    return df

def add_quantum_lambda(ticker, historical_data, lookback_periods):
    """
    Calculates the quantum lambda (λ) using an fixed window to prevent data leakage.
    Uses the logarithmic filter of the squared serial difference of the close price.
    """
    df = historical_data
    n = len(df)
    lambdas = np.full(n, np.nan)
    
    # 1. Calculate filtered returns: rho(Delta^2(Close, 1))
    c = df['Close'].values
    c_prev = df['Close'].shift(1).values
    return_p = np.log(c / c_prev)
    
    # 2. Fixed window calculation (using data from T up to t-1 for bar t)
    T = 1
    for t in range(lookback_periods, n):      
        lambdas[t] = quantum_lambda(return_p[T:t])# [T:t]
        T += 1
        
    df['λ'] = lambdas   
    df.dropna(inplace=True)        
    return df

def add_boundary_energy_levels(df, market_type, window): # Fixed Critical Leakage, Jan 8 2026
    """
    STRICTLY CAUSAL Rolling Energy Levels.
    Ensures that energy walls for bar t are determined ONLY by data up to t-1.
    """
    # Use .shift(1) to ensure the rolling window EXCLUDES the current bar
    df['Rolling_Max'] = df['Close'].shift(1).rolling(window=window).max()
    df['Rolling_Min'] = df['Close'].shift(1).rolling(window=window).min()

    # Calculate energy levels row-by-row using the shifted extremes
    # This makes E_High and E_Low known at the Open of bar t.
    df['E_Low'] = df.apply(lambda x: maximum_energy_level(x['Rolling_Max'], x['λ'], market_type), axis=1)
    df['E_High'] = df.apply(lambda x: minimum_energy_level(x['Rolling_Min'], x['λ']), axis=1)

    df.drop(columns=['Rolling_Max', 'Rolling_Min'], inplace=True)
    df.dropna(inplace=True)
    return df

def add_scrodinger_gauge(historical_data):
    """
    Calculates the Schrödinger Gauge (Ö) based on the current price 
    and the quantum boundary energy levels.
    
    Formula: $Ö(t) = \log (C(t) /\sqrt(E_{low}(t) * E_{high}(t)$
    
    Args:
        historical_data (pd.DataFrame): Dataframe containing 'Close', 
                                       'E_Low', and 'E_High'.
        
    Returns:
        pd.DataFrame: Dataframe with the added 'Ö' column.
    """
    df = historical_data
    
    # Extract components
    c = df['Close'].values
    e_low = df['E_Low'].values
    e_high = df['E_High'].values
    
    e_equilibrium = np.sqrt(e_low * e_high)
    df['Ö'] = np.log(c / e_equilibrium)
    df.dropna(inplace=True)
    
    return df

def add_scrodinger_gauge_differences(historical_data):
    """
    Calculates the Schrödinger Gauge Difference (Öd).
    
    Formula: Öd(t) = [Ö(t) - Ö(t-1)] / 2
    
    Args:
        historical_data (pd.DataFrame): Dataframe containing the 'Ö' column.
        
    Returns:
        pd.DataFrame: Dataframe with the added 'Öd' column.
    """
    df = historical_data
    
    # Extract current and previous gauge values
    o_curr = df['Ö'].values
    o_prev = df['Ö'].shift(1).values
    
    # Calculate the difference normalized by 2.
    # Since Ö is bounded [-1, 1], the maximum possible difference is 2.
    # Dividing by 2 ensures Öd is also bounded within [-1, 1].
    df['Öd'] = (o_curr - o_prev) / 2.0
    
    # Fill the initial NaN resulting from the shift
    df.dropna(inplace=True)
    return df

def add_scrodinger_gauge_acceleration(historical_data):
    # Acceleration is the difference of the difference
    df = historical_data
    df['Ödd'] = (df['Öd'] - df['Öd'].shift(1)) / 2
    df.dropna(inplace=True)
    return df


import numpy as np
import pandas as pd
from qf.nn import fracdiff as frac
import numpy as np
import pandas as pd
from nn import fracdiff as frac

def add_diff_time_series(historical_data, k=14):
    """
    Implements Differentiated Time Series features using Log-Returns.
    
    1. Calculates Log-Returns L(t).
    2. Estimates fractional d and binomial weights using OLS on L(t).
    3. Calculates Ds as the dot product of weights and past log-returns.
    
    This ensures zero lookahead bias: Ds[t] is known at the Open of bar t.
    """
    df = historical_data
    c = df['Close'].values.flatten()
    n = len(df)
    
    # 1. Generate Log-Returns: L(t) = ln(C_t / C_{t-1})
    # We use a small epsilon to prevent log(0) errors if price is zero
    log_returns = np.diff(np.log(c + 1e-9))
    
    # Initialize output columns
    weight_cols = [f'W_{j}' for j in range(k)]
    for col in weight_cols:
        df[col] = 0.0
    df['Ds'] = 0.0

    # 2. Walk-forward estimation
    # log_returns[0] corresponds to the move ending at index 1
    # We start at k+1 to ensure enough history for the first OLS fit
    for i in range(k + 1, n):
        # CAUSAL STEP: Use log-returns known BEFORE bar i (indices up to i-2)
        # to estimate weights for the forecast of bar i.
        history_for_d = log_returns[:i-1]
        target_for_d = log_returns[i-1]
        
        # Estimate d and get binomial weights
        d_hat = frac.perform_ols_and_fit(history_for_d, target_for_d, k)
        weights = frac.get_binomial_weights(d_hat, k)
        
        # Assign weights to the dataframe for row i
        df.loc[df.index[i], weight_cols] = weights
        
        # 3. Calculate Ds (Differentiated Series)
        # We apply weights to past log-returns: [L_{i-1}, L_{i-2}, ... L_{i-k}]
        # This makes Ds[i] the "Forecasted Log-Return" for the current bar i.
        log_return_window = log_returns[i - k : i][::-1] 
        df.loc[df.index[i], 'Ds'] = np.dot(weights, log_return_window)

    df.dropna(inplace=True)
    return df