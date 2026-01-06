import numpy as np
import pandas as pd


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

def add_swing_ratio(historical_data):
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
        gt_abs = abs(df['G'].iloc[t])
        at = max(gt_abs, rt)
        
        # 3. Calculate Swing Ratio S(t)
        if at > 0:
            s_values[t] = gt_abs / at
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
    
    Args:
        historical_data (pd.DataFrame): Dataframe containing 'High' and 'Low'.
        epsilon (float): Small value to prevent division by zero.
        
    Returns:
        pd.DataFrame: Dataframe with 'ϴ1', 'ϴ2', 'ϴ3', 'ϴ4' columns added.
    """
    df = historical_data
    n = len(df)
    
    # Initialize result arrays
    theta = {f'ϴ{k}': np.zeros(n) for k in range(1, 5)}
    
    # Iterate through the data to find structural pivots
    # We start from index 1 because we look backward (t-j)
    for t in range(1, n):
        h_t = df['High'].iloc[t]
        l_t = df['Low'].iloc[t]
        
        # 1. Find Closest Extremes and their indices (i)
        # Closest Higher High (h_up): h(t-j) > h(t)
        i_h_up = next((j for j in range(1, t + 1) if df['High'].iloc[t-j] > h_t), 1)
        h_up = df['High'].iloc[t - i_h_up]
        
        # Closest Lower High (h_down): h(t-j) < h(t)
        i_h_down = next((j for j in range(1, t + 1) if df['High'].iloc[t-j] < h_t), 1)
        h_down = df['High'].iloc[t - i_h_down]
        
        # Closest Higher Low (l_up): l(t-j) > l(t)
        i_l_up = next((j for j in range(1, t + 1) if df['Low'].iloc[t-j] > l_t), 1)
        l_up = df['Low'].iloc[t - i_l_up]
        
        # Closest Lower Low (l_down): l(t-j) < l(t)
        i_l_down = next((j for j in range(1, t + 1) if df['Low'].iloc[t-j] < l_t), 1)
        l_down = df['Low'].iloc[t - i_l_down]

        # 2. Normalization Factors
        # Time Lookback Base B(t)
        bt_factor = max(i_h_up, i_h_down, i_l_up, i_l_down)
        
        # Price Range Base C(t)
        ct_factor = max(h_up - h_t, h_t - h_down, l_up - l_t, l_t - l_down)
        
        # 3. Normalized Vectors
        # Time vector b(t)
        b = [i_h_up/bt_factor, i_h_down/bt_factor, i_l_up/bt_factor, i_l_down/bt_factor]
        
        # Price vector c(t)
        c = [
            (h_up - h_t) / (ct_factor + epsilon),
            (h_t - h_down) / (ct_factor + epsilon),
            (l_up - l_t) / (ct_factor + epsilon),
            (l_t - l_down) / (ct_factor + epsilon)
        ]
        
        # 4. Calculate Theta Angles: arctan(b_k / (c_k + epsilon))
        for k in range(4):
            theta[f'ϴ{k+1}'][t] = np.arctan(b[k] / (c[k] + epsilon))
            
    # Add to dataframe
    for col, values in theta.items():
        df[col] = values
        
    df.dropna(inplace=True)
    return df

import numpy as np
import pandas as pd

def add_wavelets(historical_data):
    """
    Implements the Wavelet function W(t) using the sgn(Δc) adjustment.
    
    Formula: 
    W(t) = sgn(Δc(t-1)) * S(t-1) * (sum(cos(θi) + sin(θi)))^2 / A
    Where:
        A = max_{i=1...4} {(4 * (cos(θi) + sin(θi)))^2}
        θi are the price-time angles at t-1.
        
    Args:
        historical_data (pd.DataFrame): Must contain 'Close', 'S', and 'ϴ1'...'ϴ4'.
        
    Returns:
        pd.DataFrame: Dataframe with 'W' column added.
    """
    df = historical_data
    n = len(df)
    w_values = np.zeros(n)
    
    # 1. Calculate Δc(t): Bounded Percentage Difference of Close
    # We only need the sign of this difference for the wavelet formula
    c = df['Close'].values
    c_prev = df['Close'].shift(1).values
    delta_c = (c - c_prev) / (np.abs(c) + np.abs(c_prev) + 1e-9)
    sgn_delta_c = np.sign(delta_c)
    
    # 2. Iterate to calculate W(t) using t-1 values
    for t in range(1, n):
        # Retrieve the state of the system at the close of the previous bar (t-1)
        sgn_dc_prev = sgn_delta_c[t-1]
        s_prev = df['S'].iloc[t-1]
        
        # Collect angles θ1...θ4 at t-1
        angles = [df[f'ϴ{i}'].iloc[t-1] for i in range(1, 5)]
        
        # Calculate periodic components: (cos(θ) + sin(θ))
        trig_terms = [np.cos(theta) + np.sin(theta) for theta in angles]
        
        # 3. Calculate the squared sum (Numerator)
        # This represents the total constructive/destructive geometric interference
        numerator = np.square(sum(trig_terms))
        
        # 4. Calculate normalization factor A
        # A = max over i of (4 * (cos(θi) + sin(θi)))^2
        # This ensures the geometric ratio is bounded by 1.0
        a_candidates = [np.square(4 * val) for val in trig_terms]
        A = max(a_candidates)
        
        # 5. Final Wavelet Calculation W(t)
        if A > 0:
            w_values[t] = sgn_dc_prev * s_prev * (numerator / A)
        else:
            w_values[t] = 0.0
            
    df['W'] = w_values
    df.dropna(inplace=True)
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

def add_wavelet_differences(historical_data):
    """
    Calculates the Wavelet Difference (Wd).
    
    Formula: Wd(t) = Δ(W(t)) = Bounded percentage difference of W at k=1.
    
    Args:
        historical_data (pd.DataFrame): Dataframe containing the 'W' column.
        
    Returns:
        pd.DataFrame: Dataframe with added 'Wd' column.
    """
    df = historical_data
    
    # Extract current and previous values of the Wavelet W
    w = df['W'].values
    w_prev = df['W'].shift(1).values
    
    # Calculate Bounded Percentage Difference: Δ%(a, b) = (b - a) / (|a| + |b|)
    # This represents the Serial Difference Δ(W(t), 1)
    # Small epsilon added to avoid division by zero
    numerator = w - w_prev
    denominator = np.abs(w) + np.abs(w_prev) + 1e-9
    
    df['Wd'] = numerator / denominator
    
    # Fill the initial NaN resulting from the shift
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

def add_boundary_energy_levels(df, market_type, window):
    """
    Causal 14-day Rolling Energy Levels.
    Prevents the 'Physics' from knowing future price extremes.
    """
    # Use rolling window for local max/min
    df['Rolling_Max'] = df['Close'].rolling(window=window).max()
    df['Rolling_Min'] = df['Close'].rolling(window=window).min()

    # Calculate energy levels row-by-row using ONLY previous 14 days
    df['E_High'] = df.apply(lambda x: maximum_energy_level(x['Rolling_Max'], x['λ'], market_type), axis=1)
    df['E_Low'] = df.apply(lambda x: minimum_energy_level(x['Rolling_Min'], x['λ']), axis=1)

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