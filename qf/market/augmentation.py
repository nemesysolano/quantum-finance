import numpy as np
import pandas as pd

from qf.quantum import quantum_lambda
from qf.quantum import maximum_energy_level, minimum_energy_level
from qf.stats import empirical_distribution

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

def add_directional_probabilities(historical_data):
    """
    Calculates directional probabilities P↑(t) and P↓(t) based on structural breaches.
    
    Logic:
    - Resistance Breach (Gd = 1): Current gap exerts downward pressure.
    - Support Breach (Gd = -1): Current gap exerts upward pressure.
    
    Args:
        historical_data (pd.DataFrame): Dataframe containing 'S', 'Gd', and 'Close'.
        
    Returns:
        pd.DataFrame: Dataframe with 'P↑' and 'P↓' columns added.
    """
    df = historical_data
    n = len(df)
    
    # Initialize output arrays
    p_up = np.zeros(n)
    p_down = np.zeros(n)
    
    # 1. Calculate Bounded Percentage Difference for Close prices: Δc(t)
    # Formula: (c(t) - c(t-1)) / (|c(t)| + |c(t-1)|)
    c = df['Close'].values
    c_prev = df['Close'].shift(1).values
    
    delta_c = (c - c_prev) / (np.abs(c) + np.abs(c_prev))
    
    # 2. Calculate Squared Serial Difference: Δ²c(t)
    delta_c_sq = np.square(delta_c)
    
    for t in range(1, n):
        st = df['S'].iloc[t]
        gd = df['Gd'].iloc[t]
        sq_diff = delta_c_sq[t]
        
        # Support Breach Case (Ascending Trend Violation)
        if gd == -1:
            p_up[t] = min(1.0, st * sq_diff)
            p_down[t] = 1.0 - st
            
        # Resistance Breach Case (Descending Trend Violation)
        elif gd == 1:
            p_down[t] = min(1.0, st * sq_diff)
            p_up[t] = 1.0 - st
            
        # Default case (no breach history)
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
    
    # Extract values
    p = df['Close'].values
    v = df['Volume'].values
    
    # Calculate shifts (t-1)
    p_prev = df['Close'].shift(1).values
    v_prev = df['Volume'].shift(1).values
    
    # 1. Calculate Δp(t): Bounded Percentage Difference of Price
    # Formula: (p(t) - p(t-1)) / (|p(t)| + |p(t-1)|)
    delta_p = (p - p_prev) / (np.abs(p) + np.abs(p_prev))
    
    # 2. Calculate Δv(t): Bounded Percentage Difference of Volume
    delta_v = (v - v_prev) / (np.abs(v) + np.abs(v_prev))
    
    # 3. Calculate Δ²v(t): Squared Serial Difference of Volume
    delta_v_sq = np.square(delta_v)
    
    # 4. Calculate Y(t)
    df['Y'] = delta_p * delta_v_sq
    
    # Fill NaN at index 0 (result of shift) with 0.0
    
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

def add_bar_inbalance_ratio_and_difference(historical_data, k=14):
    """
    Calculates Bar Inbalance Ratio (Br) and Bar Inbalance Difference (Bd).
    
    Args:
        historical_data (pd.DataFrame): Dataframe containing 'Close'.
        k (int): Lookback window for the time inbalance B(t). Default is 14.
        
    Returns:
        pd.DataFrame: Dataframe with 'Br' and 'Bd' columns added.
    """
    df = historical_data
    n = len(df)
    
    # 1. Calculate the balance sequence b(t)
    # b(t) = b(t-1) if Δp(t) == 0, else sgn(Δp(t))
    prices = df['Close'].values
    delta_p = np.diff(prices, prepend=prices[0])
    
    b = np.zeros(n)
    # Initial value b(0) is sgn(p(0))
    b[0] = 1 if prices[0] >= 0 else -1
    
    for t in range(1, n):
        if delta_p[t] == 0:
            b[t] = b[t-1]
        else:
            # sgn(Δp(t)) = |Δp(t)| / Δp(t)
            b[t] = np.sign(delta_p[t])
            
    # 2. Calculate Time Inbalance B(t)
    # B(t) = (sum of b(t-i) for i=1 to k) / k
    # We use a rolling mean shifted by 1 to represent the sum of previous k elements
    b_series = pd.Series(b)
    B_t = b_series.rolling(window=k).mean().shift(1).fillna(0).values
    
    # 3. Calculate Bar Inbalance Ratio Br(t)
    # Br(t) = b(t) / (1 + B(t)^2)
    br = b / (1 + np.square(B_t))
    
    # 4. Calculate Bar Inbalance Difference Bd(t)
    # Bd(t) = ΔB(t) = Bounded percentage difference of B(t) (k=1)
    # Note: Using the utility function Δ% defined in the README
    B_t_prev = pd.Series(B_t).shift(1).values
    bd = (B_t - B_t_prev) / (np.abs(B_t) + np.abs(B_t_prev) + 1e-9)
    
    df['Br'] = br
    df['Bd'] = bd # Handle division by zero/NaN
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
    df = historical_data.copy()
    
    # Calculate the net difference
    # P↑: Likelihood of upward move
    # P↓: Likelihood of downward move
    df['Pd'] = df['P↑'] - df['P↓']
    
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

def add_inbalance_agression_filter(historical_data):
    """
    Calculates the Inbalance Aggression Filter (B+).
    
    Formula: B+(t) = Br(t) * |Bd(t)|
    Where:
        Br(t) = Bar Inbalance Ratio
        Bd(t) = Bar Inbalance Difference
        
    Args:
        historical_data (pd.DataFrame): Dataframe containing 'Br' and 'Bd'.
        
    Returns:
        pd.DataFrame: Dataframe with added 'B+' column.
    """
    df = historical_data.copy()
    
    # Extract previously calculated components
    # Br(t): Current directional inbalance ratio [-1, 1]
    # Bd(t): Velocity of time-inbalance change [-1, 1]
    br = df['Br'].values
    bd = df['Bd'].values
    
    # Calculate B+: Captures aggressive directional shifts
    # Using the absolute value of Bd ensures the sign of B+ 
    # always matches the direction of the imbalance (Br).
    df['B+'] = br * np.abs(bd)
    
    # Handle potential NaNs from early lookbacks
    df.dropna(inplace=True)
    
    return df

def add_boundary_energy_levels(historical_data: pd.DataFrame, quantization_level: int = 1e2):
    """
    Calculates and adds the quantum boundary energy levels (E_low, E_high)
    based on the empirical distribution of daily close price ratios (simple returns).
    """
    # Price Ratio (Simple Return) = P(t) / P(t-1)
    daily_returns = (historical_data['Close'] / historical_data['Close'].shift(1)).dropna()
    empirical_dist = empirical_distribution(daily_returns, quantization_level)
    λ = quantum_lambda(empirical_dist['X'], empirical_dist['P'])
    print(f"Computed quantum lambda: {λ}")
    lower_boundaries = np.vectorize(lambda x: maximum_energy_level(x, λ))
    upper_boundaries = np.vectorize(lambda x: minimum_energy_level(x, λ))
    historical_data['E_Low'] = lower_boundaries(historical_data['Low'])
    historical_data['E_High'] = upper_boundaries(historical_data['High'])
    historical_data.dropna(inplace=True)

import numpy as np
import pandas as pd

def add_scrodinger_gauge(historical_data):
    """
    Calculates the Schrödinger Gauge (Ö) based on the current price 
    and the quantum boundary energy levels.
    
    Formula: Ö(t) = [2*C(t) - (E_high + E_low)] / (E_high - E_low)
    
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
    
    # Calculate the Schrödinger Gauge
    # This centers the price between the boundaries:
    # Ö = 1.0  at the Upper Boundary (E_High)
    # Ö = -1.0 at the Lower Boundary (E_Low)
    # Ö = 0.0  at the equilibrium point
    numerator = (2 * c) - (e_high + e_low)
    denominator = e_high - e_low + 1e-9 # Epsilon for stability
    
    df['Ö'] = numerator / denominator
    
    # Ensure any division by zero or NaN is handled
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