import numpy as np
import pandas as pd

# Ensure this import exists in your project structure
from qf.quantum import  maximum_energy_level, minimum_energy_level
from qf.stats import empirical_distribution
from qf.quantum import quantum_lambda
from qf.context import default_quantization_level


def add_breaking_gap(data: pd.DataFrame, quantization_delta: float) -> pd.DataFrame:
    """
    Calculates the Breaking Gap (G(t)) based on Fast Trend structural violations.
    
    SIGN CONVENTION:
    - Positive G (+): Resistance Breach (Upward Break).
    - Negative G (-): Support Breach (Downward Break).
    
    Applies linear decay to the magnitude.
    """
    G = pd.Series(0.0, index=data.index)
    G_b = 0.0  # Last Structural Breach Value (Signed)
    k = 0      # Bars elapsed

    for t in range(3, len(data)):
        current_time = data.index[t]
        
        l_t, h_t = data['Low'].iloc[t], data['High'].iloc[t]
        l_tm1, h_tm1 = data['Low'].iloc[t-1], data['High'].iloc[t-1]
        l_tm2, h_tm2 = data['Low'].iloc[t-2], data['High'].iloc[t-2]
        c_tm2, c_tm3 = data['Close'].iloc[t-2], data['Close'].iloc[t-3]
        
        new_breach = False
        breach_val = 0.0

        # Case 1: Support Breach (Ascending Trend Violation) -> NEGATIVE Sign
        if (c_tm3 < c_tm2) and (l_t < l_tm2) and (l_tm2 < l_tm1):
            new_breach = True
            # Magnitude is (l_tm2 - l_t). We apply negative sign.
            breach_val = -(l_tm2 - l_t)

        # Case 2: Resistance Breach (Descending Trend Violation) -> POSITIVE Sign
        elif (c_tm3 > c_tm2) and (h_t > h_tm2) and (h_tm2 > h_tm1):
            new_breach = True
            # Magnitude is (h_t - h_tm2). We apply positive sign.
            breach_val = (h_t - h_tm2)
        
        # --- Update State ---
        if new_breach:
            G_b = breach_val
            k = 0 
            G.loc[current_time] = G_b
        else:
            k += 1
            # Linear Decay on Magnitude, preserving Sign
            magnitude = max(abs(G_b) - k * quantization_delta, 0.0)
            # Restore sign
            G.loc[current_time] = magnitude if G_b >= 0 else -magnitude

    data['G'] = G
    data.dropna(inplace=True)
    return data

def add_swing_ratio(historical_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the Swing Ratio (S) based on the current decayed gap (G)
    and the local price volatility (4-bar range).
    
    Formula:
    R(t) = max(High[t...t-3]) - min(Low[t...t-3])
    A(t) = max(|G(t)|, R(t))
    S(t) = |G(t)| / A(t)
    
    Args:
        historical_data (pd.DataFrame): DataFrame containing 'High', 'Low', and 'G' columns.
        
    Returns:
        pd.DataFrame: Original DataFrame with 'S' column added.
    """
    df = historical_data
    
    # Ensure 'G' (Breaking Gap) exists
    if 'G' not in df.columns:
        raise ValueError("The 'G' column (Breaking Gap) must be added before calculating Swing Ratio.")

    # 1. Calculate Local Range (R) over a 4-bar window (t, t-1, t-2, t-3)
    # Using rolling windows to find max high and min low
    # The window=4 includes the current row and the 3 previous rows.
    local_high = df['High'].rolling(window=4).max()
    local_low = df['Low'].rolling(window=4).min()
    R = local_high - local_low

    # 2. Get the absolute magnitude of the current decayed gap
    abs_G = df['G'].abs()

    # 3. Calculate Absolute Reference (A) 
    # A is the element-wise maximum of the gap magnitude and the local range
    A = np.maximum(abs_G, R)

    # 4. Compute Swing Ratio S(t)
    # We use np.divide to handle cases where A might be zero (e.g., flat market with no gap)
    # If A is 0, the result is 0 (Neutral pressure).
    S = np.divide(
        abs_G,
        A,
        out=np.zeros_like(abs_G),
        where=A != 0
    )

    df['S'] = S
    df['A'] = A
    
    # Drop NaNs created by the rolling window (first 3 rows will be NaN)
    df.dropna(subset=['S'], inplace=True)
    
    return df
def add_directional_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Directional Probabilities (P↑, P↓) based on:
    1. The Swing Ratio (S(t))
    2. The last structural breach type (G_b)
    3. The price percentage change (r_p) using an exponential squashing function.
    """
    # 1. Calculate the percentage difference r_p(t) (referred to as Δ_%c(t) in README)
    # Formula: 2 * (c(t) - c(t-1)) / (c(t) + c(t-1))
    c_t = df['Close']
    c_prev = df['Close'].shift(1)
    rp = 2 * (c_t - c_prev) / (c_t + c_prev)
    
    # Initialize probability columns
    df['P↑'] = 0.5
    df['P↓'] = 0.5
    
    # We assume 'G' is a column identifying the breach:
    # 'resistance' for Descending Trend Violation
    # 'support' for Ascending Trend Violation
    # We assume 'S' is the pre-calculated Swing Ratio column
    
    # Resistance Breach Case: Downward pressure
    # P↓ = min(1, S(t) * exp(-rp(t)))
    # P↑ = 1 - P↓
    res_mask = df['G'] > 0
    df.loc[res_mask, 'P↓'] = np.minimum(1.0, df.loc[res_mask, 'S'] * np.exp(-rp[res_mask]))
    df.loc[res_mask, 'P↑'] = 1.0 - df.loc[res_mask, 'P↓']
    
    # Support Breach Case: Upward pressure
    # P↑ = min(1, S(t) * exp(rp(t)))
    # P↓ = 1 - P↑
    sup_mask = df['G'] < 0
    df.loc[sup_mask, 'P↑'] = np.minimum(1.0, df.loc[sup_mask, 'S'] * np.exp(rp[sup_mask]))
    df.loc[sup_mask, 'P↓'] = 1.0 - df.loc[sup_mask, 'P↑']
    
    # Handle the probability difference used in baseline models: Pd = P↑ - P↓
    df['Pd'] = df['P↑'] - df['P↓']
    
    return df

def add_price_volume_strength_oscillator(historical_data: pd.DataFrame, price: str) -> pd.DataFrame:
    """
    Calculates and adds the Price-Volume Strength Oscillator (Y) to the DataFrame.
    The new column is named 'Y_{price}'.
    
    Formula: Y(t) = [2(p(t) - p(t-1)) / (p(t) + p(t-1))] * [v(t) / v(t-1)]
    """
    df = historical_data
    
    # Ensure necessary columns exist
    if price not in df.columns or 'Volume' not in df.columns:
        raise ValueError(f"Columns '{price}' and 'Volume' must exist in the DataFrame.")

    # 1. Get current and previous price and volume series
    p_t = df[price]
    p_prev = p_t.shift(1)
    
    v_t = df['Volume']
    v_prev = v_t.shift(1)

    # 2. Calculate Price Term: 2 * (p(t) - p(t-1)) / (p(t) + p(t-1))
    # We use np.divide to handle cases where (p(t) + p(t-1)) might be 0 (unlikely but possible)
    price_sum = p_t + p_prev
    price_diff = p_t - p_prev
    
    price_term = np.divide(
        2 * price_diff,
        price_sum,
        out=np.zeros_like(p_t, dtype=float),
        where=price_sum != 0
    )

    # 3. Calculate Volume Term: v(t) / v(t-1)
    # Handle division by zero if v(t-1) is 0
    volume_term = np.divide(
        v_t,
        v_prev,
        out=np.zeros_like(v_t, dtype=float),
        where=v_prev != 0
    )

    # 4. Compute Oscillator Y(t)
    y_values = price_term * volume_term
    
    # 5. Assign to new column
    col_name = f"Y_{price}"
    df[col_name] = y_values
    
    # Remove the first row which will be NaN due to shifting
    # (Optional: depends on if you want to preserve the original shape)
    df.dropna(subset=[col_name], inplace=True)
    
    return df


def add_closest_higher_high(historical_data):
    highs = historical_data['High']
    num_rows = len(historical_data)
    result_val = np.full(num_rows, np.nan)
    result_days = np.full(num_rows, np.nan)

    for t in range(1, num_rows):
        current_high = highs.iloc[t]
        for i in range(1, t + 1):
            past_high = highs.iloc[t - i]
            if past_high > current_high:
                result_val[t] = past_high
                result_days[t] = i
                break
    historical_data['h_↑'] = result_val
    historical_data['Dh_↑'] = result_days
    historical_data.dropna(inplace=True)

def add_closest_lower_high(historical_data):
    highs = historical_data['High']
    num_rows = len(historical_data)
    result_val = np.full(num_rows, np.nan)
    result_days = np.full(num_rows, np.nan)

    for t in range(1, num_rows):
        current_high = highs.iloc[t]
        for i in range(1, t + 1):
            past_high = highs.iloc[t - i]
            if past_high < current_high:
                result_val[t] = past_high
                result_days[t] = i
                break
    historical_data['h_↓'] = result_val
    historical_data['Dh_↓'] = result_days
    historical_data.dropna(inplace=True)

def add_closest_higher_low(historical_data):
    lows = historical_data['Low']
    num_rows = len(historical_data)
    result_val = np.full(num_rows, np.nan)
    result_days = np.full(num_rows, np.nan)

    for t in range(1, num_rows):
        current_low = lows.iloc[t]
        for i in range(1, t + 1):
            past_low = lows.iloc[t - i]
            if past_low > current_low:
                result_val[t] = past_low
                result_days[t] = i
                break
    historical_data['l_↑'] = result_val
    historical_data['Dl_↑'] = result_days
    historical_data.dropna(inplace=True)

def add_closest_lower_low(historical_data):
    lows = historical_data['Low']
    num_rows = len(historical_data)
    result_val = np.full(num_rows, np.nan)
    result_days = np.full(num_rows, np.nan)
    for t in range(1, num_rows):
        current_low = lows.iloc[t]
        for i in range(1, t + 1):
            past_low = lows.iloc[t - i]
            if past_low < current_low:
                result_val[t] = past_low
                result_days[t] = i
                break
    historical_data['l_↓'] = result_val
    historical_data['Dl_↓'] = result_days
    historical_data.dropna(inplace=True)

def identify_pivots(df, window=5):
    df['is_pivot_high'] = df['High'].rolling(window=window*2+1, center=True).max() == df['High']
    df['is_pivot_low'] = df['Low'].rolling(window=window*2+1, center=True).min() == df['Low']
    return df

def get_nearest_structural_extreme(current_idx, current_val, df, pivot_col, price_col, comparison):    
    for past_idx in range(current_idx - 1, -1, -1):
        if df.iat[past_idx, df.columns.get_loc(pivot_col)]: # Check if it is a pivot
            past_val = df.iat[past_idx, df.columns.get_loc(price_col)]
            if comparison(past_val, current_val):
                return past_idx
    return -1

def add_cosine_and_sine_for_price_time_angles(df):
    """
    Calculates Price-Time angles based on Fractal Pivots and ATR Normalization.
    
    Implements:
    A. Fractal Pivot Search (Structural points instead of noise).
    B. ATR Normalization (Gann Box stabilization).
    C. Edge Case Handling (Breakout constants).
    """
    
    # 1. Ensure Dependencies
    # We need ATR for normalization. Assuming 'ATR' or 'Atrp14' exists. 
    # If using 'Atrp14' (percentage), we convert back to absolute ATR approx or calculate it.
    # Here we will calculate a standard 14-period ATR for safety.
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR_14'] = true_range.rolling(14).mean()

    # 2. Identify Structural Pivots (Suggestion A)
    # Using a 5-day window (2 days before, 2 days after)
    df = identify_pivots(df, window=2)

    # Initialize columns for angles (theta)
    # 1: Higher High, 2: Lower High, 3: Higher Low, 4: Lower Low
    thetas = {1: [], 2: [], 3: [], 4: []}
    
    # Comparisons for the 4 extremes
    # Higher High: Past High > Current High
    # Lower High: Past High < Current High
    # Higher Low: Past Low > Current Low
    # Lower Low: Past Low < Current Low
    comparisons = {
        1: ('is_pivot_high', 'High', lambda p, c: p > c),
        2: ('is_pivot_high', 'High', lambda p, c: p < c),
        3: ('is_pivot_low', 'Low', lambda p, c: p > c),
        4: ('is_pivot_low', 'Low', lambda p, c: p < c)
    }

    # Iterate through the DataFrame
    # Note: Iterating rows is slow in Pandas, but necessary for complex lookback logic 
    # that varies per row.
    
    for i in range(len(df)):
        if i < 20: # Skip beginning where ATR/Pivots might be unstable
            for k in thetas: thetas[k].append(0)
            continue
            
        atr = df.iat[i, df.columns.get_loc('ATR_14')]
        if pd.isna(atr) or atr == 0:
            atr = df.iat[i, df.columns.get_loc('Close')] * 0.01 # Fallback
            
        current_time_idx = i
        
        for k, (pivot_col, price_col, comp_func) in comparisons.items():
            current_price = df.iat[i, df.columns.get_loc(price_col)]
            
            # Find closest structural extreme
            past_idx = get_nearest_structural_extreme(
                current_time_idx, current_price, df, pivot_col, price_col, comp_func
            )
            
            if past_idx != -1:
                # Suggestion B: Stabilized Normalization
                # Slope = (Delta Price) / (Delta Time * ATR)
                # This normalizes the "speed" of the move relative to volatility.
                
                past_price = df.iat[past_idx, df.columns.get_loc(price_col)]
                delta_price = current_price - past_price
                delta_time = current_time_idx - past_idx # Number of bars
                
                # Gann Theory: 45 degrees (slope 1) is a "balanced" market.
                # If price moves 1 ATR in 1 day, slope is 1.
                normalized_slope = delta_price / (delta_time * atr)
                
                # Calculate Angle in Radians
                angle = np.arctan(normalized_slope)
                thetas[k].append(angle)
            else:
                # Suggestion C: Edge Case Handling (Breakout)
                # If we are making a New High (no Higher High found), 
                # we are in "blue sky" mode. 
                # Resistance is effectively infinite or vertical (90 deg / pi/2).
                if k == 1: # Higher High (Resistance) missing -> Bullish Breakout
                    thetas[k].append(np.pi / 2)
                elif k == 4: # Lower Low (Support) missing -> Bearish Breakdown
                    thetas[k].append(-np.pi / 2)
                else:
                    thetas[k].append(0)


    # 3. Add Cosine and Sine features
    for j in range(len(thetas)):
        k = j + 1
        df[f'cos_θ{k}'] = np.cos(thetas[k])
        df[f'sin_θ{k}'] = np.sin(thetas[k])


    # Clean up temporary columns
    df.drop(columns=['is_pivot_high', 'is_pivot_low', 'ATR_14'], inplace=True, errors='ignore')
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

def add_wavelets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the Wavelet W(t) based on README definitions.
    
    W(t) = Δ%c(t-1) * S(t-1) * sum(cos(θ) + sin(θ))
    """
    # 1. Calculate Signed Momentum: Δ%c(t)
    c = df['Close']
    rp = 2 * (c - c.shift(1)) / (c + c.shift(1)) #
    
    # 2. Sum the Geometric Components (Trig Sum)
    trig_sum = 0
    for i in range(1, 5):
        trig_sum += df[f'cos_θ{i}'] + df[f'sin_θ{i}'] #
        
    # 3. Compute W(t) using lagged values (t-1)
    # rp.shift(1) is the only component that dictates the initial sign.
    df['W'] = rp.shift(1) * df['S'].shift(1) * trig_sum.shift(1) #
    
    df.dropna(subset=['W'], inplace=True)
    return df

def add_wavelets_differences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the Wavelet Difference Wd(t) = W(t) - W(t-1).
    This is the target for the Wavelet Difference Forecast model.
    """
    if 'W' not in df.columns:
        df = add_wavelets(df)
        
    df['Wd'] = df['W'] - df['W'].shift(1) #
    df.dropna(subset=['Wd'], inplace=True)
    return df