import numpy as np
import pandas as pd

# Ensure this import exists in your project structure
from qf.stats.normalizers import default_quantization_delta 

def add_structural_direction(historical_data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the structural direction column (S_d) to the DataFrame.
    """
    df = historical_data
    
    # Vectorized comparison for efficiency
    prev_high = df['High'].shift(1)
    prev_low = df['Low'].shift(1)
    
    conditions = [
        (df['High'] > prev_high) & (df['Low'] >= prev_low), # Ascending
        (df['Low'] < prev_low) & (df['High'] <= prev_high)  # Descending
    ]
    choices = [1, -1]
    
    # default=np.nan implies 'Continuation' initially
    df['Sd'] = np.select(conditions, choices, default=np.nan)
    
    # Fill NaNs with the previous valid value (Continuation logic)
    df['Sd'] = df['Sd'].ffill().fillna(0)
    
    return df

def add_slow_trend_run(historical_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the Slow Trend Run (Rs).
    Rs(t) = c(t) - c(t_s - 1)
    """
    # Avoid SettingWithCopy warnings
    df = historical_data
    close_prices = df['Close'].values
    structural_directions = df['Sd'].values
    num_rows = len(df)

    slow_trend_run = np.full(num_rows, np.nan)
    t_s = -1  # Start index of the slow trend

    # Iterating from 1 because we need i-1
    for i in range(1, num_rows):
        sd_t = structural_directions[i]
        sd_t_minus_1 = structural_directions[i-1]

        # A new slow trend starts when the structural direction changes.
        if sd_t != sd_t_minus_1:
            t_s = i

        # Calculate slow trend run if t_s is set validly
        # We need t_s > 0 to safely access close_prices[t_s - 1]
        if t_s > 0:
            slow_trend_run[i] = close_prices[i] - close_prices[t_s - 1]

    df['Rs'] = slow_trend_run
    df.dropna(inplace=True)
    return df

def add_fast_trend_run(historical_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and adds the Fast Trend Run (Rf) to the DataFrame.
    """
    df = historical_data
    close_prices = df['Close'].values
    num_rows = len(df)
    
    fast_trend_run = np.full(num_rows, np.nan)
    t_f = 0  # Start index of the fast trend

    # Logic:
    # If direction changes at index i (relative to i-1), the new run starts from i-1.
    
    if num_rows > 0:
        fast_trend_run[0] = 0 # Initialize first element

    for i in range(1, num_rows):
        # We need at least 3 points to compare two deltas (i vs i-1, and i-1 vs i-2)
        if i > 1:
            delta_current = close_prices[i] - close_prices[i-1]
            delta_prev = close_prices[i-1] - close_prices[i-2]
            
            # If signs differ, direction changed.
            # Using np.sign ensures we catch + to - or - to +
            if np.sign(delta_current) != np.sign(delta_prev):
                t_f = i - 1
        
        # Rf(t) = c(t) - c(t_f)
        fast_trend_run[i] = close_prices[i] - close_prices[t_f]

    df['Rf'] = fast_trend_run
    df.dropna(inplace=True)
    return df

def add_breaking_gap(historical_data: pd.DataFrame, quantization_delta: float = default_quantization_delta) -> pd.DataFrame:
    """
    Calculates and adds the Breaking Gap (G) to the DataFrame.
    Includes logic for linear decay based on the time elapsed since the last breach (k).
    """
    df = historical_data
    index = df.index
    num_rows = len(df)
    gaps = np.zeros(num_rows)
    
    slow_trend_direction = df['Sd'].values
    low = df['Low'].values
    high = df['High'].values
    
    last_breach_val = 0.0 # G_b
    k = 0 # Counter for decay bars

    # Start at 2 because we look back at t-2
    for idx in range(2, num_rows):
        # t is current, t-1 is prev, t-2 is 2 bars ago
        # Trend check uses t-1 (trend established before current bar)
        trend_dir = slow_trend_direction[idx-1]
        
        is_violation = False
        current_gap = 0.0

        # Ascending trend violation (Break Low)
        if trend_dir > 0 and low[idx] < low[idx-2]:
            current_gap = low[idx-2] - low[idx]
            is_violation = True

        # Descending trend violation (Break High)
        elif trend_dir < 0 and high[idx] > high[idx-2]:
            current_gap = high[idx] - high[idx-2]
            is_violation = True

        if is_violation:
            gaps[idx] = current_gap
            last_breach_val = current_gap # Set G_b
            k = 1 # Reset counter (k=1 for first bar after breach, effectively)
        else:
            # Decay Logic: G(t) = max(G_b - k * Q, 0)
            gaps[idx] = max(last_breach_val - (k * quantization_delta), 0.0)
            k += 1

    df['G'] = gaps    
    return df

def add_fast_swing_ratio(historical_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Fast Swing Ratio (Sf).
    Sf(t) = min(2, (G(t) / |Rf(t)|)^2)
    """
    df = historical_data
    breaking_gap = df['G']
    fast_trend_run_abs = df['Rf'].abs()

    # Vectorized calculation handling division by zero
    ratio_squared = np.divide(
        breaking_gap, 
        fast_trend_run_abs, 
        out=np.zeros_like(breaking_gap), 
        where=fast_trend_run_abs!=0
    ) ** 2
    
    df['Sf'] = np.minimum(2, ratio_squared)
    # Note: We usually don't dropna here unless gaps/Rf generated NaNs
    return df

def add_last_opposite(historical_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the last opposite slow trend run (R*).
    Optimized to O(N) complexity.
    """
    df = historical_data
    rs = df['Rs'].values
    num_rows = len(df)
    rs_star = np.full(num_rows, np.nan)

    last_positive_run = np.nan
    last_negative_run = np.nan

    for i in range(num_rows):
        current_val = rs[i]
        
        # If current run is Positive
        if current_val > 0:
            rs_star[i] = last_negative_run # The last opposite was negative
            last_positive_run = current_val # Update last seen positive
            
        # If current run is Negative
        elif current_val < 0:
            rs_star[i] = last_positive_run # The last opposite was positive
            last_negative_run = current_val # Update last seen negative
            
        # If current run is 0, we generally carry forward logic or do nothing
        # depending on specific needs. Here we leave as nan or previous.
        # Assuming 0 doesn't update "direction".
        else:
             # Fallback if needed, or leave NaN. 
             # Logic usually implies strict >0 or <0 for trend runs.
             pass

    df['R*'] = rs_star
    df.dropna(inplace=True)
    return df

def add_slow_swing_ratio(historical_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Slow Swing Ratio (Ss).
    Ss(t) = min(2, (|Rs(t)| / |R*_s(t)|)^2)
    """
    df = historical_data
    rs_abs = df['Rs'].abs()
    rs_star_abs = df['R*'].abs()

    # Vectorized calculation
    ratio_squared = np.divide(
        rs_abs, 
        rs_star_abs, 
        out=np.zeros_like(rs_abs), 
        where=rs_star_abs!=0
    ) ** 2
    
    df['Ss'] = np.minimum(2, ratio_squared)
    return df

def add_directional_probabilities(historical_data):
    """
    Adds directional probabilities (P_up and P_down) to the DataFrame.
    """
    df = historical_data
    p_up = pd.Series(np.nan, index=df.index)
    p_down = pd.Series(np.nan, index=df.index)
    
    sf = df['Sf']
    ss = df['Ss']
    rf = df['Rf']
    rs = df['Rs']

    # Conditions
    conflicting_fast_up_slow_down = (rf > 0) & (rs < 0)
    conflicting_fast_down_slow_up = (rf < 0) & (rs > 0)
    aligned_up = (rf > 0) & (rs > 0)
    aligned_down = (rf < 0) & (rs < 0)
    
    # --- Conflicting Trends ---
    # Formula: P_reversal = Sf / (Sf + Ss) if Sf > 0 else 0.5
    
    sum_swings = sf + ss
    conflicting_prob = pd.Series(0.5, index=df.index)
    
    # Calculate ratio where Sf > 0. Avoid division by zero if sum_swings is 0.
    valid_mask = (sf > 0) & (sum_swings > 0)
    conflicting_prob.loc[valid_mask] = sf.loc[valid_mask] / sum_swings.loc[valid_mask]
    
    # Fast Ascending vs Slow Descending -> P_down is the reversal probability
    p_down.loc[conflicting_fast_up_slow_down] = conflicting_prob.loc[conflicting_fast_up_slow_down]
    
    # Fast Descending vs Slow Ascending -> P_up is the reversal probability
    p_up.loc[conflicting_fast_down_slow_up] = conflicting_prob.loc[conflicting_fast_down_slow_up]

    # --- Aligned Trends ---
    # Formula: P_continuation = min(1, Ss/4 + Sf)
    
    aligned_prob = np.minimum(1, (ss / 4) + sf)
    
    # Both Ascending -> P_up is continuation
    p_up.loc[aligned_up] = aligned_prob.loc[aligned_up]
    
    # Both Descending -> P_down is continuation
    p_down.loc[aligned_down] = aligned_prob.loc[aligned_down]
    
    # --- Complementary Probabilities ---
    df['P↑'] = p_up.fillna(1 - p_down)
    df['P↓'] = p_down.fillna(1 - p_up)
    
    historical_data.dropna(inplace=True)
    return df

import numpy as np
import pandas as pd

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

def add_relative_volume(ticker, historical_data):
    market_cap = ticker.info.get('marketCap')
    historical_data['RV'] = historical_data['Volume'] / (market_cap / historical_data['Close'])    
    historical_data.dropna(inplace=True)        
