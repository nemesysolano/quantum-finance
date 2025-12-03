import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

from qf.stats.normalizers import quantize

def add_structural_direction(historical_data):
    high_prices = historical_data['High']
    low_prices = historical_data['Low']
    num_rows = len(historical_data)
    structural_direction = np.full(num_rows, np.nan)

    for i in range(1, num_rows):
        h_t = high_prices.iloc[i]
        h_t_minus_1 = high_prices.iloc[i-1]
        l_t = low_prices.iloc[i]
        l_t_minus_1 = low_prices.iloc[i-1]

        if h_t > h_t_minus_1 and l_t >= l_t_minus_1:
            structural_direction[i] = 1

        elif l_t < l_t_minus_1 and h_t <= h_t_minus_1:
            structural_direction[i] = -1

        else:
            structural_direction[i] = structural_direction[i-1]

    historical_data['structural_direction'] = structural_direction
    historical_data.dropna(inplace=True)
    
def add_slow_trend_run(historical_data):
    close_prices = historical_data['Close']
    structural_directions = historical_data['structural_direction']
    num_rows = len(historical_data)

    slow_trend_run = np.full(num_rows, np.nan)
    t_s = -1  # Start index of the slow trend

    for i in range(1, num_rows):
        sd_t = structural_directions.iloc[i]
        sd_t_minus_1 = structural_directions.iloc[i-1]

        # A new slow trend starts when the structural direction changes.
        # We also need to handle the initial NaN values.
        if sd_t != sd_t_minus_1 and not np.isnan(sd_t):
            t_s = i

        # Calculate slow trend run if t_s is set and we can access c_{t_s-1}
        if t_s > 0:
            slow_trend_run[i] = close_prices.iloc[i] - close_prices.iloc[t_s - 1]

    historical_data['slow_trend_run'] = slow_trend_run
    historical_data.dropna(inplace=True)

def add_breaking_gap(historical_data):
    # Assuming the slow trend direction is determined by the sign of the current slow_trend_run or structural_direction (Sd(t))
    # Since the slow_trend_run is R_s(t) = c(t) - c(t_s-1), its sign indicates the current trend.
    
    high_prices = historical_data['High']
    low_prices = historical_data['Low']
    slow_trend_run = historical_data['slow_trend_run']
    num_rows = len(historical_data)

    breaking_gap = np.full(num_rows, 0.0)  # Initialize all to 0.0 (The first one is zero)

    # Start loop at 2 because the logic relies on t-2
    for i in range(2, num_rows):
        
        # Determine the slow trend direction at t
        # A positive slow trend run (R_s(t) > 0) implies an ascending trend.
        # A negative slow trend run (R_s(t) < 0) implies a descending trend.
        # The direction is implicit in the slow_trend_run R_s(t)
        
        current_slow_trend_run = slow_trend_run.iloc[i]
        
        gap = 0.0 # Default value if violation occurs (will be overwritten)

        # --- Trend Violation for Slow Ascending Trends ---
        # If the slow trend is ascending (R_s(t) > 0)
        if current_slow_trend_run > 0:
            # Violation occurs when l(t) < l(t-2)
            if low_prices.iloc[i] < low_prices.iloc[i-2]:
                # G(t) = l(t-2) - l(t)
                gap = low_prices.iloc[i-2] - low_prices.iloc[i]
            else:
                # No violation: G(t) is the previous breaking gap G(t-1)
                gap = breaking_gap[i-1]

        # --- Trend Violation for Slow Descending Trends ---
        # If the slow trend is descending (R_s(t) < 0)
        elif current_slow_trend_run < 0:
            # Violation occurs when h(t) > h(t-2)
            if high_prices.iloc[i] > high_prices.iloc[i-2]:
                # G(t) = h(t) - h(t-2)
                gap = high_prices.iloc[i] - high_prices.iloc[i-2]
            else:
                # No violation: G(t) is the previous breaking gap G(t-1)
                gap = breaking_gap[i-1]
        
        # --- Undefined/Zero Trend Case ---
        # If R_s(t) = 0 (e.g., at the start of the series or if c(t) = c(t_s-1)),
        # we assume no current directional trend, so the gap carries over.
        else: 
             gap = breaking_gap[i-1]


        breaking_gap[i] = gap

    historical_data['breaking_gap'] = quantize(breaking_gap    )
    historical_data.dropna(inplace=True)