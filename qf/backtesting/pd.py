from qf.backtesting.util import Transaction, apply_integer_nudge, dynamic_slippage
import numpy as np

import numpy as np
import pandas as pd

def add_average_momentum(historical_data, k=14):
    """
    Calculates Average Momentum M(t) and its Signal Line Mσ (EMA).
    
    Args:
        historical_data (pd.DataFrame): Dataframe containing 'Close', 'P↑', and 'P↓'.
        k (int): Period for the Exponential Moving Average of M.
        
    Returns:
        pd.DataFrame: Dataframe with added 'M' and 'Mσ' columns.
    """
    df = historical_data
    
    # 1. Bounded Percentage Difference: Δ_%(c(t-1), c(t))
    c = df['Close'].values
    c_prev = df['Close'].shift(1).values
    delta_c = (c - c_prev) / (np.abs(c) + np.abs(c_prev) + 1e-12)
    
    # 2. Extract Probabilities
    p_up = df['P↑'].values
    p_down = df['P↓'].values
    
    # 3. Calculate Components: M = Δc * e^(P↑ - P↓) + Δc * e^(P↓ - P↑)
    # This simplifies to M = 2 * Δc * cosh(P↑ - P↓)
    diff_p = p_up - p_down
    m_t = delta_c * (np.exp(diff_p) + np.exp(-diff_p))
    
    df['M'] = m_t
    
    # 4. Calculate Signal Line Mσ using EMA
    # Mσ acts as a trend filter for the momentum itself
    df['Mσ'] = df['M'].ewm(span=k, adjust=False).mean()
    
    R = np.abs(df['M'] / (df['Mσ'] + 0.0001))

    df['R'] = R
    df.dropna(inplace=True)
    
    return df

from qf.backtesting.util import Transaction, apply_integer_nudge, dynamic_slippage
import numpy as np

def simulate_trading_pd(ticker, y_test, physics_test, reward=3, initial_cap=10000, k_window=14):    
    transaction_log = []    
    cash = initial_cap
    equity_curve = [initial_cap]
    longs, shorts, winner_longs, winner_shorts, loser_longs, loser_shorts = 0, 0, 0, 0, 0, 0
    
    y_actual = y_test.values
    o_d, o_dd = physics_test['Öd'].values, physics_test['Ödd'].values
    atr_values = physics_test['ATR'].values
    high_values, low_values = physics_test['High'].values, physics_test['Low'].values
    e_low, e_high = physics_test['E_Low'].values, physics_test['E_High'].values
    price_values = physics_test['Close'].values
    W = physics_test['W'].values 
    Id = physics_test['Id'].values
    M = physics_test['M'].values
    R = physics_test['R'].values
    Yd = physics_test['Yd'].values

    # Calculate the SL factor as the inverse of the reward ratio
    sl_factor = 1.0 / reward

    for i in range(len(y_actual) - 1):
        if cash <= 0:
            equity_curve.append(0); continue

        rel_perf = (cash - initial_cap) / initial_cap
        risk_rate = np.clip(0.02 + (rel_perf * 0.1), 0.01, 0.05)
        risk_amount = initial_cap * risk_rate * np.clip(abs(W[i]), 1.0, 3.0)

        price = price_values[i]
        atr = atr_values[i]
        friction = price * dynamic_slippage(atr/price)
        threshold = int(np.sign(o_d[i]) + np.sign(o_dd[i]))
        
        next_high = high_values[i+1]
        next_low = low_values[i+1]
        next_close = price_values[i+1]
        next_bar_return = next_close - price

        side = 0
        net = 0
        reason = 0

        # LONG SIGNAL
        if Id[i] > 0 and R[i] > 2 and M[i] > 0 and Yd[i] > 0:
            side = 1; longs += 1
            tp_dist = apply_integer_nudge(price, min(atr, e_high[i] - price), True, True)
            # Use dynamic sl_factor instead of 0.33
            sl_dist = apply_integer_nudge(price, sl_factor * min(atr, price - e_low[i]), False, True)
            sl_dist = max(sl_dist, price * 0.0001)
                
            if next_low <= (price - sl_dist): # SL Hit
                net = -(risk_amount + friction); loser_longs += 1; reason = -1
            elif next_high >= (price + tp_dist): # TP Hit
                # Use reward parameter instead of hardcoded 3
                net = (risk_amount * reward) - friction; winner_longs += 1; reason = 1
            else:
                # Use sl_factor to normalize realized profit
                realized = next_bar_return / (sl_factor * atr)
                net = (risk_amount * realized) - friction
                if net > 0 and  np.sign(o_d[i]) < 0: # Momentum Reversal
                    reason = 0
                    if next_bar_return > 0: winner_longs += 1
                    else: loser_longs += 1
        # SHORT SIGNAL
        elif Id[i] < 0 and R[i] > 2 and M[i] < 0 and Yd[i] < 0:
            side = -1; shorts += 1
            tp_dist = apply_integer_nudge(price, min(atr, price - e_low[i]), True, False)
            # Use dynamic sl_factor instead of 0.33
            sl_dist = apply_integer_nudge(price, sl_factor * min(atr, e_high[i] - price), False, False)
            sl_dist = max(sl_dist, price * 0.0001)

            if next_high >= (price + sl_dist): # SL Hit
                net = -(risk_amount + friction); loser_shorts += 1; reason = -1
            elif next_low <= (price - tp_dist): # TP Hit
                # Use reward parameter instead of hardcoded 3
                net = (risk_amount * reward) - friction; winner_shorts += 1; reason = 1
            else:
                # Use sl_factor to normalize realized profit
                realized = -next_bar_return / (sl_factor * atr)
                net = (risk_amount * realized) - friction
                if net > 0 and o_d[i] > 0: # Momentum Reversal
                    reason = 0
                    if next_bar_return < 0: winner_shorts += 1
                    else: loser_shorts += 1

        if side != 0:
            cash += net
            transaction_log.append(Transaction(
                ticker=ticker, trade_id=len(transaction_log), entry_index=i, exit_index=i+1,
                duration=1, side=side, entry_price=price, exit_price=next_close,
                pl=net, tp_price=price + (tp_dist if side==1 else -tp_dist),
                sl_price=price - (sl_dist if side==1 else -sl_dist), exit_reason=reason
            ))
        equity_curve.append(cash)

    return equity_curve, cash, longs, shorts, winner_longs, winner_shorts, loser_longs, loser_shorts, transaction_log

def simulate_trading_pd1(ticker, y_test, physics_test, reward, initial_cap=10000, k_window=14):        
    transaction_log = []    
    cash = initial_cap
    equity_curve, trade_returns = [initial_cap], []
    longs, shorts, winner_longs, winner_shorts, loser_longs, loser_shorts = 0, 0, 0, 0, 0, 0
    y_actual = y_test.values
    atr_values, price_values = physics_test['ATR'].values, physics_test['Close'].values
    high_values, low_values = physics_test['High'].values, physics_test['Low'].values
    e_low, e_high = physics_test['E_Low'].values, physics_test['E_High'].values
    M = physics_test['M'].values
    R = physics_test['R'].values
    
    for i in range(1, len(y_actual) - 1):
        if cash <= 0.25*initial_cap:
            equity_curve.append(0); continue
        
        k_factor = np.clip(np.mean(trade_returns[-k_window:]) * 0.5, -0.01, 0.02) if len(trade_returns) >= k_window else 0
        risk_rate = np.clip(0.02 + ((cash - initial_cap)/initial_cap * 0.05) + k_factor, 0.01, 0.05)
        risk_amount = initial_cap * risk_rate
        
        price = price_values[i]
        atr = atr_values[i]
        friction = price * dynamic_slippage(atr/price)
        
        next_high = high_values[i+1]
        next_low = low_values[i+1]
        next_close = price_values[i+1]
        next_bar_return = next_close - price

        side = 0
        net = 0
        reason = 0
# Calculate the SL factor based on the reward parameter
    # If reward=3, sl_factor=0.33. If reward=7, sl_factor=0.142
    sl_factor = 1.0 / reward

    for i in range(0, len(y_actual) - 1):
        # ... [Risk and Friction logic remains the same] ...
        if cash <= 0.25*initial_cap:
            equity_curve.append(0); continue

        # LONG EXECUTION
        if R[i] > 2 and M[i] > 0:
            side = 1; longs += 1
            # TP target is at a full ATR distance
            tp_dist = apply_integer_nudge(price, min(atr, price - e_low[i]), True, True)
            # SL is a fraction (1/reward) of the ATR target distance
            sl_dist = apply_integer_nudge(price, sl_factor * min(atr, e_high[i] - price), False, True)
            sl_dist = max(sl_dist, price * 0.0001)
            
            if next_low <= (price - sl_dist): # SL Hit
                net = -(risk_amount + friction); loser_longs += 1; reason = -1
            elif next_high >= (price + tp_dist): # TP Hit
                net = (risk_amount * reward) - friction; winner_longs += 1; reason = 1
            else:
                # Normalize partial return by the risk unit (sl_factor * atr)
                net = (risk_amount * (next_bar_return / (sl_factor * atr))) - friction
                reason = 0
                if next_bar_return > 0: winner_longs += 1
                else: loser_longs += 1

        # SHORT EXECUTION
        elif R[i] > 2 and M[i] < 0:
            side = -1; shorts += 1
            tp_dist = apply_integer_nudge(price, min(atr, price - e_low[i]), True, False)
            sl_dist = apply_integer_nudge(price, sl_factor * min(atr, e_high[i] - price), False, False)
            sl_dist = max(sl_dist, price * 0.0001)
            
            if next_high >= (price + sl_dist): # SL Hit
                net = -(risk_amount + friction); loser_shorts += 1; reason = -1
            elif next_low <= (price - tp_dist): # TP Hit
                net = (risk_amount * reward) - friction; winner_shorts += 1; reason = 1
            else:
                # Normalize partial return by the risk unit (sl_factor * atr)
                net = (risk_amount * (-next_bar_return / (sl_factor * atr))) - friction
                reason = 0
                if next_bar_return < 0: winner_shorts += 1
                else: loser_shorts += 1

        if side != 0:
            cash += net
            trade_returns.append(net / risk_amount)
            transaction_log.append(Transaction(
                ticker=ticker, trade_id=len(transaction_log), entry_index=i, exit_index=i+1,
                duration=1, side=side, entry_price=price, exit_price=next_close,
                pl=net, tp_price=price + (tp_dist if side==1 else -tp_dist),
                sl_price=price - (sl_dist if side==1 else -sl_dist), exit_reason=reason
            ))
        equity_curve.append(cash)

    return equity_curve, cash, longs, shorts, winner_longs, winner_shorts, loser_longs, loser_shorts, transaction_log