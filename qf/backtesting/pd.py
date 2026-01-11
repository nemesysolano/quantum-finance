from qf.backtesting.util import Transaction, apply_integer_nudge, dynamic_slippage, Position
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

def simulate_trading_pd(ticker, y_test, physics_test, reward, initial_cap=10000):    
    transaction_log = []    
    cash = initial_cap
    equity_curve = [initial_cap]
    
    # Tracking metrics
    longs, shorts = 0, 0
    winner_longs, winner_shorts = 0, 0
    loser_longs, loser_shorts = 0, 0
    
    # State variable for the open position
    active_position = None

    # Data pre-fetching
    price_values = physics_test['Close'].values
    high_values = physics_test['High'].values
    low_values = physics_test['Low'].values
    atr_values = physics_test['ATR'].values
    e_low = physics_test['E_Low'].values
    e_high = physics_test['E_High'].values
    
    M = physics_test['M'].values
    R = physics_test['R'].values
    Id = physics_test['Id'].values
    Yd = physics_test['Yd'].values
    o_d = physics_test['Öd'].values
    W = physics_test['W'].values 

    # Stop Loss factor based on Reward Ratio
    sl_factor = 1.0 / reward

    for i in range(len(price_values)):
        if cash <= 0:
            equity_curve.append(0)
            continue

        price = price_values[i]
        curr_high = high_values[i]
        curr_low = low_values[i]

        # 1. CHECK FOR EXITS ON ACTIVE POSITION
        if active_position is not None:
            pos = active_position
            # Only exit on bars after the entry bar to avoid look-ahead bias
            if i > pos.entry_index:
                exit_signal = False
                exit_reason = 0
                exit_price = price
                net_pl = 0

                # LONG EXIT CHECKS
                if pos.side == 1:
                    if curr_low <= pos.sl:
                        exit_price = pos.sl
                        exit_reason = -1 # SL
                        net_pl = -(pos.amount + friction_at_entry) # Simplified logic for example
                        loser_longs += 1
                        exit_signal = True
                    elif curr_high >= pos.tp:
                        exit_price = pos.tp
                        exit_reason = 1 # TP
                        net_pl = (pos.amount * reward) - friction_at_entry
                        winner_longs += 1
                        exit_signal = True
                    elif np.sign(o_d[i]) < 0: # Momentum Reversal
                        exit_price = price
                        exit_reason = 0 
                        # Realized P/L calculation based on entry
                        realized_r = (exit_price - pos.entry_price) / (sl_factor * atr_values[pos.entry_index])
                        net_pl = (pos.amount * realized_r) - friction_at_entry
                        if net_pl > 0: winner_longs += 1
                        else: loser_longs += 1
                        exit_signal = True

                # SHORT EXIT CHECKS
                elif pos.side == -1:
                    if curr_high >= pos.sl:
                        exit_price = pos.sl
                        exit_reason = -1
                        net_pl = -(pos.amount + friction_at_entry)
                        loser_shorts += 1
                        exit_signal = True
                    elif curr_low <= pos.tp:
                        exit_price = pos.tp
                        exit_reason = 1
                        net_pl = (pos.amount * reward) - friction_at_entry
                        winner_shorts += 1
                        exit_signal = True
                    elif o_d[i] > 0: # Momentum Reversal
                        exit_price = price
                        exit_reason = 0
                        realized_r = (pos.entry_price - exit_price) / (sl_factor * atr_values[pos.entry_index])
                        net_pl = (pos.amount * realized_r) - friction_at_entry
                        if net_pl > 0: winner_shorts += 1
                        else: loser_shorts += 1
                        exit_signal = True

                if exit_signal:
                    cash += net_pl
                    transaction_log.append(Transaction(
                        ticker=ticker,
                        trade_id=len(transaction_log),
                        entry_index=pos.entry_index,
                        exit_index=i,
                        duration=i - pos.entry_index,
                        side=pos.side,
                        entry_price=pos.entry_price,
                        exit_price=exit_price,
                        pl=net_pl,
                        tp_price=pos.tp,
                        sl_price=pos.sl,
                        exit_reason=exit_reason
                    ))
                    active_position = None

        # 2. CHECK FOR NEW ENTRIES
        if active_position is None:
            rel_perf = (cash - initial_cap) / initial_cap
            risk_rate = np.clip(0.02 + (rel_perf * 0.1), 0.01, 0.05)
            risk_amount = initial_cap * risk_rate * np.clip(abs(W[i]), 1.0, 3.0)
            
            atr = atr_values[i]
            friction_at_entry = price * dynamic_slippage(atr/price) # Captured at entry for exit calc
            
            # LONG SIGNAL
            if Id[i] > 0 and R[i] > 2 and M[i] > 0 and Yd[i] > 0:
                tp_dist = apply_integer_nudge(price, min(atr, e_high[i] - price), True, True)
                sl_dist = apply_integer_nudge(price, sl_factor * min(atr, price - e_low[i]), False, True)
                sl_dist = max(sl_dist, price * 0.0001)
                
                active_position = Position(
                    ticker=ticker, entry_index=i, entry_price=price,
                    amount=risk_amount, side=1, tp=price + tp_dist, sl=price - sl_dist
                )
                longs += 1

            # SHORT SIGNAL
            elif Id[i] < 0 and R[i] > 2 and M[i] < 0 and Yd[i] < 0:
                tp_dist = apply_integer_nudge(price, min(atr, price - e_low[i]), True, False)
                sl_dist = apply_integer_nudge(price, sl_factor * min(atr, e_high[i] - price), False, False)
                sl_dist = max(sl_dist, price * 0.0001)

                active_position = Position(
                    ticker=ticker, entry_index=i, entry_price=price,
                    amount=risk_amount, side=-1, tp=price - tp_dist, sl=price + sl_dist
                )
                shorts += 1

        equity_curve.append(cash)

    return equity_curve, cash, longs, shorts, winner_longs, winner_shorts, loser_longs, loser_shorts, transaction_log