from qf.backtesting.util import Transaction, apply_integer_nudge, dynamic_slippage, Position
import numpy as np
import pandas as pd

def simulate_trading_yd(ticker, y_test, physics_test, reward, initial_cap=10000):    
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
    
    Yd = physics_test['Yd'].values # Momentum is used for exits
    Ydd = physics_test['Ydd'].values # Acceleration is used for entries
  
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
                        net_pl = -(pos.amount + pos.friction_at_entry) # Simplified logic for example
                        loser_longs += 1
                        exit_signal = True
                    elif curr_high >= pos.tp:
                        exit_price = pos.tp
                        exit_reason = 1 # TP
                        net_pl = (pos.amount * reward) - pos.friction_at_entry
                        winner_longs += 1
                        exit_signal = True
                    elif Yd[i] > 0 and Ydd[i] < 0 or price > e_high[i]: # Momentum Reversal
                        exit_price = price
                        exit_reason = 0 
                        realized_r = (exit_price - pos.entry_price) / (sl_factor * atr_values[pos.entry_index])
                        net_pl = (pos.amount * realized_r) - pos.friction_at_entry

                        if net_pl > 0:
                            winner_longs += 1                        
                            exit_signal = True

                # SHORT EXIT CHECKS
                elif pos.side == -1:
                    if curr_high >= pos.sl:
                        exit_price = pos.sl
                        exit_reason = -1
                        net_pl = -(pos.amount + pos.friction_at_entry)
                        loser_shorts += 1
                        exit_signal = True
                    elif curr_low <= pos.tp:
                        exit_price = pos.tp
                        exit_reason = 1
                        net_pl = (pos.amount * reward) - pos.friction_at_entry
                        winner_shorts += 1
                        exit_signal = True
                    elif Yd[i] < 0 and Ydd[i] > 0 or price < e_low[i]: # Momentum Reversal
                        exit_price = price
                        exit_reason = 0
                        realized_r = (pos.entry_price - exit_price) / (sl_factor * atr_values[pos.entry_index])
                        net_pl = (pos.amount * realized_r) - friction_at_entry
                        if net_pl > 0: 
                            winner_shorts += 1
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
                        exit_reason=exit_reason,
                        friction_at_entry=pos.friction_at_entry
                    ))
                    active_position = None

        # 2. CHECK FOR NEW ENTRIES
        if active_position is None:
            rel_perf = (cash - initial_cap) / initial_cap
            risk_rate = np.clip(0.02 + (rel_perf * 0.1), 0.01, 0.05)
            risk_amount = initial_cap * risk_rate
            
            atr = atr_values[i]
            friction_at_entry = price * dynamic_slippage(atr/price) # Captured at entry for exit calc
            
            # LONG SIGNAL
            if Yd[i] < 0 and Ydd[i] > 0:
                tp_dist = apply_integer_nudge(price, min(atr, e_high[i] - price), True, True)
                sl_dist = apply_integer_nudge(price, sl_factor * min(atr, price - e_low[i]), False, True)
                sl_dist = max(sl_dist, price * 0.0001)
                
                active_position = Position(
                    ticker=ticker, entry_index=i, entry_price=price,
                    amount=risk_amount, side=1, tp=price + tp_dist, sl=price - sl_dist, friction_at_entry=friction_at_entry
                )
                longs += 1

            # SHORT SIGNAL
            elif Yd[i] > 0 and Ydd[i] < 0:
                tp_dist = apply_integer_nudge(price, min(atr, price - e_low[i]), True, False)
                sl_dist = apply_integer_nudge(price, sl_factor * min(atr, e_high[i] - price), False, False)
                sl_dist = max(sl_dist, price * 0.0001)

                active_position = Position(
                    ticker=ticker, entry_index=i, entry_price=price,
                    amount=risk_amount, side=-1, tp=price - tp_dist, sl=price + sl_dist, friction_at_entry=friction_at_entry
                )
                shorts += 1

        equity_curve.append(cash)

    return equity_curve, cash, longs, shorts, winner_longs, winner_shorts, loser_longs, loser_shorts, transaction_log