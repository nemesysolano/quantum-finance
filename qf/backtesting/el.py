from qf.backtesting.util import Position, Transaction, apply_integer_nudge, dynamic_slippage
import numpy as np

def simulate_trading_el(ticker, y_test, physics_test, reward, initial_cap=10000):    
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

    # Calculate the SL factor as the inverse of the reward ratio
    sl_factor = 1.0 / reward
    active_position = None

    for i in range(len(y_actual) - 1):
        if cash <= 0:
            equity_curve.append(0); continue

        rel_perf = (cash - initial_cap) / initial_cap
        risk_rate = np.clip(0.02 + (rel_perf * 0.1), 0.01, 0.05)
        risk_amount = initial_cap * risk_rate * np.clip(abs(W[i]), 1.0, 3.0)

        price = price_values[i]
        atr = atr_values[i]
        friction = price * dynamic_slippage(atr/price)
        net_pl = 0
        exit_reason = 0

        # 1. CHECK FOR EXIT OF ACTIVE POSITION
        # If there's an active position, check for exit conditions
        if active_position is not None:
            pos = active_position
            current_low = low_values[i]
            current_high = high_values[i]
            exit_signal = False
            exit_price = price
            
            if i > pos.entry_index:
                if pos.side == 1:
                    if current_low <= pos.sl:
                        exit_price = pos.sl
                        exit_reason = -1
                        net_pl = -(pos.amount + pos.friction_at_entry)
                        loser_longs += 1
                        exit_signal = True
                    elif current_high >= pos.tp:
                        exit_price = pos.tp
                        exit_reason = 1
                        net_pl = (pos.amount * reward) - pos.friction_at_entry
                        winner_longs += 1
                        exit_signal = True
                    elif o_d[i] < 0: # Momentum Reversal
                        exit_price = price
                        exit_reason = 0
                        realized_r = (exit_price - pos.entry_price)
                        net_pl = (pos.amount * realized_r) - pos.friction_at_entry
                        if net_pl > 0:
                            winner_longs += 1
                            exit_signal = True
                else:
                    if current_high >= pos.sl:
                        exit_price = pos.sl
                        exit_reason = -1
                        net_pl = -(pos.amount + pos.friction_at_entry)
                        loser_shorts += 1
                        exit_signal = True
                    elif current_low <= pos.tp:
                        exit_price = pos.tp
                        exit_reason = 1
                        net_pl = (pos.amount * reward) - pos.friction_at_entry
                        winner_shorts += 1
                        exit_signal = True
                    elif o_d[i] > 0: # Momentum Reversal
                        exit_price = price
                        exit_reason = 0
                        realized_r = (pos.entry_price - exit_price)
                        net_pl = (pos.amount * realized_r) - pos.friction_at_entry
                        if net_pl > 0:
                            winner_shorts += 1
                            exit_signal = True
            
            if exit_signal:
                cash += net_pl
                transaction_log.append(Transaction(
                    ticker=ticker,
                    trade_id=len(transaction_log) + 1,
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
                    
        if active_position is None:
            # LONG SIGNAL
            tp_dist = e_high[i] - price
            sl_dist = price - e_low[i]
            if np.abs(tp_dist/sl_dist) < reward and o_dd[i] > 0:
                longs += 1
                active_position = Position(
                    ticker=ticker, entry_index=i, entry_price=price,
                    amount=risk_amount, side=1, tp=price + tp_dist, sl=price - sl_dist, friction_at_entry=friction
                )
            else:
                # SHORT SIGNAL
                tp_dist = price - e_low[i]
                sl_dist = e_high[i] - price
                if np.abs(tp_dist/sl_dist) < reward and o_dd[i] < 0:
                    shorts += 1
                    active_position = Position(
                        ticker=ticker, entry_index=i, entry_price=price,
                        amount=risk_amount, side=-1, tp=price - tp_dist, sl=price + sl_dist, friction_at_entry=friction
                    )
        equity_curve.append(cash)

    return equity_curve, cash, longs, shorts, winner_longs, winner_shorts, loser_longs, loser_shorts, transaction_log