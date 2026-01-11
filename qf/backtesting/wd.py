from qf.backtesting.util import Transaction, apply_integer_nudge, dynamic_slippage
import numpy as np

def simulate_trading_wd(ticker, y_test, physics_test, reward, initial_cap=10000):    
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
        if Id[i] > 0 and W[i] > 0 and threshold == 2:
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
                if net > 0 and  (np.sign(o_dd[i]) < 0 and np.sign(o_d[i]) > 0): # Momentum Reversal
                    reason = 0
                    if next_bar_return > 0: winner_longs += 1
                    else: loser_longs += 1
        # SHORT SIGNAL
        elif Id[i] < 0 and W[i] < 0 and threshold == -2:
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
                if net > 0 and (np.sign(o_dd[i]) > 0 and np.sign(o_d[i]) < 0): # Momentum Reversal
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