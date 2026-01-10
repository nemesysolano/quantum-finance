import sys
from multiprocessing import Pool
from qf.backtesting.util import apply_integer_nudge, dynamic_slippage
import qf.market as mkt
import os
import tensorflow as tf
from qf.nn import fracdiff
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
keras = tf.keras

def simulate_trading_ds(y_test, physics_test, initial_cap=10000, k_window=14): 
    transaction_log = []
    
    physics_test  = physics_test[physics_test['Ds'] != 0]
    y_test = y_test[y_test.index.isin(physics_test.index)]
    
    cash = initial_cap
    equity_curve, trade_returns = [initial_cap], []
    longs, shorts, winner_longs, winner_shorts, loser_longs, loser_shorts = 0, 0, 0, 0, 0, 0
    
    y_actual = y_test.values
    o_d, o_dd = physics_test['Öd'].values, physics_test['Ödd'].values
    atr_values, price_values = physics_test['ATR'].values, physics_test['Close'].values
    e_low, e_high = physics_test['E_Low'].values, physics_test['E_High'].values
    ds = physics_test['Ds'].values
    
    for i in range(len(y_actual) - 1):
        if cash <= 0:
            equity_curve.append(0); continue

        k_factor = np.clip(np.mean(trade_returns[-k_window:]) * 0.5, -0.01, 0.02) if len(trade_returns) >= k_window else 0
        risk_rate = np.clip(0.02 + ((cash - initial_cap)/initial_cap * 0.05) + k_factor, 0.01, 0.05)
        risk_amount = initial_cap * risk_rate

        atr, price = atr_values[i], price_values[i]
        threshold = int(np.sign(o_d[i]) + np.sign(o_dd[i]))
        next_bar_return = y_actual[i + 1]
        friction = cash * dynamic_slippage(atr/price) * 2
        
        if ds[i] < 0:
            longs += 1
            tp_dist = apply_integer_nudge(price, min(atr, e_high[i] - price), True, True)
            sl_dist = apply_integer_nudge(price, min(0.33 * atr, price - e_low[i]), False, True)
            
            if next_bar_return >= tp_dist:
                net = (risk_amount * 3) - friction; winner_longs += 1
            elif next_bar_return <= -sl_dist:
                net = -(risk_amount + friction); loser_longs += 1
            else:
                net = (risk_amount * (next_bar_return / (0.33 * atr))) - friction
                if next_bar_return > 0: winner_longs += 1
                else: loser_longs += 1
            cash += net; trade_returns.append(net / risk_amount)

        # SHORT EXECUTION
        elif ds[i] > 0:
            shorts += 1
            tp_dist = apply_integer_nudge(price, min(atr, price - e_low[i]), True, False)
            sl_dist = apply_integer_nudge(price, 0.33 * min(atr, e_high[i] - price), False, False)
            
            if next_bar_return <= -tp_dist:
                net = (risk_amount * 3) - friction; winner_shorts += 1
            elif next_bar_return >= sl_dist:
                net = -(risk_amount + friction); loser_shorts += 1
            else:
                net = (risk_amount * (-next_bar_return / (0.33 * atr))) - friction
                if next_bar_return < 0: winner_shorts += 1
                else: loser_shorts += 1
            cash += net; trade_returns.append(net / risk_amount)

        equity_curve.append(cash)
    return equity_curve, cash, longs, shorts, winner_longs, winner_shorts, loser_longs, loser_shorts, transaction_log