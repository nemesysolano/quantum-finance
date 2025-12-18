import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LinearRegression 
import os
import qf.market as mkt
import matplotlib.pyplot as plt
import qf.nn.models.base.pricevoldiff as pv_lib
import qf.nn.models.base.probdiff as pd_lib
import qf.nn.models.base.wavelets as ang_lib
import qf.nn.models.base.gauge as gauge_lib
from qf.nn.trainers.base import base_trainer
from types import SimpleNamespace

# --- NEW CONSTANTS (FEES AND REALISM) ---
BASE_CASH = 10000
TRADING_DAYS_PER_YEAR = 252

# Realistic Annual Risk-Free Rate (e.g., 4.0% - Proxy for 10Y Treasury yield)
ANNUAL_RISK_FREE_RATE = 0.04 
# --- REMOVED ALL FEES AND TAXES (Gross PNL only) ---
# ------------------------------------------

base_model_names = ('pricevol', 'wavelets', 'gauge')

def extract_meta_features(historical_data, models, k=14):
    """
    Executes all three base models and applies the logical sign corrections 
    to create a aligned dataset for the Meta-Learner.
    """
    m_pv, m_ang, m_g = models

    # Generate raw inputs for each model
    X_pv = pv_lib.create_inputs(historical_data, k)
    X_ang = ang_lib.create_inputs(historical_data, k) 
    X_g = gauge_lib.create_inputs(historical_data, k)

    # Align sample sizes
    min_samples = min(len(X_pv), len(X_ang), len(X_g))
    X_pv, X_ang, X_g = X_pv[-min_samples:],  X_ang[-min_samples:], X_g[-min_samples:]
    
    # Get Raw Model Predictions
    y_pv_raw = m_pv.predict(X_pv).flatten()
    y_ang_raw = m_ang.predict(X_ang).flatten()
    y_g_raw = m_g.predict(X_g).flatten()

    # APPLY TRADING SYSTEM LOGIC (Sign Corrections) - kept as is
    corrected_pv = y_pv_raw
    corrected_ang = y_ang_raw
    corrected_g = y_g_raw

    # Stack into feature matrix for the Meta-Learner
    return np.column_stack([corrected_pv, corrected_ang, corrected_g])

def get_limits(close_t, energy_evels, direction, risk_pct=0.01):    
    risk_dollars = 0
    reward_dollars = 0
    if type(energy_evels) is tuple:
        e_low, e_high = energy_evels
        if direction == 1:  # LONG
            risk_dollars = close_t - e_low
            reward_dollars = e_high - close_t
        elif direction == -1:  # SHORT
            risk_dollars = e_high - close_t
            reward_dollars = close_t - e_low

    
    # The Stop Loss is defined by the risk percentage of the entry price
    risk_dollars = min(risk_dollars, close_t * risk_pct)
    # The Take Profit is 3 times the risk (1:3 Risk:Reward)
    reward_dollars = max(risk_dollars * 3, reward_dollars)
    return risk_dollars, reward_dollars

def load_base_models(ticker):
    base_model_path = lambda name: os.path.join(os.getcwd(), 'models', f'{ticker}-{name}.keras')        
    for name in base_model_names:        
        if not os.path.exists(base_model_path(name)):
            d = {
                'epochs': 100,
                'patience': 50,
                'lookback': 14,
                'l2_rate': 1e-6,
                'dropout_rate': 0.20,
                'scale_features': 'yes',
                'ticker': ticker,
                'model': name
            }
            args = SimpleNamespace(**d)
            base_trainer(args)

    models = tuple([tf.keras.models.load_model(base_model_path(name)) for name in base_model_names])
    return models

def calculate_stats(equity_curve):
    """
    Calculates key performance metrics (Return, Max Drawdown, Sharpe Ratio).
    """
    history = np.array(equity_curve)
    
    # 1. Total Return
    initial_equity = history[0]
    final_equity = history[-1]
    if final_equity <= initial_equity or initial_equity == 0 or len(history) < 2:
        total_return = 0.0
    else:
        total_return = (final_equity - initial_equity) / initial_equity
    
    # 2. Max Drawdown
    peak = np.maximum.accumulate(history)
    if np.max(peak) == 0:
        max_drawdown = 0.0
    else:
        drawdown = (peak - history) / peak
        max_drawdown = np.max(drawdown)
    
    # 3. Sharpe Ratio (Annualized)
    daily_returns = np.diff(history) / history[:-1]
    
    if len(daily_returns) < 2 or np.std(daily_returns) == 0:
        sharpe_ratio = 0.0
    else:
        avg_daily_return = np.mean(daily_returns)
        std_daily_return = np.std(daily_returns)
        
        annualized_return = avg_daily_return * TRADING_DAYS_PER_YEAR
        annualized_volatility = std_daily_return * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        if annualized_volatility == 0:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = (annualized_return - ANNUAL_RISK_FREE_RATE) / annualized_volatility

    return {
        'Return': float(total_return),
        'Max Drawdown': float(max_drawdown),
        'Sharpe Ratio': float(sharpe_ratio)
    }

def train_and_test_ensemble(base_models, data, thresholds):
    """
    Simulates a non-intraday strategy (EOD PNL) with 0% tax and 1.0% transaction rate.
    """
    # 1. Use the centralized dataset splitting from mkt.create_datasets
    _, _, _, meta_train_data, test_data = mkt.create_datasets(data)

    # 2. TRAIN META-LEARNER 
    meta_X_train = extract_meta_features(meta_train_data, base_models)
    
    meta_prices = meta_train_data['Close'].values[-len(meta_X_train)-1:]
    meta_y_train = np.diff(meta_prices) / meta_prices[:-1] 

    meta_model = LinearRegression()
    meta_model.fit(meta_X_train, meta_y_train)

    # 3. BACKTEST ON FINAL UNSEEN SET 
    test_X_all = extract_meta_features(test_data, base_models)
    test_X_meta = test_X_all[:-1]
    
    execution_length = len(test_X_meta)
    
    test_opens  = test_data['Open'].values[-execution_length:]
    test_closes = test_data['Close'].values[-execution_length:]
    test_highs  = test_data['High'].values[-execution_length:]
    test_lows   = test_data['Low'].values[-execution_length:] 
    
    test_E_High = test_data['E_High'].values[-execution_length:]
    test_E_Low  = test_data['E_Low'].values[-execution_length:] 

    raw_predictions_pct = meta_model.predict(test_X_meta)

    # --- POSITION SIZING CONSTANTS ---
    STARTING_CAPITAL = BASE_CASH 
    MAX_EQUITY_RISK_PCT = 0.01   
    # ---------------------------------

    results = {}
    
    for threshold in thresholds:
        equity_curve = [BASE_CASH] 
        current_equity = STARTING_CAPITAL 
        
        for i in range(len(test_X_meta)):
            pred_pct = raw_predictions_pct[i] 
            entry_price = test_opens[i] 
            close_price = test_closes[i] 
            
            pnl = 0
            share_count = 0
            
            # Calculate Risk (R) and Reward (3R)
            direction = 1 if pred_pct > threshold else -1 if pred_pct < -threshold else 0
            risk_dollars, reward_dollars = get_limits(entry_price, (test_E_Low[i], test_E_High[i]), direction)
            
            if direction != 0 and risk_dollars > 0:
                
                # 1. Position Sizing
                dollar_risk_limit = current_equity * MAX_EQUITY_RISK_PCT
                share_count = np.floor(dollar_risk_limit / risk_dollars)
                
                # 2. Affordability Check
                max_affordable_shares = np.floor(current_equity / entry_price)
                share_count = min(share_count, max_affordable_shares)
                
                
                if share_count >= 1: 
                    
                    if pred_pct > threshold: # LONG
                        # Simplified PNL for Non-Intraday Strategy (Open-to-Close)
                        gross_pnl = close_price - entry_price 
                            
                    elif pred_pct < -threshold: # SHORT
                        # Simplified PNL for Non-Intraday Strategy (Open-to-Close)
                        gross_pnl = -(close_price - entry_price)

                    # 4. Scale PnL by shares
                    gross_pnl *= share_count
                    pnl = gross_pnl
                    
                else:
                    # Trade filtered out: PNL is zero for this day
                    pnl = 0
            
            current_equity += pnl 
            equity_curve.append(current_equity)
        
        results[threshold] = equity_curve
    return test_data, results, meta_model

def best_threshold(results):
    """
    Finds the best threshold based on total return and returns the full stats.
    """
    max_return = -float('inf')
    best_history = []
    best_thresh = 0
    best_stats = {}
    
    for threshold, history in results.items():
        if len(history) < 2: continue
            
        stats = calculate_stats(history)
        current_return = stats['Return']
        
        if current_return > max_return:
            max_return = current_return
            best_history = history
            best_thresh = threshold
            best_stats = stats
            
    best_stats['Best Threshold'] = best_thresh

    return best_history, best_stats

def meta_trainer_run(ticker):
    data = mkt.import_market_data(ticker)        
    
    base_models = load_base_models(ticker)
    magnitude_thresholds = [5e-5, 0.001, 0.003, 0.005, 0.010, 0.015, 0.020, 0.030]
    
    test_data, backtest_results, meta_model = train_and_test_ensemble(base_models, data, magnitude_thresholds)
    
    best_hist, best_stats = best_threshold(backtest_results)

    return best_hist, test_data, best_stats, meta_model

def meta_trainer(args):
    ticker = args.ticker.upper()
    
    best_hist, test_data, best_stats, meta_model = meta_trainer_run(ticker)

    output_file = os.path.join(os.getcwd(), "test-results", f"report-backtest.json")
    mode = 'a' if os.path.exists(output_file) else 'w'
    with open(output_file, mode) as f:
        report_data = {
            "history": best_hist,
            "stats": best_stats
        }

        if mode == 'w':
            print("{", file=f) 
            print(f"\"{ticker}\": {json.dumps(report_data)}", file=f) 
        else:
            print(f", \"{ticker}\": {json.dumps(report_data)}", file=f)