import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LinearRegression 
import os
import qf.market as mkt
import matplotlib.pyplot as plt
import qf.nn.models.base.pricevoldiff as pv_lib
import qf.nn.models.base.probdiff as pd_lib
import qf.nn.models.base.priceangle as ang_lib
import qf.nn.models.base.gauge as gauge_lib
import qf.nn as nn
import argparse

base_model_names = ('pricevol', 'priceangle', 'gauge')

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

    # APPLY TRADING SYSTEM LOGIC (Sign Corrections)
    # Price-Volume: Direct Predictor (Keep Sign)
    # Prob-Diff & Angles: Contrarian Predictors (Invert Sign)
    # Gauge: Direct Predictor (Keep Sign)
    corrected_pv = y_pv_raw
    corrected_ang = y_ang_raw
    corrected_g = y_g_raw

    # Stack into feature matrix for the Meta-Learner
    return np.column_stack([corrected_pv, corrected_ang, corrected_g])

# START OF NEW FUNCTION FOR RISK MANAGEMENT
def get_limits(close_t, energy_evels, direction, risk_pct=0.01):
    """
    Calculates the dollar value of the Stop Loss (Risk) and Take Profit (Reward) 
    targets based on the current price and a 1:3 Risk:Reward ratio.
    
    Args:
        close_t (float): The closing price at time t.
        risk_pct (float): The percentage of the price to use for the Stop Loss (Risk).
                          Default is 1.0% (0.01).
                          
    Returns:
        tuple: (risk_dollars, reward_dollars)
    """

    risk_dollars = 0
    reward_dollars = 0
    if type(energy_evels) is tuple:
        e_low, e_high = energy_evels
        # Optionally, you could use energy levels to adjust risk here.
        # For simplicity, we are not using them in this calculation.
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
    # These are the absolute dollar differences from the entry price.
    return risk_dollars, reward_dollars
# END OF NEW FUNCTION

def load_base_models(args):
    ticker = args.ticker.upper()
    base_model_path = lambda name: os.path.join(os.getcwd(), 'models', f'{ticker}-{name}.keras')        
    for name in base_model_names:        
        if not os.path.exists(base_model_path(name)):
            args.model = name
            nn.base_trainer(args)

    models = tuple([tf.keras.models.load_model(base_model_path(name)) for name in base_model_names])
    return models

def train_and_test_ensemble(ticker, base_models, data, thresholds):
    """
    Trains on PERCENTAGE targets, Backtests on DOLLAR P&L.
    """
    # 1. Define Data Partitions (60/20/20 split standard)
    n = len(data)
    # Assuming base models trained on first 60%.
    # We train meta-learner on next 20% (Validation).
    meta_train_start, meta_train_end = int(n * 0.6), int(n * 0.8)
    test_start = meta_train_end # Test on the final 20%

    meta_train_data = data.iloc[meta_train_start:meta_train_end].copy()
    test_data = data.iloc[test_start:]

    # 2. TRAIN META-LEARNER (Level 1) using Linear Regression on PERCENTAGE
    meta_X_train = extract_meta_features(meta_train_data, base_models)
    
    # Calculate Percentage Change Targets for Training: (C_t+1 - C_t) / C_t
    meta_prices = meta_train_data['Close'].values[-len(meta_X_train)-1:]
    # np.diff(p) is (p[i+1]-p[i]), divide by p[i] for percentage
    meta_y_train = np.diff(meta_prices) / meta_prices[:-1] 

    meta_model = LinearRegression()
    meta_model.fit(meta_X_train, meta_y_train)
    # print(f"Meta-Learner Trained for {ticker} on % targets.")

    # 3. BACKTEST ON FINAL UNSEEN SET
    test_X_meta = extract_meta_features(test_data, base_models)
    test_prices = test_data['Close'].values[-len(test_X_meta)-1:]
    test_E_High  = test_data['E_High'].values[-len(test_X_meta)-1:]
    test_E_Low  = test_data['E_High'].values[-len(test_X_meta)-1:]


    # Predict Expected PERCENTAGE Move
    raw_predictions_pct = meta_model.predict(test_X_meta)

    results = {}
    # Use 1% as the default risk percentage for the stop loss calculation
    DEFAULT_RISK_PCT = 0.01 
    
    for threshold in thresholds:
        equity_curve = [0]
        
        for i in range(len(test_X_meta)):
            pred_pct = raw_predictions_pct[i] 
            entry_price = test_prices[i]
            exit_price = test_prices[i+1] # The Close price on the next day
            
            # Calculate Risk (R) and Reward (3R) in dollars
            direction = 1 if pred_pct > threshold else -1 if pred_pct < -threshold else 0
            risk_dollars, reward_dollars = get_limits(entry_price, (test_E_Low[i], test_E_High[i]), direction)
            
            pnl = 0
            
            if pred_pct > threshold: # LONG TRADE
                sl_limit = entry_price - risk_dollars
                tp_limit = entry_price + reward_dollars
                
                if exit_price <= sl_limit:
                    pnl = -risk_dollars  # Hit Stop Loss (Loss = -R)
                elif exit_price >= tp_limit:
                    pnl = reward_dollars # Hit Take Profit (Gain = +3R)
                else:
                    # Close-to-close P&L if neither limit was hit
                    pnl = exit_price - entry_price 
                    
            elif pred_pct < -threshold: # SHORT TRADE
                sl_limit = entry_price + risk_dollars
                tp_limit = entry_price - reward_dollars
                
                if exit_price >= sl_limit:
                    pnl = -risk_dollars # Hit Stop Loss (Loss = -R)
                elif exit_price <= tp_limit:
                    pnl = reward_dollars # Hit Take Profit (Gain = +3R)
                else:
                    # Close-to-close P&L
                    pnl = -(exit_price - entry_price) # Negative of the price change
                
            # If no trade (prediction inside threshold), pnl remains 0
            
            equity_curve.append(equity_curve[-1] + pnl)
        
        results[threshold] = equity_curve

    return results, meta_model

def best_threshold(results):
    max_pnl = -float('inf')
    best_history = []
    best_thresh = 0
    for threshold, history in results.items():
        if history[-1] > max_pnl:
            max_pnl = history[-1]
            best_history = history
            best_thresh = threshold
    return best_history, best_thresh, max_pnl

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', type=str, help='Ticker symbol in NYSE')    
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--patience', type=int, default=50, help='Number of epochs with no improvement after which training will be stopped.')
    parser.add_argument('--lookback', type=int, default=14, help='Lookback period required for certain indicators (like RSI).')
    parser.add_argument('--l2_rate', type=float, default=1e-6, help='Number of epochs with no improvement after which training will be stopped.')
    parser.add_argument('--dropout_rate', type=float, default=0.20, help='Number of epochs with no improvement after which training will be stopped.')
    parser.add_argument('--scale_features', type=str, default='yes', choices=['yes', 'no'])

    arguments = parser.parse_args()
    return arguments

if __name__ == "__main__":
    args = arguments()
    ticker = args.ticker.upper()    
    data = mkt.import_market_data(ticker)        
    
    base_models = load_base_models(args)
    # Thresholds represent PERCENTAGE moves now (0.005 = 0.5%, 0.01 = 1%)
    # This normalizes risk across $5 stocks and $1000 stocks.
    magnitude_thresholds = [5e-5, 0.001, 0.003, 0.005, 0.010, 0.015, 0.020, 0.030]
    
    backtest_results, _ = train_and_test_ensemble(ticker, base_models, data, magnitude_thresholds)
    best_hist, best_t, max_p = best_threshold(backtest_results)

    output_file = os.path.join(os.getcwd(), "test-results", f"report-backtest.json")
    mode = 'a' if os.path.exists(output_file) else 'w'
    with open(output_file, mode) as f:
        if mode == 'w':
            print("{", file=f) 
        print(f"\"{ticker}\": {best_hist},", file=f)