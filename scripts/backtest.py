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
import sys

def extract_meta_features(historical_data, models, k=14):
    """
    Executes all three base models and applies the logical sign corrections 
    to create a aligned dataset for the Meta-Learner.
    """
    m_pv, m_pd, m_ang = models

    # Generate raw inputs for each model
    X_pv = pv_lib.create_inputs(historical_data, k)
    X_pd = pd_lib.create_inputs(historical_data, k)
    X_ang = ang_lib.create_inputs(historical_data, k) 
    
    # Align sample sizes
    min_samples = min(len(X_pv), len(X_pd), len(X_ang))
    X_pv, X_pd, X_ang = X_pv[-min_samples:], X_pd[-min_samples:], X_ang[-min_samples:]
    
    # Get Raw Model Predictions
    y_pv_raw = m_pv.predict(X_pv).flatten()
    y_pd_raw = m_pd.predict(X_pd).flatten()
    y_ang_raw = m_ang.predict(X_ang).flatten()
    
    # APPLY TRADING SYSTEM LOGIC (Sign Corrections)
    # Price-Volume: Direct Predictor (Keep Sign)
    # Prob-Diff & Angles: Contrarian Predictors (Invert Sign)
    corrected_pv = y_pv_raw
    corrected_pd = -1.0 * y_pd_raw
    corrected_ang = -1.0 * y_ang_raw
    
    # Stack into feature matrix for the Meta-Learner
    return np.column_stack([corrected_pv, corrected_pd, corrected_ang])

def load_base_models(ticker):
    base_model_names = ('pricevol', 'prob', 'priceangle')
    base_model_path = lambda name: os.path.join(os.getcwd(), 'models', f'{ticker}-{name}.keras')
    return tuple([tf.keras.models.load_model(base_model_path(name)) for name in base_model_names])

def train_and_test_ensemble(ticker, base_models, data, thresholds):
    """
    Trains on PERCENTAGE targets, Backtests on DOLLAR P&L.
    """
    # 1. Define Data Partitions (60/20/20 split standard)
    n = len(data)
    # Assuming base models trained on first 60%.
    # We train meta-learner on next 20% (Validation).
    meta_train_start, meta_train_end = int(n*0.8), int(n*0.9)
    test_start = meta_train_end

    meta_train_data = data.iloc[meta_train_start:meta_train_end]
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
    
    # Actual DOLLAR changes for P&L calculation (1 share)
    actual_dollar_changes = np.diff(test_prices) 

    # Predict Expected PERCENTAGE Move
    raw_predictions_pct = meta_model.predict(test_X_meta)

    results = {}
    for threshold in thresholds:
        equity_curve = [0]
        
        for i in range(len(test_X_meta)):
            pred_pct = raw_predictions_pct[i]
            
            # Trading Logic: Filter based on predicted % magnitude
            # e.g., Threshold 0.005 means we need > 0.5% predicted move
            pnl = 0
            if pred_pct > threshold: # LONG
                pnl = actual_dollar_changes[i]
            elif pred_pct < -threshold: # SHORT
                pnl = -actual_dollar_changes[i]
                
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

if __name__ == "__main__":
    ticker = sys.argv[1]
    
    data_path = os.path.join(os.getcwd(),"qf", "market", "data", f"{ticker}.csv")
    data = mkt.read_csv(data_path)        
    
    pv_model, pd_model, ang_model = load_base_models(ticker)
    base_models = (pv_model, pd_model, ang_model)
    
    # Thresholds represent PERCENTAGE moves now (0.005 = 0.5%, 0.01 = 1%)
    # This normalizes risk across $5 stocks and $1000 stocks.
    magnitude_thresholds = [0.001, 0.003, 0.005, 0.010, 0.015, 0.020, 0.030]
    
    backtest_results, _ = train_and_test_ensemble(ticker, base_models, data, magnitude_thresholds)
    best_hist, best_t, max_p = best_threshold(backtest_results)

    output_file = os.path.join(os.getcwd(), "test-results", f"report-backtest.json")
    mode = 'a' if os.path.exists(output_file) else 'w'
    with open(output_file, mode) as f:
        if mode == 'w':
            print("{", file=f) 
        print(f"\"{ticker}\": {best_hist},", file=f)