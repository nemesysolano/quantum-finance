import json
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import qf.market as mkt
from .ensemble import ANNUAL_RISK_FREE_RATE, TRADING_DAYS_PER_YEAR, train_and_test_sgd_ensemble, train_and_test_linear_ensemble, train_and_test_combined_ensemble, base_model_names
from types import SimpleNamespace

def load_base_models(ticker):
    base_model_path = lambda name: os.path.join(os.getcwd(), 'models', f'{ticker}-{name}.keras')        
    for name in base_model_names:        
        if os.path.exists(base_model_path(name)):
            models = tuple([tf.keras.models.load_model(base_model_path(name)) for name in base_model_names])            
        else:
            print(f"ERROR: {base_model_path} can't be loaded")
            exit()
            # d = {
            #     'epochs': 100,
            #     'patience': 50,
            #     'lookback': 14,
            #     'l2_rate': 1e-6,
            #     'dropout_rate': 0.20,
            #     'scale_features': 'yes',
            #     'ticker': ticker,
            #     'model': name
            # }
            # args = SimpleNamespace(**d)
            # base_trainer(args)

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

def best_threshold(results):
    """
    Finds the best threshold based on total return and returns the full stats.
    """
    if results is None:
        return None, None
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

def meta_trainer_run(ticker, trainer, quantization_level, interval):
    v3_config = {
        'conviction_threshold': 0.4,  # 2/4 models agreement
        'risk_circuit_breaker': 0.035, # 3.5% structural risk limit
        'volatility_dampening': 50,    # Adaptive learning speed
        'learning_rate_init': 0.025,    # Base SGD step size,
        'loser_threshold': 0.70
    }

    data = mkt.import_market_data(ticker, interval)        
    base_models = load_base_models(ticker)
    magnitude_thresholds = [5e-5, 0.001, 0.003, 0.005, 0.010, 0.015, 0.020, 0.030]
    
    # Executing with V3 Production Config
    test_data, backtest_results, meta_model = trainer(
        base_models, data, magnitude_thresholds, v3_config
    )

    if meta_model is None:
        return None, None, None, None
    
    best_hist, best_stats = best_threshold(backtest_results)
    return best_hist, test_data, best_stats, meta_model

def meta_trainer(args):
    ticker = args.ticker.upper()
    backtest_name = args.backtest
    quantization_level = args.quantization_level
    interval = args.interval

    backest_names ={
        'sgd': train_and_test_sgd_ensemble,
        'linear': train_and_test_linear_ensemble,
        'combined': train_and_test_combined_ensemble
    }

    try:
        best_hist, _, best_stats, _ = meta_trainer_run(ticker, backest_names[backtest_name], quantization_level, interval)
    except ValueError as cause:
        print(f"Value error {str(cause)} ocurred for {ticker}")
        exit(-1)

    output_file = os.path.join(os.getcwd(), "test-results", f"report-backtest.json")
    mode = 'a' if os.path.exists(output_file) else 'w'
    with open(output_file, mode) as f:
        
        report_data = {
            "history":  best_hist,
            "stats":  best_stats
        }

        if not best_hist is None:
            if mode == 'w':
                print("{", file=f) 
                print(f"\"{ticker}\": {json.dumps(report_data)}", file=f) 
            else:
                print(f", \"{ticker}\": {json.dumps(report_data)}", file=f)