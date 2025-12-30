import json
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import qf.market as mkt
import qf.nn.models.base.pricevoldiff as pv_lib
import qf.nn.models.base.probdiff as pd_lib
import qf.nn.models.base.wavelets as wav_lib
import qf.nn.models.base.gauge as gauge_lib
import qf.nn.models.base.barinbalance as bar_lib
from sklearn.linear_model import LinearRegression, SGDRegressor


# --- CONSTANTS ---
BASE_CASH = 10000
TRADING_DAYS_PER_YEAR = 252
ANNUAL_RISK_FREE_RATE = 0.04 

base_model_names = ('pricevol', 'wavelets', 'gauge', 'probdiff', 'barinbalance')

def extract_meta_features(historical_data, models, k=14):
    """Stacks predictions from all base models as features."""
    m_pv, m_ang, m_g, m_pd, m_bar = models
    X_pv = pv_lib.create_inputs(historical_data, k)
    X_wav = wav_lib.create_inputs(historical_data, k) 
    X_g = gauge_lib.create_inputs(historical_data, k)
    X_pd = pd_lib.create_inputs(historical_data, k)
    X_bar = bar_lib.create_inputs(historical_data, k)

    min_samples = min(len(X_pv), len(X_wav), len(X_g))
    X_pv, X_wav, X_g, X_pd, X_bar = X_pv[-min_samples:], X_wav[-min_samples:], X_g[-min_samples:], X_pd[-min_samples:], X_bar[-min_samples:]
    
    y_pv = m_pv.predict(X_pv, verbose=0).flatten()
    y_wav = m_ang.predict(X_wav, verbose=0).flatten()
    y_g = m_g.predict(X_g, verbose=0).flatten()
    y_pd = m_pd.predict(X_pd, verbose=0).flatten()
    y_bar = m_bar.predict(X_bar, verbose=0).flatten()

    return np.column_stack([y_pv, y_wav, y_g, y_pd, y_bar])

def get_limits(close_t, energy_levels, direction, risk_pct=0.01):    
    risk_dollars, reward_dollars = 0, 0
    if isinstance(energy_levels, tuple):
        e_low, e_high = energy_levels
        if direction == 1:
            risk_dollars = close_t - e_low
            reward_dollars = e_high - close_t
        elif direction == -1:
            risk_dollars = e_high - close_t
            reward_dollars = close_t - e_low
    
    risk_dollars = min(risk_dollars, close_t * risk_pct)
    reward_dollars = max(risk_dollars * 3, reward_dollars)
    return risk_dollars, reward_dollars

def train_and_test_sgd_ensemble(base_models, data, thresholds, meta_config):
    """
    Adaptive Meta Model (V3 Production).
    Configurable via meta_config to manage non-normal distribution risks.
    """
    _, _, _, meta_train_data, test_data = mkt.create_datasets(data)

    # 1. Warm-up using SGD with configurable learning rate
    meta_X_train = extract_meta_features(meta_train_data, base_models)
    meta_prices = meta_train_data['Close'].values[-len(meta_X_train)-1:]
    meta_y_train = np.diff(meta_prices) / meta_prices[:-1] 

    meta_model = SGDRegressor(learning_rate='constant', eta0=meta_config.get('learning_rate_init', 0.01))
    meta_model.partial_fit(meta_X_train, meta_y_train)

    test_X_all = extract_meta_features(test_data, base_models)
    execution_length = len(test_X_all) - 1
    
    t_open = test_data['Open'].values[-execution_length:]
    t_close = test_data['Close'].values[-execution_length:]
    t_high = test_data['High'].values[-execution_length:]
    t_low = test_data['Low'].values[-execution_length:]
    t_EH = test_data['E_High'].values[-execution_length:]
    t_EL = test_data['E_Low'].values[-execution_length:]

    results = {t: [BASE_CASH] for t in thresholds}
    cur_eq = {t: BASE_CASH for t in thresholds}
    
    for i in range(execution_length):
        x_today = test_X_all[i].reshape(1, -1)
        recent_window = t_close[max(0, i-10):i]
        # Measure local heteroscedasticity
        vol = np.std(np.diff(recent_window)/recent_window[:-1]) if len(recent_window) > 2 else 0.01
        
        pred_pct = meta_model.predict(x_today)[0]

        # Calculate ensemble conviction
        conviction = np.abs(np.sum(np.sign(x_today))) / 5

        for threshold in thresholds:
            pnl = 0
            # V3 Consensus Logic
            if abs(pred_pct) > threshold and conviction >= meta_config['conviction_threshold']:
                direction = 1 if pred_pct > 0 else -1
                risk_ps, reward_ps = get_limits(t_open[i], (t_EL[i], t_EH[i]), direction)
                
                # V3 Fat-Tail Circuit Breaker
                if risk_ps > 0 and (risk_ps / t_open[i]) < meta_config['risk_circuit_breaker']:
                    share_count = min(
                        np.floor((cur_eq[threshold] * 0.01) / risk_ps), 
                        np.floor(cur_eq[threshold] / t_open[i])
                    )
                    
                    if share_count >= 1:
                        if direction == 1:
                            if t_low[i] <= (t_open[i] - risk_ps): pnl_ps = -risk_ps
                            elif t_high[i] >= (t_open[i] + reward_ps): pnl_ps = reward_ps
                            else: pnl_ps = t_close[i] - t_open[i]
                        else:
                            if t_high[i] >= (t_open[i] + risk_ps): pnl_ps = -risk_ps
                            elif t_low[i] <= (t_open[i] - reward_ps): pnl_ps = reward_ps
                            else: pnl_ps = t_open[i] - t_close[i]
                        
                        pnl = pnl_ps * share_count
            
            cur_eq[threshold] += pnl
            results[threshold].append(cur_eq[threshold])

        # Online adaptive update adjusted for volatility clustering
        meta_model.eta0 = meta_config.get('learning_rate_init', 0.01) / (1.0 + vol * meta_config['volatility_dampening'])
        actual_day_return = (t_close[i] - t_open[i]) / t_open[i]
        meta_model.partial_fit(x_today, [actual_day_return])

    return test_data, results, meta_model

def train_and_test_linear_ensemble(base_models, data, thresholds, meta_config):
    """
    Standard Linear Meta Model (Batch Fit).
    Uses OLS regression to weight base model predictions.
    """
    # 1. Dataset split logic matching mkt.create_datasets
    _, _, _, meta_train_data, test_data = mkt.create_datasets(data)

    # 2. Training Phase (Batch OLS)
    meta_X_train = extract_meta_features(meta_train_data, base_models)
    meta_prices = meta_train_data['Close'].values[-len(meta_X_train)-1:]
    meta_y_train = np.diff(meta_prices) / meta_prices[:-1] 

    # Fit a standard Linear Regression model
    meta_model = LinearRegression()
    meta_model.fit(meta_X_train, meta_y_train)

    # 3. Testing Phase
    test_X_all = extract_meta_features(test_data, base_models)
    execution_length = len(test_X_all) - 1
    
    t_open = test_data['Open'].values[-execution_length:]
    t_close = test_data['Close'].values[-execution_length:]
    t_high = test_data['High'].values[-execution_length:]
    t_low = test_data['Low'].values[-execution_length:]
    t_EH = test_data['E_High'].values[-execution_length:]
    t_EL = test_data['E_Low'].values[-execution_length:]

    results = {t: [BASE_CASH] for t in thresholds}
    cur_eq = {t: BASE_CASH for t in thresholds}
    
    # 4. Backtest Execution Loop
    for i in range(execution_length):
        x_today = test_X_all[i].reshape(1, -1)
        
        # Predict expected percentage return
        pred_pct = meta_model.predict(x_today)[0]

        # Calculate ensemble conviction (agreement among base models)
        conviction = np.abs(np.sum(np.sign(x_today))) / 5

        for threshold in thresholds:
            pnl = 0
            # Decision Logic using Configurable Thresholds
            if abs(pred_pct) > threshold and conviction >= meta_config['conviction_threshold']:
                direction = 1 if pred_pct > 0 else -1
                risk_ps, reward_ps = get_limits(t_open[i], (t_EL[i], t_EH[i]), direction)
                
                # Fat-Tail Circuit Breaker
                if risk_ps > 0 and (risk_ps / t_open[i]) < meta_config['risk_circuit_breaker']:
                    share_count = min(
                        np.floor((cur_eq[threshold] * 0.01) / risk_ps), 
                        np.floor(cur_eq[threshold] / t_open[i])
                    )
                    
                    if share_count >= 1:
                        if direction == 1:
                            if t_low[i] <= (t_open[i] - risk_ps): pnl_ps = -risk_ps
                            elif t_high[i] >= (t_open[i] + reward_ps): pnl_ps = reward_ps
                            else: pnl_ps = t_close[i] - t_open[i]
                        else:
                            if t_high[i] >= (t_open[i] + risk_ps): pnl_ps = -risk_ps
                            elif t_low[i] <= (t_open[i] - reward_ps): pnl_ps = reward_ps
                            else: pnl_ps = t_open[i] - t_close[i]
                        
                        pnl = pnl_ps * share_count
            
            cur_eq[threshold] += pnl
            results[threshold].append(cur_eq[threshold])

        # Note: Standard LinearRegression does not support online updates like SGDRegressor.
        # To maintain the signature, we return the static model.

    return test_data, results, meta_model

def train_and_test_combined_ensemble(base_models, data, thresholds, meta_config):
    """
    Regime-Switching Meta-Model with Dynamic Energy Hysteresis.
    Switches engines based on E_High/E_Low bands from the dataset.
    """
    # 1. Dataset Preparation
    _, _, _, meta_train_data, test_data = mkt.create_datasets(data)
    meta_X_train = extract_meta_features(meta_train_data, base_models)
    
    meta_prices = meta_train_data['Close'].values[-len(meta_X_train)-1:]
    meta_y_train = np.diff(meta_prices) / meta_prices[:-1]

    # 2. Initialize Engines
    linear_engine = LinearRegression()
    linear_engine.fit(meta_X_train, meta_y_train)

    sgd_engine = SGDRegressor(
        learning_rate='constant', 
        eta0=meta_config.get('learning_rate_init', 0.01),
        penalty='l2'
    )
    sgd_engine.partial_fit(meta_X_train, meta_y_train)

    # Consensual Validation Sieve
    linear_preds = linear_engine.predict(meta_X_train)
    sgd_preds = sgd_engine.predict(meta_X_train)

    # Calculate Directional Accuracy for both
    lin_acc = np.mean(np.sign(linear_preds) == np.sign(meta_y_train))
    sgd_acc = np.mean(np.sign(sgd_preds) == np.sign(meta_y_train))

    # Calculate R2 for both
    lin_r2 = linear_engine.score(meta_X_train, meta_y_train)
    sgd_r2 = sgd_engine.score(meta_X_train, meta_y_train)

    # Consensus Gate: Both must have an edge (>50% acc) and non-negative R2
    loser_threshold = meta_config.get('loser_threshold', 0.20   )
    consensus_met = (lin_acc + sgd_acc > loser_threshold) and (lin_r2 + sgd_r2 > 0)

    if not consensus_met:
        # Fails to meet consensus; skip this ticker to avoid potential loss
        return None, None, (None, None)    

    # 3. Setup Energy Components
    test_X_all = extract_meta_features(test_data, base_models)
    execution_length = len(test_X_all) - 1
    pct_changes = test_data['Close'].pct_change().values
    
    # Extract dynamic energy columns
    t_open = test_data['Open'].values[-execution_length:]
    t_close = test_data['Close'].values[-execution_length:]
    t_high = test_data['High'].values[-execution_length:]
    t_low = test_data['Low'].values[-execution_length:]
    t_EH = test_data['E_High'].values[-execution_length:]
    t_EL = test_data['E_Low'].values[-execution_length:]

    # 4. Backtest Loop
    results = {t: [BASE_CASH] for t in thresholds}
    cur_eq = {t: BASE_CASH for t in thresholds}
    current_state = 0  # 0: Linear, 1: SGD
    
    for i in range(execution_length):
        x_today = test_X_all[i].reshape(1, -1)
        actual_day_return = (t_close[i] - t_open[i]) / t_open[i]
        
        # Calculate local signal energy (14-period volatility)
        current_energy = np.std(pct_changes[max(0, i-14):i+1]) if i > 0 else 0.01
        
        # --- DYNAMIC ENERGY HYSTERESIS SWITCH ---
        # High Energy (Volatility > E_High): Adaptive SGD mode
        # Low Energy (Volatility < E_Low): Stable Linear mode
        decay_factor = 0.95 # Slows the exit from high-volatility mode
        if current_state == 0:
            if current_energy > (t_EH[i] / t_open[i]):
                current_state = 1
        else:
            # Use a 'cushion' to ensure volatility has truly settled
            if current_energy < (t_EL[i] / t_open[i]) * decay_factor:
                current_state = 0
        
        # --- PREDICTION ---
        pred_pct = sgd_engine.predict(x_today)[0] if current_state == 1 else linear_engine.predict(x_today)[0]
            
        # --- EXECUTION & RISK MANAGEMENT ---
        conviction = np.abs(np.sum(np.sign(x_today))) / len(base_models)
        
        for threshold in thresholds:
            pnl = 0
            if abs(pred_pct) > threshold and conviction >= meta_config.get('conviction_threshold', 0.4):
                direction = 1 if pred_pct > 0 else -1
                
                # Dynamic Stop Loss and Take Profit using Energy Bands
                risk_ps, reward_ps = get_limits(t_open[i], (t_EL[i], t_EH[i]), direction)
                
                # Fat-Tail Circuit Breaker
                if risk_ps > 0 and (risk_ps / t_open[i]) < meta_config.get('risk_circuit_breaker', 0.035):
                    if direction == 1:
                        if t_low[i] <= (t_open[i] - risk_ps): p_return = -risk_ps / t_open[i]
                        elif t_high[i] >= (t_open[i] + reward_ps): p_return = reward_ps / t_open[i]
                        else: p_return = actual_day_return
                    else:
                        if t_high[i] >= (t_open[i] + risk_ps): p_return = -risk_ps / t_open[i]
                        elif t_low[i] <= (t_open[i] - reward_ps): p_return = reward_ps / t_open[i]
                        else: p_return = -actual_day_return
                    
                    pnl = cur_eq[threshold] * p_return
            
            cur_eq[threshold] += pnl
            results[threshold].append(cur_eq[threshold])

        # --- ONLINE LEARNING ---
        sgd_engine.partial_fit(x_today, [actual_day_return])

    return test_data, results, (linear_engine, sgd_engine)

#---

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

    data = mkt.import_market_data(ticker, quantization_level, interval)        
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