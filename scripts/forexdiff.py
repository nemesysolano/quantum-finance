from multiprocessing import Pool
import numpy as np
import qf.market as mkt
import qf.nn.fracdiff as frac
import sys, os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import qf.nn.fracdiff as frac

def report_goodness_of_fit(y_true, y_pred, pred_deltas, anchors):
    """
    Revised to correctly align predicted deltas with the actual bar result.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    pred_deltas = np.array(pred_deltas)
    anchors = np.array(anchors)

    # 1. Calculate Actual Move for the bar: (Entry -> Exit)
    # This matches the pred_deltas which were for this specific interval.
    actual_deltas = (y_true - anchors) / (np.abs(anchors) + np.abs(y_true))

    delta_mae = mean_absolute_error(actual_deltas, pred_deltas)
    sign_match = np.mean(np.sign(actual_deltas) == np.sign(pred_deltas))

    # Traditional price-space metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return {"mae": mae, "rmse": rmse, "sign_match": sign_match, "delta_mae": delta_mae}

def create_targets(historical_data, k):
    """Returns the actual closing prices for the target steps (t+1)."""
    # .flatten() prevents (N, 1) shape issues from DataFrame slicing
    return historical_data['Close'].values[k+1:].flatten()

def create_inputs(historical_data, k):
    """Aligns features, indicators, and intra-day path data for zero-leakage."""
    c = historical_data['Close'].values.flatten()
    h = historical_data['High'].values.flatten()
    l = historical_data['Low'].values.flatten()
    S = historical_data['Ö'].values.flatten()
    Eh = historical_data['E_High'].values.flatten()
    El = historical_data['E_Low'].values.flatten()
    
    def delta_p(a, b): return (b - a) / (abs(a) + abs(b))
    
    # 1. Transform into stationary Δ% series
    diffs = np.array([delta_p(c[i-1], c[i]) for i in range(1, len(c))])
    
    # X_all[j] contains [diffs[j], ..., diffs[j+k-1]]
    # For j=0, it ends at diffs[k-1] (the change from c[k-1] to c[k]).
    # This is the correct feature set to predict c[k+1].
    X_all = np.array([diffs[i:i+k] for i in range(len(diffs)-k)])
    
    # FIX: Use X_all[:-1] so X_tar[i] ends at diffs[k+i-1] (data known at time t)
    X_tar = X_all[:-1] 
    
    # 3. Knowledge at time t: Indicators and the entry anchor price
    anchors = S[k:-1] 
    raw_anchors = c[k:-1]
    
    # 4. Future path data (t+1) for SL/TP verification
    h_target = h[k+1:]
    l_target = l[k+1:]
    energy_high_target = Eh[k+1:]
    energy_low_target = El[k+1:]
    
    # Return the full diffs array to allow walk-forward fitting of 'd'
    return diffs, X_tar, anchors, raw_anchors, h_target, l_target, energy_high_target, energy_low_target


def estimate_y(diffs, X_tar, anchors, k):
    """Walk-forward estimation of fractional predictions with no target leakage."""
    y_preds, d_vals, pred_deltas = [], [], []
    for i in range(len(X_tar)):
        # The feature vector X_tar[i] already ends at diffs[k+i-1] (the diff ending at time t).
        # We fit 'd' using the sequence of diffs prior to this.
        history_for_d = diffs[:k+i-1]
        target_for_d = diffs[k+i-1]
        
        # d is calculated using the relationship between past moves and the move that just finished
        d_hat = frac.perform_ols_and_fit(history_for_d, target_for_d, k)
        
        # Now predict the NEXT move (t to t+1) using the fitted d
        pred, pred_delta = frac.predict_next_price(X_tar[i], d_hat, k, anchors[i])
        y_preds.append(pred)
        d_vals.append(d_hat)
        pred_deltas.append(pred_delta)
    return np.array(y_preds), d_vals, pred_deltas

def simulate_trading_no_hedge(y_actual, anchors, y_preds, h_target, l_target, e_high, e_low, pred_deltas, atr_history, initial_cap=500):
    """
    Non-hedged version incorporating 'Strong Protection' logic:
    1. Regime Detection (Entropy + Volatility Spikes)
    2. Dynamic Delta Strength tracking
    3. Confidence-based SL/TP scaling
    4. Energy Level safety rails
    """
    cash = initial_cap
    equity_curve = [initial_cap]
    longs, shorts = 0, 0
    winner_longs, winner_shorts = 0, 0
    loser_longs, loser_shorts = 0, 0
    
    # Trackers for the 'Strong Protection' logic
    delta_strength = 9e-6
    stop_loss_scaler_min = 1/5
    stop_loss_scaler_max = 1/2

    for i in range(len(y_preds)):
        price_entry = anchors[i]
        p_delta = pred_deltas[i]
        atr = atr_history[i]
        n_high, n_low, n_close = h_target[i], l_target[i], y_actual[i]
        energy_h, energy_l = e_high[i], e_low[i]

        # --- REGIME DETECTION (PROTECTION 1) ---
        # Skip trades in unstable or overly volatile environments
        price_window = anchors[max(0, i-20):i+1]
        entropy = frac.get_shannon_entropy(price_window)
        vol_spike = frac.is_binary_event(atr_history[:i+1])

        is_stable = (entropy < 0.85) and (not vol_spike) and (np.abs(p_delta) >= delta_strength)
        
        # Smoothly update the strength threshold for the next bar
        delta_strength = 0.25 * delta_strength + 0.75 * np.abs(p_delta)
        
        if not is_stable:
            equity_curve.append(cash)
            continue

        # --- CONFIDENCE SCALING (PROTECTION 2) ---
        # Scale the stop loss based on the magnitude of the predicted move
        conf_shift = np.sign(p_delta) * np.power(10, np.floor(np.log10(delta_strength + 1e-9)))
        
        stop_loss_scaler = np.clip(
            stop_loss_scaler_min * (1 + conf_shift), 
            stop_loss_scaler_min, 
            stop_loss_scaler_max
        )
        
        # Distance to the predicted target
        tp_dist_base = abs(y_preds[i] - price_entry)
        contracts = max(1, int(cash // 2500)) 
        pos_size = int(cash // price_entry) * contracts

        if p_delta > 0: # Long conviction
            longs += 1
            # Primary target is predicted price, capped by Energy High
            tp_p = min(price_entry + tp_dist_base, energy_h)
            # Stop loss is scaled by confidence, but cannot be lower than Energy Low
            sl_p = max(price_entry - (tp_dist_base * stop_loss_scaler), energy_l)
            
            if n_low <= sl_p:
                res = pos_size * (sl_p - price_entry)
                loser_longs += 1
            elif n_high >= tp_p:
                res = pos_size * (tp_p - price_entry)
                winner_longs += 1
            else:
                res = pos_size * (n_close - price_entry)
                if res > 0: winner_longs += 1
                elif res < 0: loser_longs += 1
                
        else: # Short conviction
            shorts += 1
            tp_p = max(price_entry - tp_dist_base, energy_l)
            sl_p = min(price_entry + (tp_dist_base * stop_loss_scaler), energy_h)
            
            if n_high >= sl_p:
                res = pos_size * (price_entry - sl_p)
                loser_shorts += 1
            elif n_low <= tp_p:
                res = pos_size * (price_entry - tp_p)
                winner_shorts += 1
            else:
                res = pos_size * (price_entry - n_close)
                if res > 0: winner_shorts += 1
                elif res < 0: loser_shorts += 1

        cash += res
        equity_curve.append(cash)

    # Return matches the required interface
    return (
        equity_curve, cash, longs, shorts, winner_longs, winner_shorts, loser_longs, loser_shorts,
        0, 0, 0, 0, 0, 0, 0, 0 # Placeholders for the extended stats
    )

def simulate_trading_hedged(y_actual, anchors, y_preds, h_target, l_target, e_high, e_low, pred_deltas, atr_history, initial_cap=500):
    """
    Defensive Strategy optimized for <7% Max Drawdown.
    - Tight SL-to-TP ratio.
    - Reduced position sizing (De-leveraging).
    - Asymmetric 'Quick-Exit' Hedges.
    """
    cash = initial_cap
    equity_curve = [initial_cap]
    longs, shorts = 0, 0
    winner_longs, winner_shorts = 0, 0
    loser_longs, loser_shorts = 0, 0
    
    max_buy_win_val, max_buy_loss_val = 0, 0
    max_sell_win_val, max_sell_loss_val = 0, 0
    largest_hold_buy_winner_time, largest_hold_buy_loser_time = 0, 0
    largest_hold_sell_winner_time, largest_hold_sell_loser_time = 0, 0

    conviction_threshold = np.median(np.abs(pred_deltas)) if len(pred_deltas) > 0 else 0

    for i in range(len(y_preds)):
        price_entry = anchors[i]
        p_delta = pred_deltas[i]
        atr = atr_history[i]
        n_high, n_low, n_close = h_target[i], l_target[i], y_actual[i]

        # 1. DE-LEVERAGING: Higher divisor = lower drawdown. 
        # Moving from 1000/1500 to 2500 to dampen equity swings.
        contracts = max(1, int(cash // 2500)) 

        # Conviction Weighting (Requested: 75/25, 65/35)
        if abs(p_delta) > conviction_threshold:
            primary_w, hedge_w = 0.75, 0.25
        else:
            primary_w, hedge_w = 0.65, 0.35

        p_side, h_side = ('long', 'short') if p_delta > 0 else ('short', 'long')
        cap_p, cap_h = cash * primary_w, cash * hedge_w
        
        # 2. DEFENSIVE RISK SCALING
        tp_dist = atr * 1.2       # Let winners run slightly more...
        sl_primary_mult = 0.8     # ...but cut Primary losses MUCH faster (0.8x vs old 1.5x)
        sl_hedge_mult = 0.4       # Hedge is purely for 'flash crash' protection
        
        # --- PRIMARY POSITION ---
        pos_p = int(cap_p // price_entry) * contracts
        if p_side == 'long':
            longs += 1
            tp_p, sl_p = price_entry + tp_dist, price_entry - (tp_dist * sl_primary_mult)
            if n_low <= sl_p:
                res_p = pos_p * (sl_p - price_entry)
                loser_longs += 1
            elif n_high >= tp_p:
                res_p = pos_p * (tp_p - price_entry)
                winner_longs += 1
            else:
                res_p = pos_p * (n_close - price_entry)
                if res_p > 0: winner_longs += 1
                elif res_p < 0: loser_longs += 1
        else: # Short
            shorts += 1
            tp_p, sl_p = price_entry - tp_dist, price_entry + (tp_dist * sl_primary_mult)
            if n_high >= sl_p:
                res_p = pos_p * (price_entry - sl_p)
                loser_shorts += 1
            elif n_low <= tp_p:
                res_p = pos_p * (price_entry - tp_p)
                winner_shorts += 1
            else:
                res_p = pos_p * (price_entry - n_close)
                if res_p > 0: winner_shorts += 1
                elif res_p < 0: loser_shorts += 1

        # --- HEDGE POSITION ---
        pos_h = int(cap_h // price_entry) * contracts
        if h_side == 'long':
            longs += 1
            tp_h, sl_h = price_entry + tp_dist, price_entry - (tp_dist * sl_hedge_mult)
            if n_low <= sl_h:
                res_h = pos_h * (sl_h - price_entry)
                loser_longs += 1
            elif n_high >= tp_h:
                res_h = pos_h * (tp_h - price_entry)
                winner_longs += 1
            else:
                res_h = pos_h * (n_close - price_entry)
                if res_h > 0: winner_longs += 1
                elif res_h < 0: loser_longs += 1
        else: # Short
            shorts += 1
            tp_h, sl_h = price_entry - tp_dist, price_entry + (tp_dist * sl_hedge_mult)
            if n_high >= sl_h:
                res_h = pos_h * (price_entry - sl_h)
                loser_shorts += 1
            elif n_low <= tp_h:
                res_h = pos_h * (price_entry - tp_h)
                winner_shorts += 1
            else:
                res_h = pos_h * (price_entry - n_close)
                if res_h > 0: winner_shorts += 1
                elif res_h < 0: loser_shorts += 1

        # 3. STATS RECONCILIATION
        res_long = res_p if p_side == 'long' else res_h
        res_short = res_p if p_side == 'short' else res_h
        
        if res_long > max_buy_win_val: max_buy_win_val, largest_hold_buy_winner_time = res_long, i
        if res_long < max_buy_loss_val: max_buy_loss_val, largest_hold_buy_loser_time = res_long, i
        if res_short > max_sell_win_val: max_sell_win_val, largest_hold_sell_winner_time = res_short, i
        if res_short < max_sell_loss_val: max_sell_loss_val, largest_hold_sell_loser_time = res_short, i

        cash += (res_p + res_h)
        equity_curve.append(cash)

    return (
        equity_curve, cash, longs, shorts, winner_longs, winner_shorts, loser_longs, loser_shorts, 
        max_buy_win_val, max_buy_loss_val, max_sell_win_val, max_sell_loss_val,
        largest_hold_buy_winner_time, largest_hold_buy_loser_time, largest_hold_sell_winner_time, largest_hold_sell_loser_time
    )

def create_backtest_stats(
        ticker,
        equity_curve, final_capital, long_trades, short_trades, winner_longs, winner_shorts, loser_longs, loser_shorts, 
        max_buy_win_val, max_buy_loss_val, max_sell_win_val, max_sell_loss_val,
        largest_hold_buy_winner_time, largest_hold_buy_loser_time, largest_hold_sell_winner_time, largest_hold_sell_loser_time
    ):
    # Ensure 1D array even if input is a list of arrays or 2D matrix
    equity_array = np.ravel(equity_curve)
    # If final_capital was an array due to the previous bug, take the first value
    f_cap = float(np.ravel(final_capital)[0]) if np.ndim(final_capital) > 0 else final_capital
    
    initial_capital = equity_array[0]
    total_return_pct = (f_cap - initial_capital) / initial_capital
    
    # Now np.diff will produce a simple 1D vector
    returns = np.diff(equity_array) / equity_array[:-1]
    
    # Standard deviation of returns (volatility)
    volatility = np.std(returns) if len(returns) > 0 else 0
    
    # Sharpe Ratio (Assuming 0 risk-free rate for simplicity)
    # Annualization factor depends on the timeframe of your data (e.g., np.sqrt(252))
    sharpe_ratio = (np.mean(returns) / volatility) if volatility != 0 else 0

    # 3. Drawdown Analysis
    # Peak equity reached up to each point in time
    running_max = np.maximum.accumulate(equity_array)
    drawdowns = (equity_array - running_max) / running_max
    max_drawdown = np.min(drawdowns)

    # 4. Summary Statistics
    stats = {
        "Ticker": ticker,
        "Initial Capital": initial_capital,
        "Final Capital": final_capital,
        "Total Return (%)": total_return_pct * 100,
        "Max Drawdown (%)": max_drawdown * 100,
        "Volatility (per step)": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Number of Steps": len(equity_array),
        "Peak Equity": np.max(equity_array),
        "Final Drawdown (%)": drawdowns[-1] * 100,
        "Long Trades": long_trades,
        "Short Trades": short_trades,
        "Winner Longs": winner_longs,
        "Winner Shorts": winner_shorts,
        "Loser Longs": loser_longs,
        "Loser Shorts": loser_shorts,
        "Largest Buy Winner": max_buy_win_val,
        "Largest Buy Loser": max_buy_loss_val,
        "Largest Sell Winner": max_sell_win_val,
        "Largest Sell Loser": max_sell_loss_val,
        "Largest Hold Buy Winner": largest_hold_buy_winner_time,
        "Largest Hold Buy Loser": largest_hold_buy_loser_time,
        "Largest Hold Sell Winner": largest_hold_sell_winner_time,
        "Largest Hold Sell Loser": largest_hold_sell_loser_time
    }
    return stats

def back_test(params):
    (k, ticker, quantization_level, interval, trader_name) = params
    trader = traders[trader_name]
    # try:
    data = mkt.import_market_data(ticker, quantization_level, interval)
        
    # Generate Inputs and Targets
    X_est, X_tar, S_tar, anchors, h_tar, l_tar, energy_high_target, energy_low_target = create_inputs(data, k)
    y_actual = create_targets(data, k)
    # Calculate ATR history for the whole series
    atr_history = frac.get_atr(y_actual, window=14)

    # Forecasting
    y_preds, _, pred_deltas = estimate_y(X_est, X_tar, anchors, k)
    
    # Align to the length of predictions
    n = len(y_preds)
    assert len(pred_deltas) == n
    assert len(y_actual) == n + 1

    goodness = report_goodness_of_fit(
        y_actual[:n], 
        y_preds, 
        pred_deltas, 
        anchors[:n] # Pass anchors for correct return calculation
    )
    
    # FIX: Remove d_history and align h_tar/l_tar correctly
    # Inside back_test(params):
    (
        equity, final, longs, shorts, winner_longs, winner_shorts, loser_longs, loser_shorts, 
        max_buy_win_val, max_buy_loss_val, max_sell_win_val, max_sell_loss_val,
        largest_hold_buy_winner_time, largest_hold_buy_loser_time, largest_hold_sell_winner_time, largest_hold_sell_loser_time                
    ) = trader(
        y_actual[:n], 
        anchors[:n], # REPLACED S_tar with anchors
        y_preds, 
        h_tar[:n], 
        l_tar[:n],
        energy_high_target[:n],
        energy_low_target[:n],
        pred_deltas,
        atr_history
    )
    # y_actual, S_tar, y_preds, h_target, l_target, e_high, e_low, pred_deltas, atr_history, initial_cap=10000

    stats = create_backtest_stats(
        ticker,
        equity, final, longs, shorts, winner_longs, winner_shorts, loser_longs, loser_shorts, 
        max_buy_win_val, max_buy_loss_val, max_sell_win_val, max_sell_loss_val,
        largest_hold_buy_winner_time, largest_hold_buy_loser_time, largest_hold_sell_winner_time, largest_hold_sell_loser_time                
    )
    print(f"Finished backtesting for {ticker}")
    return {'stats': stats, 'goodness': goodness}
    # except Exception as cause:
    #     print(f"failed to backtest {ticker} cause=f{cause}")
    #     return None
    
traders = {    
    'no-hedge': simulate_trading_no_hedge,
    'hedge': simulate_trading_hedged
}

if __name__ == '__main__':
    tickers_file = sys.argv[1]    
    trader = sys.argv[2] if len(sys.argv) > 2 else 'no-hedge'
    quantization_level = float(sys.argv[3]) if len(sys.argv) > 3 else 1e+3
    interval = sys.argv[4] if len(sys.argv) > 4 else '1d'
    k = np.clip(int(sys.argv[5]), 14, 30) if len(sys.argv) > 5 else 14
    
    
    tickers = [(k, ticker, quantization_level, interval, trader) for ticker in np.loadtxt(tickers_file, dtype=str)]
    with Pool(processes=4) as pool:
        results = list(map(back_test, tickers))

    output_file = os.path.join(os.getcwd(), "test-results", f"report-{trader}-forex.csv")
    os.remove(output_file) if os.path.exists(output_file) else None
    with open(output_file, 'w') as f: #  {"mae": mae, "rmse": rmse, "r2": r2, "sign_match": sign_match}
        print("Ticker,Initial Capital,Final Capital,Total Return (%),Max Drawdown (%),Volatility (per step),Sharpe Ratio,Number of Steps,Peak Equity,Final Drawdown,Long Trades,Short Trades,Winner Longs,Winner Shorts,Loser Longs,Loser Shorts,Delta MAE,Largest Buy Winner,Largest Buy Loser,Largest Sell Winner,Largest Sell Loser,Largest Hold Buy Winner,Largest Hold Buy Loser,Largest Hold Sell Winner,Largest Hold Sell Loser", file=f)    
        for result in results:    
            if result is None:
                continue            
            stats = result['stats']
            goodness = result['goodness']
            print(f"{stats['Ticker']},{stats['Initial Capital']:.2f},{stats['Final Capital']:.2f},{stats['Total Return (%)']:.2f},{stats['Max Drawdown (%)']:.2f},{stats['Volatility (per step)']:.4f},{stats['Sharpe Ratio']:.4f},{stats['Number of Steps']},{stats['Peak Equity']:.2f},{stats['Final Drawdown (%)']:.2f},{stats['Long Trades']},{stats['Short Trades']},{stats['Winner Longs']},{stats['Winner Shorts']},{stats['Loser Longs']},{stats['Loser Shorts']},{goodness['delta_mae']:.4f},{stats['Largest Buy Winner']},{stats['Largest Buy Loser']},{stats['Largest Sell Winner']},{stats['Largest Sell Loser']},{stats['Largest Hold Buy Winner']},{stats['Largest Hold Buy Loser']},{stats['Largest Hold Sell Winner']},{stats['Largest Hold Sell Loser']}", file=f)