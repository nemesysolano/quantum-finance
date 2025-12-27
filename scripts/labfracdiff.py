import numpy as np
import qf.market as mkt
import pandas as pd
import qf.nn.fracdiff as frac
import sys
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def report_goodness_of_fit(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    sign_match = np.mean(np.sign(y_true) == np.sign(y_pred))
    
    print("\n--- Sequential Goodness of Fit Report ---")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Sign Match: {sign_match:.4f}")
    print(f"Count: {len(y_true)}")

import numpy as np
import qf.nn.fracdiff as frac

def create_inputs(historical_data, k):
    c = historical_data['Close'].values.flatten()
    S = historical_data['Ö'].values.flatten()
    e_high = historical_data['E_High'].values.flatten()
    e_low = historical_data['E_Low'].values.flatten()
    
    # Bounded Percentage Difference: Δ%(a, b)
    def delta_p(a, b): return (b - a) / (abs(a) + abs(b))
    
    # Create the stationary series of 1-step serial differences
    diffs = np.array([delta_p(c[i-1], c[i]) for i in range(1, len(c))])
    X_all = np.array([diffs[i:i+k] for i in range(len(diffs)-k)])
    
    # Aligning for t+1 prediction
    X_est = X_all[:-1]   # Windows ending at t-1 (to estimate d)
    X_tar = X_all[1:]    # Windows ending at t (to predict t+1)
    
    # Metadata for the prediction step (t)
    S_tar = S[k+1:]
    e_high_tar = e_high[k+1:]
    e_low_tar = e_low[k+1:]
    raw_anchors = c[k:-1] # The price at time t used as the projection base
    
    return X_est, X_tar, S_tar, e_low_tar, e_high_tar, raw_anchors

def create_targets(historical_data, k):
    return historical_data['Close'].values[k+1:]

def estimate_y(X_est, X_tar, raw_anchors, k):
    y_preds, d_history = [], []
    for i in range(len(X_tar)):
        # Target for d-estimation is the diff that just happened at time t
        d_hat = frac.perform_ols_and_fit(X_est[i], X_tar[i][-1], k)
        pred = frac.predict_next_price(X_tar[i], d_hat, k, raw_anchors[i])
        y_preds.append(pred)
        d_history.append(d_hat)
    return np.array(y_preds), np.array(d_history)

def simulate_trading(y_actual, S_tar, y_preds, d, e_high, e_low, initial_cap=10000):
    """
    Simulates trading with dynamic Take Profit (based on prediction) 
    and Stop Loss (0.3 * TP distance).
    """
    cap = initial_cap
    equity_curve = [cap]
    longs, shorts = 0, 0
    
    for i in range(len(y_preds)):
        # price_now is the actual price at time t
        # y_actual[i] is the actual price at time t+1
        price_now = y_actual[i-1] if i > 0 else y_preds[i]/1.01 
        
        # Calculate expected distance for the forecast
        expected_move = y_preds[i] - price_now
        tp_dist = abs(expected_move)
        sl_dist = 0.3 * tp_dist
        
        # Decision Logic: Forecast direction + Gauge alignment
        if expected_move > 0 and S_tar[i] < 0:
            # LONG POSITION
            tp_price = price_now + tp_dist
            sl_price = price_now - sl_dist
            
            # Check if actual price range at t+1 hits SL or TP
            # (Note: We use y_actual as a proxy for the path)
            if y_actual[i] <= sl_price:
                # Stop Loss hit
                cap *= (sl_price / price_now)
            elif y_actual[i] >= tp_price:
                # Take Profit hit
                cap *= (tp_price / price_now)
            else:
                # Normal close at end of period
                cap *= (y_actual[i] / price_now)
            longs += 1
            
        elif expected_move < 0 and S_tar[i] > 0:
            # SHORT POSITION
            tp_price = price_now - tp_dist
            sl_price = price_now + sl_dist
            
            if y_actual[i] >= sl_price:
                # Stop Loss hit
                cap *= (price_now / sl_price)
            elif y_actual[i] <= tp_price:
                # Take Profit hit
                cap *= (price_now / tp_price)
            else:
                # Normal close at end of period
                cap *= (price_now / y_actual[i])
            shorts += 1
            
        equity_curve.append(cap)
        
    return equity_curve, cap, longs, shorts

def create_backtest_stats(equity_curve, final_capital, long_trades, short_trades):
    """
    Extracts performance statistics from the simulated equity curve.
    """
    equity_array = np.array(equity_curve)
    initial_capital = equity_array[0]
    
    # 1. Basic Return Metrics
    total_return_pct = (final_capital - initial_capital) / initial_capital
    
    # 2. Daily/Step Returns for Volatility and Sharpe
    # We use simple returns between each step in the equity curve
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
        "Short Trades": short_trades
    }

    return stats

if __name__ == '__main__':
    ticker = sys.argv[1]
    k = 14
    data = mkt.import_market_data(ticker)
    
    X_est, X_tar, S_tar, e_low_tar, e_high_tar, anchors = create_inputs(data, k)
    y_actual = create_targets(data, k)
    
    y_preds, d_vals = estimate_y(X_est, X_tar, anchors, k)
    
    # Slice to ensure perfect alignment
    n = len(y_preds)
    report_goodness_of_fit(y_actual[:n], y_preds)
    
    equity, final, longs, shorts = simulate_trading(y_actual[:n], S_tar[:n], y_preds, d_vals, e_high_tar[:n], e_low_tar[:n])
    stats = create_backtest_stats(equity, final, longs, shorts)

    output_file = os.path.join(os.getcwd(), "test-results", f"report-labfracdiff.md")
    mode = 'a' if os.path.exists(output_file) else 'w'
    with open(output_file, mode) as f:
        if mode == 'w':
            print("Ticker|Initial Capital| Final Capital|Total Return (%)|Max Drawdown (%)|Volatility (per step)|Sharpe Ratio|Number of Steps|Peak Equity|Final Drawdown|Long Trades|Short Trades", file=f)
            print("|-----|---|---|----|---|---|----|---|---|----|---|---|", file=f)
        print(f"{ticker}|{stats['Initial Capital']:.2f}|{stats['Final Capital']:.2f}|{stats['Total Return (%)']:.2f}|{stats['Max Drawdown (%)']:.2f}|{stats['Volatility (per step)']:.4f}|{stats['Sharpe Ratio']:.4f}|{stats['Number of Steps']}|{stats['Peak Equity']:.2f}|{stats['Final Drawdown (%)']:.2f}|{stats['Long Trades']}|{stats['Short Trades']}", file=f)