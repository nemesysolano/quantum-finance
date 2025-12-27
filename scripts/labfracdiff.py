from multiprocessing import Pool
import numpy as np
import qf.market as mkt
import qf.nn.fracdiff as frac
import sys, os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def report_goodness_of_fit(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    sign_match = np.mean(np.sign(y_true) == np.sign(y_pred))
 
    return {"mae": mae, "rmse": rmse, "r2": r2, "sign_match": sign_match}

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
    Eh = historical_data['Eh'].values.flatten()
    El = historical_data['El'].values.flatten()
    
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
    S_at_t = S[k:-1] 
    raw_anchors = c[k:-1]
    
    # 4. Future path data (t+1) for SL/TP verification
    h_target = h[k+1:]
    l_target = l[k+1:]
    energy_high_target = Eh[k+1:]
    energy_low_target = El[k+1:]
    
    # Return the full diffs array to allow walk-forward fitting of 'd'
    return diffs, X_tar, S_at_t, raw_anchors, h_target, l_target, energy_high_target, energy_low_target


def estimate_y(diffs, X_tar, raw_anchors, k):
    """Walk-forward estimation of fractional predictions with no target leakage."""
    y_preds, d_vals = [], []
    for i in range(len(X_tar)):
        # The feature vector X_tar[i] already ends at diffs[k+i-1] (the diff ending at time t).
        # We fit 'd' using the sequence of diffs prior to this.
        history_for_d = diffs[:k+i-1]
        target_for_d = diffs[k+i-1]
        
        # d is calculated using the relationship between past moves and the move that just finished
        d_hat = frac.perform_ols_and_fit(history_for_d, target_for_d, k)
        
        # Now predict the NEXT move (t to t+1) using the fitted d
        pred = frac.predict_next_price(X_tar[i], d_hat, k, raw_anchors[i])
        y_preds.append(pred)
        d_vals.append(d_hat)
    return np.array(y_preds), d_vals

def simulate_trading(y_actual, S_at_t, y_preds, h_target, l_target, energy_high_target, energy_low_target, initial_cap=10000):
    cap = initial_cap
    equity_curve = [cap]
    longs, shorts = 0, 0
    
    # Start from 0 to capture the first prediction
    for i in range(len(y_preds) - 1):
        # price_now is the close at time t
        price_now = y_actual[i] 
        # y_preds[i] is the forecast for y_actual[i+1]
        expected_move = y_preds[i] - price_now
        tp_dist = abs(expected_move)
        sl_dist = 0.3 * tp_dist
        
        # Indicator at time t decides the trade for t+1
        if expected_move > 0 and S_at_t[i] < 0:
            tp_price, sl_price = price_now + tp_dist, price_now - sl_dist
            # Check high/low of the NEXT candle (i+1)
            if l_target[i+1] <= sl_price:
                cap *= (sl_price / price_now)
            elif h_target[i+1] >= tp_price:
                cap *= (tp_price / price_now)
            else:
                cap *= (y_actual[i+1] / price_now)
            longs += 1
            
        elif expected_move < 0 and S_at_t[i] > 0:
            tp_price, sl_price = price_now - tp_dist, price_now + sl_dist
            if h_target[i+1] >= sl_price:
                cap *= (price_now / sl_price)
            elif l_target[i+1] <= tp_price:
                cap *= (price_now / tp_price)
            else:
                cap *= (price_now / y_actual[i+1])
            shorts += 1
            
        equity_curve.append(cap)
    return equity_curve, cap, longs, shorts

def create_backtest_stats(ticker, equity_curve, final_capital, long_trades, short_trades):
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
        "Short Trades": short_trades
    }

    return stats

def back_test(params):
    (k, ticker) = params
    data = mkt.import_market_data(ticker)
    
    # Generate Inputs and Targets
    X_est, X_tar, S_tar, anchors, h_tar, l_tar, energy_high_target, energy_low_target = create_inputs(data, k)
    y_actual = create_targets(data, k)
    
    # Forecasting
    y_preds, d_history = estimate_y(X_est, X_tar, anchors, k)
    
    # Align to the length of predictions
    n = len(y_preds)
    goodness = report_goodness_of_fit(y_actual[:n], y_preds)
    
    # FIX: Remove d_history and align h_tar/l_tar correctly
    equity, final, longs, shorts = simulate_trading(
        y_actual[:n], 
        S_tar[:n], 
        y_preds, 
        h_tar[:n], 
        l_tar[:n]
        energy_high_target[:n],
        energy_low_target[:n]
    )
    
    stats = create_backtest_stats(ticker, equity, final, longs, shorts)
    print(f"Finished backtesting for {ticker}")
    return {'stats': stats, 'goodness': goodness}

if __name__ == '__main__':
    tickers_file = sys.argv[1]    
    k = 14
    tickers = [(k, ticker) for ticker in np.loadtxt(tickers_file, dtype=str)]
    with Pool(processes=4) as pool:
        results = pool.map(back_test, tickers)

    
    output_file = os.path.join(os.getcwd(), "test-results", f"report-labfracdiff.md")
    os.remove(output_file) if os.path.exists(output_file) else None
    with open(output_file, 'w') as f: #  {"mae": mae, "rmse": rmse, "r2": r2, "sign_match": sign_match}
        print("|Ticker|Initial Capital|Final Capital|Total Return (%)|Max Drawdown (%)|Volatility (per step)|Sharpe Ratio|Number of Steps|Peak Equity|Final Drawdown|Long Trades|Short Trades||MAE|RMSE|R2|Sign Match|", file=f)
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|", file=f)
    
        for result in results:    
            stats = result['stats']
            goodness = result['goodness']
            print(f"|{stats['Ticker']}|{stats['Initial Capital']:.2f}|{stats['Final Capital']:.2f}|{stats['Total Return (%)']:.2f}|{stats['Max Drawdown (%)']:.2f}|{stats['Volatility (per step)']:.4f}|{stats['Sharpe Ratio']:.4f}|{stats['Number of Steps']}|{stats['Peak Equity']:.2f}|{stats['Final Drawdown (%)']:.2f}|{stats['Long Trades']}|{stats['Short Trades']}||{goodness['mae']:.4f}|{goodness['rmse']:.4f}|{goodness['r2']:.4f}|{goodness['sign_match']:.4f}|", file=f)