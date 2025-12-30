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

def simulate_trading(y_actual, anchors, y_preds, h_target, l_target, e_high, e_low, pred_deltas, atr_history, initial_cap=500):
    """
    Revised simulate_trading with corrected alignment (Shift) and Adjusted Exit logic.
    
    Args:
        y_actual: Actual close prices for prediction targets (t+1).
        anchors: Entry prices known at time t (c[k:-1]).
        y_preds: Predicted prices for t+1.
        h_target/l_target: Intraday high/low between t and t+1.
        e_high/e_low: Energy band boundaries at time t.
        pred_deltas: Predicted fractional changes.
        atr_history: Trailing volatility buffer.
    """
    cash = initial_cap
    equity_curve = [initial_cap]
    longs, shorts = 0, 0
    winner_longs, winner_shorts = 0, 0
    loser_longs, loser_shorts = 0, 0
    delta_strength = 9e-6
    contracts = 10    
    stop_loss_scaler_min = 1/5
    stop_loss_scaler_max = 1/2

    # Iterate through each prediction step
    for i in range(len(y_preds)):
        # 1. Decision context at time t (Shift Fix)
        # Entry price is the anchor price known at time t
        price_entry = anchors[i] 
        p_delta = pred_deltas[i]
        vol_buffer = atr_history[i] * (1.5 - np.abs(p_delta))
        
        # Stability filters (Regime Detection)
        price_window = anchors[max(0, i-20):i+1]
        entropy = frac.get_shannon_entropy(price_window)
        vol_spike = frac.is_binary_event(atr_history[:i+1])

        is_stable = (entropy < 0.85) and (not vol_spike) and (np.abs(p_delta) >= delta_strength) and (delta_strength < 1)
        delta_strength = 0.25*delta_strength + 0.75*np.abs(p_delta)
        if not is_stable:
            equity_curve.append(cash)
            continue
            
        # Target/Exit logic (using t+1 data for evaluation)
        # tp_dist = abs(y_preds[i] - price_entry)
        n_high, n_low, n_close = h_target[i], l_target[i], y_actual[i]        
        
        stop_loss_scaler = np.clip(
            stop_loss_scaler_min*(1 + np.sign(p_delta) * np.power(10,np.floor(np.log10(delta_strength)))), 
            stop_loss_scaler_min, 
            stop_loss_scaler_max
        )
        take_profit_scaler = 1 + np.sign(p_delta) * np.power(10,np.floor(np.log10(delta_strength)))
        tp_dist = abs(y_preds[i] - price_entry)*take_profit_scaler
        
        # --- LONG LOGIC ---
        if p_delta > 0:
            tp_price = price_entry + tp_dist
            sl_price = price_entry - stop_loss_scaler * tp_dist # min(stop_loss_scaler * tp_dist, abs(price_entry - e_low[i]) + vol_buffer)
            
            if cash >= price_entry:
                pos_shares = int(cash // price_entry)*contracts
                # Adjusted Exit: Check path between t and t+1
                if n_low <= sl_price: # Stop Loss Triggered
                    cash += pos_shares * (sl_price - price_entry)
                    loser_longs += 1
                    longs += 1

                elif n_high >= tp_price: # Take Profit Triggered
                    cash += pos_shares * (tp_price - price_entry)
                    winner_longs += 1
                    longs += 1
                

        # --- SHORT LOGIC ---
        elif p_delta < 0:
            tp_price = price_entry - tp_dist
            sl_price = price_entry + stop_loss_scaler * tp_dist #min(stop_loss_scaler * tp_dist, abs(e_high[i] - price_entry) + vol_buffer)
            
            if cash > 0:
                pos_shares = int(cash // price_entry)*contracts
                
                # Adjusted Exit: PnL = (Entry - Exit) for shorts
                if n_high >= sl_price: # Stop Loss Triggered (Price Up)
                    cash += (price_entry - sl_price) * pos_shares
                    loser_shorts += 1
                    shorts += 1

                elif n_low <= tp_price: # Take Profit Triggered (Price Down)
                    cash += (price_entry - tp_price) * pos_shares
                    winner_shorts += 1
                    shorts += 1
                

        equity_curve.append(cash)
        
    return equity_curve, cash, longs, shorts, winner_longs, winner_shorts, loser_longs, loser_shorts



def create_backtest_stats(ticker, equity_curve, final_capital, long_trades, short_trades, winner_longs, winner_shorts, loser_longs, loser_short):
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
        "Loser Shorts": loser_short
    }

    return stats

def back_test(params):
    (k, ticker, quantization_level, interval) = params
    try:
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
        equity, final, longs, shorts, winner_longs, winner_shorts, loser_longs, loser_shorts= simulate_trading(
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

        stats = create_backtest_stats(ticker, equity, final, longs, shorts, winner_longs, winner_shorts, loser_longs, loser_shorts)
        print(f"Finished backtesting for {ticker}")
        return {'stats': stats, 'goodness': goodness}
    except Exception as cause:
        print(f"failed to backtest {ticker} cause=f{cause}")
        return None
    
if __name__ == '__main__':
    tickers_file = sys.argv[1]    
    quantization_level = float(sys.argv[2]) if len(sys.argv) > 2 else 1e+3
    interval = '15m'
    k = 14
    tickers = [(k, ticker, quantization_level, interval) for ticker in np.loadtxt(tickers_file, dtype=str)]
    with Pool(processes=4) as pool:
        results = pool.map(back_test, tickers)

    output_file = os.path.join(os.getcwd(), "test-results", f"report-forex.md")
    os.remove(output_file) if os.path.exists(output_file) else None
    with open(output_file, 'w') as f: #  {"mae": mae, "rmse": rmse, "r2": r2, "sign_match": sign_match}
        print("|Ticker|Initial Capital|Final Capital|Total Return (%)|Max Drawdown (%)|Volatility (per step)|Sharpe Ratio|Number of Steps|Peak Equity|Final Drawdown|Long Trades|Short Trades|Winner Longs|Winner Shorts|Loser Longs|Loser Shorts|Delta MAE|", file=f)
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|", file=f)
    
        for result in results:    
            if result is None:
                continue
            
            stats = result['stats']
            goodness = result['goodness']
            print(f"|{stats['Ticker']}|{stats['Initial Capital']:.2f}|{stats['Final Capital']:.2f}|{stats['Total Return (%)']:.2f}|{stats['Max Drawdown (%)']:.2f}|{stats['Volatility (per step)']:.4f}|{stats['Sharpe Ratio']:.4f}|{stats['Number of Steps']}|{stats['Peak Equity']:.2f}|{stats['Final Drawdown (%)']:.2f}|{stats['Long Trades']}|{stats['Short Trades']}|{stats['Winner Longs']}|{stats['Winner Shorts']}|{stats['Loser Longs']}|{stats['Loser Shorts']}|{goodness['delta_mae']:.4f}|", file=f)