import sys
from multiprocessing import Pool
import qf.market as mkt
import os
import tensorflow as tf
from qf.nn import fracdiff
import qf.nn.models.base as base
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
keras = tf.keras

def dynamic_slippage(atr_pct, base_median_bps=0.01, base_sigma=0.5):
    """
    Generates a log-normal slippage distribution scaled by ATR%.
    Mimics market microstructure: tight spreads usually, but sudden 'fat-tail' spikes.
    """
    # Generate log-normal noise (real-world execution distribution)
    noise = np.random.lognormal(mean=np.log(base_median_bps), sigma=base_sigma)
    
    # Apply volatility penalty: scaling the impact based on ATR/Price ratio
    turbulence_factor = np.clip(atr_pct / 0.008, 1.0, 8.0) 
    
    return (noise * turbulence_factor) / 10000

def discretize_take_profit(tp, current_price):
    pass

def simulate_trading_wd(y_test, physics_test, initial_cap=10000):    
    cash = initial_cap
    equity_curve = [initial_cap]
    longs, shorts = 0, 0
    winner_longs, winner_shorts = 0, 0
    loser_longs, loser_shorts = 0, 0
    
    y_actual = y_test.values if isinstance(y_test, pd.Series) else y_test
    o_d = physics_test['Öd'].values
    o_dd = physics_test['Ödd'].values
    atr_values = physics_test['ATR'].values
    e_low = physics_test['E_Low'].values
    e_high = physics_test['E_High'].values
    price_values = physics_test['Close'].values
    W = physics_test['W'].values # Transaction Momentum Reservoir
    
    for i in range(len(y_actual) - 1):
        if cash <= 0:
            equity_curve.append(0)
            continue

        # 1. DYNAMIC BASE RISK (Performance-Anchored)
        # We calculate risk rate relative to initial capital to prevent 'cheating'
        rel_performance = (cash - initial_cap) / initial_cap
        base_risk_rate = np.clip(0.02 + (rel_performance * 0.1), 0.01, 0.05)

        # 2. THE "W" MOMENTUM RESERVOIR (Smart Compounding)
        # Instead of compounding on 'cash', we compound on 'initial_cap' 
        # but scale it by the current transaction momentum (W).
        # We take the absolute value of W to determine the 'confidence' multiplier.
        w_multiplier = np.clip(abs(W[i]), 1.0, 3.0) 
        risk_amount = initial_cap * base_risk_rate * w_multiplier

        # Signal/Market Variables
        threshold = int(np.sign(o_d[i]) + np.sign(o_dd[i]))
        next_bar_return = y_actual[i + 1]
        yesterday_atr = atr_values[i]
        current_price = price_values[i]
        
        # Friction (Slippage) using the dynamic model for higher realism
        atr_pct = yesterday_atr / current_price
        slippage_cost = dynamic_slippage(atr_pct) 

        # LONG SIGNAL
        if threshold == 2 and W[i] > 0:
            longs += 1
            tp = min(yesterday_atr, e_high[i] - current_price)
            sl = -min(0.33 * yesterday_atr, current_price - e_low[i])
            
            # Friction scales with current account size
            friction = cash * slippage_cost * 2
            
            if next_bar_return >= tp:
                cash += (risk_amount * 3) - friction
                winner_longs += 1
            elif next_bar_return <= sl:
                cash -= (risk_amount + friction)
                loser_longs += 1
            else:
                realized = next_bar_return / (0.33 * yesterday_atr)
                cash += (risk_amount * realized) - friction
                if next_bar_return > 0: winner_longs += 1
                else: loser_longs += 1

        # SHORT SIGNAL
        elif threshold == -2 and W[i] < 0:
            shorts += 1
            tp = -min(yesterday_atr, current_price - e_low[i])
            sl = 0.33 * min(yesterday_atr, e_high[i] - current_price)
            
            friction = cash * slippage_cost * 2
            
            if next_bar_return <= tp:
                cash += (risk_amount * 3) - friction
                winner_shorts += 1
            elif next_bar_return >= sl:
                cash -= (risk_amount + friction)
                loser_shorts += 1
            else:
                realized = -next_bar_return / (0.33 * yesterday_atr)
                cash += (risk_amount * realized) - friction
                if next_bar_return < 0: winner_shorts += 1
                else: loser_shorts += 1

        equity_curve.append(cash)

    return equity_curve, cash, longs, shorts, winner_longs, winner_shorts, loser_longs, loser_shorts

def simulate_trading_pd(y_test, physics_test, initial_cap=10000, k_window=14):    
    cash = initial_cap
    equity_curve = [initial_cap]
    
    # Transaction Tracker for Momentum-based Risk Scaling
    trade_returns = [] 
    
    longs, shorts = 0, 0
    winner_longs, winner_shorts = 0, 0
    loser_longs, loser_shorts = 0, 0
    
    y_actual = y_test.values if isinstance(y_test, pd.Series) else y_test
    o_d = physics_test['Öd'].values
    o_dd = physics_test['Ödd'].values
    atr_values = physics_test['ATR'].values
    e_low = physics_test['E_Low'].values
    e_high = physics_test['E_High'].values
    price_values = physics_test['Close'].values
    p_up = physics_test['P↑'].values
    p_down = physics_test['P↓'].values
    
    for i in range(len(y_actual) - 1):
        if cash <= 0:
            equity_curve.append(0)
            continue

        # 1. DYNAMIC RISK RATE (Transaction Momentum + Account Performance)
        # Calculate recent strategy momentum from last k=14 trades
        k_factor = 0
        if len(trade_returns) >= k_window:
            # Average normalized return of the last k transactions
            avg_recent_return = np.mean(trade_returns[-k_window:])
            # Sensitivity: every +1% avg return adds ~0.5% to the risk rate
            k_factor = np.clip(avg_recent_return * 0.5, -0.01, 0.02)

        rel_performance = (cash - initial_cap) / initial_cap
        base_risk = 0.02 + (rel_performance * 0.05)
        
        # Combine both factors: protects capital during cold streaks
        risk_rate = np.clip(base_risk + k_factor, 0.01, 0.05)
        risk_amount = initial_cap * risk_rate

        # 2. LOG-NORMAL SLIPPAGE
        current_price = price_values[i]
        yesterday_atr = atr_values[i]
        atr_pct = yesterday_atr / current_price
        slippage_cost = dynamic_slippage(atr_pct)

        # 3. SIGNAL LOGIC (Schrödinger Gauge Direction & Acceleration)
        threshold = int(np.sign(o_d[i]) + np.sign(o_dd[i]))
        next_bar_return = y_actual[i + 1] 

        # --- LONG EXECUTION ---
        if threshold == 2 and p_up[i] > p_down[i]:
            longs += 1
            tp = min(yesterday_atr, e_high[i] - current_price)
            sl = -min(0.33 * yesterday_atr, current_price - e_low[i])
            
            friction = cash * slippage_cost * 2
            net_profit = 0
            
            if next_bar_return >= tp:
                net_profit = (risk_amount * 3) - friction
                winner_longs += 1
            elif next_bar_return <= sl:
                net_profit = -(risk_amount + friction)
                loser_longs += 1
            else:
                realized_ratio = next_bar_return / (0.33 * yesterday_atr)
                net_profit = (risk_amount * realized_ratio) - friction
                if next_bar_return > 0: winner_longs += 1
                else: loser_longs += 1
            
            cash += net_profit
            trade_returns.append(net_profit / risk_amount) # Record normalized result

        # --- SHORT EXECUTION ---
        elif threshold == -2 and p_down[i] > p_up[i]:
            shorts += 1
            tp = -min(yesterday_atr, current_price - e_low[i])
            sl = 0.33 * min(yesterday_atr, e_high[i] - current_price)
            
            friction = cash * slippage_cost * 2
            net_profit = 0

            if next_bar_return <= tp:
                net_profit = (risk_amount * 3) - friction
                winner_shorts += 1
            elif next_bar_return >= sl:
                net_profit = -(risk_amount + friction)
                loser_shorts += 1
            else:
                realized_ratio = -next_bar_return / (0.33 * yesterday_atr)
                net_profit = (risk_amount * realized_ratio) - friction
                if next_bar_return < 0: winner_shorts += 1
                else: loser_shorts += 1
            
            cash += net_profit
            trade_returns.append(net_profit / risk_amount)

        equity_curve.append(cash)

    return (equity_curve, cash, longs, shorts, winner_longs, winner_shorts, loser_longs, loser_shorts)

    
def create_backtest_stats(ticker,equity_curve, cash, long_trades, short_trades, winner_longs, winner_shorts, loser_longs, loser_shorts):
    # Ensure 1D array even if input is a list of arrays or 2D matrix
    equity_array = np.ravel(equity_curve)
    # If final_capital was an array due to the previous bug, take the first value
    f_cap = float(np.ravel(cash)[0]) if np.ndim(cash) > 0 else cash
    
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
        "Final Capital": cash,
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
        "Loser Shorts": loser_shorts
    }
    return stats

def back_test(params):
    (k, ticker, interval, simulator_function) = params        
    # try:
    historical_dataset = mkt.import_market_data(ticker, interval, k)
    historical_dataset['ATR'] = fracdiff.get_atr(historical_dataset['Close'], k)
    historical_dataset.dropna(inplace=True)
    y_test = historical_dataset['Close'].pct_change().shift(-1)
    y_test.dropna(inplace=True)    
    physics_test = historical_dataset.loc[y_test.index, ['Ö', 'Öd', 'Ödd', 'ATR','E_High', 'E_Low', 'Close', 'P↑', 'P↓', 'W', 'Wd']]

    equity_curve, cash, longs, shorts, winner_longs, winner_shorts, loser_longs, loser_shorts = simulator_function(y_test, physics_test)
    stats = create_backtest_stats(ticker, equity_curve, cash, longs, shorts, winner_longs, winner_shorts, loser_longs, loser_shorts)
    return stats
    # except:
    #     print(f"ERROR: backtersing {ticker}")
    #     return  None
    
simulators = {"wd": simulate_trading_wd, "pd": simulate_trading_pd}

if __name__ == '__main__':
    tickers_file = sys.argv[1]    
    simulator = sys.argv[2]
    simulator_function = simulators[simulator]
    k = 14
    tickers = [(k, ticker, "1d", simulator_function) for ticker in np.loadtxt(tickers_file, dtype=str)]
    with Pool(processes=4) as pool:
        result = list(map(back_test, tickers))
    
    output_file = os.path.join(os.getcwd(), "test-results", f"report-stocksgauge-{simulator}.csv")
    os.remove(output_file) if os.path.exists(output_file) else None
    with open(output_file, 'w') as f: #
        print(
           "Ticker,Initial Capital,Final Capital,Total Return (%),Max Drawdown (%),Volatility (per step),Sharpe Ratio,Number of Steps,Peak Equity,Final Drawdown,Long Trades,Short Trades,Winner Longs,Winner Shorts,Loser Longs,Loser Shorts,", 
        file=f)    
        for stats in result:    
            if stats is None:
                continue            
            print(f"{stats['Ticker']},{stats['Initial Capital']:.2f},{stats['Final Capital']:.2f},{stats['Total Return (%)']:.2f},{stats['Max Drawdown (%)']:.2f},{stats['Volatility (per step)']:.4f},{stats['Sharpe Ratio']:.4f},{stats['Number of Steps']},{stats['Peak Equity']:.2f},{stats['Final Drawdown (%)']:.2f},{stats['Long Trades']},{stats['Short Trades']},{stats['Winner Longs']},{stats['Winner Shorts']},{stats['Loser Longs']},{stats['Loser Shorts']}", file=f)