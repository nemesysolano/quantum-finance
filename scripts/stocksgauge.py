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
from qf.nn.models.base.gauge.metafeatures import train_gauge_meta_ensemble
keras = tf.keras

def random_slipage(lower, upper):
    return np.random.uniform(lower, upper)

def simulate_trading_no_hedge(y_test, physics_test, initial_cap=10000):    
    cash = initial_cap
    equity_curve = [initial_cap]
    longs, shorts = 0, 0
    winner_longs, winner_shorts = 0, 0
    loser_longs, loser_shorts = 0, 0
    
    y_actual = y_test.values if isinstance(y_test, pd.Series) else y_test
    # Extract physics for dynamic thresholding
    o_d = physics_test['Öd'].values
    o_dd = physics_test['Ödd'].values
    atr_values = physics_test['ATR'].values
    e_low = physics_test['E_Low'].values
    e_high = physics_test['E_High'].values
    price_values = physics_test['Close'].values

    # Convert basis points to a decimal multiplier (e.g., 2bps = 0.0002)
    # This represents the cumulative friction for a round-trip trade
    
    # Start from 0, but evaluate outcome at i + 1
    # We stop at len(probs) - 1 to avoid index out of bounds
    for i in range(len(y_actual) - 1):
        slippage_cost = random_slipage(5, 10) / 10000

        # Dynamic Threshold calculation
        threshold = np.int(np.sign(o_d[i]) + np.sign(o_dd[i]))
        
        # Reference price/ATR at time of signal (t)
        # Outcome is the move from t to t+1
        next_bar_return = y_actual[i + 1] 
        yesterday_atr = atr_values[i]
        current_price = price_values[i]
        upper_distance = e_high[i] - current_price
        lower_distance = current_price - e_low[i]

        # 1. LONG SIGNAL
        if threshold == 2:
            longs += 1
            tp = min(yesterday_atr,upper_distance)
            sl = -min(0.33 * yesterday_atr, lower_distance)
            
            # Slippage is applied as a deduction from the profit/loss realized
            if next_bar_return >= tp:
                cash += ((initial_cap * 0.02) * 3) - (cash * slippage_cost)
                winner_longs += 1
            elif next_bar_return <= sl:
                cash -= ((initial_cap * 0.02) + (cash * slippage_cost))
                loser_longs += 1
            else:
                # Market close exit (pro-rata)
                realized = next_bar_return / (0.33 * yesterday_atr)
                cash += ((initial_cap * 0.02) * realized) - (cash * slippage_cost)
                if next_bar_return > 0: winner_longs += 1
                else: loser_longs += 1

        # 2. SHORT SIGNAL
        elif threshold == -2:
            shorts += 1
            tp = -min(yesterday_atr, lower_distance)
            sl = 0.33 * min(yesterday_atr, upper_distance)
            
            if next_bar_return <= tp:
                cash += ((initial_cap * 0.02) * 3) - (cash * slippage_cost)
                winner_shorts += 1
            elif next_bar_return >= sl:
                cash -= ((initial_cap * 0.02) + (cash * slippage_cost))
                loser_shorts += 1
            else:
                realized = -next_bar_return / (0.33 * yesterday_atr)
                cash += ((initial_cap * 0.02) * realized) - (cash * slippage_cost)
                if next_bar_return < 0: winner_shorts += 1
                else: loser_shorts += 1

        equity_curve.append(cash)

    return equity_curve, cash, longs, shorts, winner_longs, winner_shorts, loser_longs, loser_shorts


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
    (k, ticker, interval) = params        
    gru_model_file = os.path.join(os.getcwd(), 'models', f'{ticker}-gauge.keras')
    gru_model = keras.models.load_model(gru_model_file)
    print(f"Loaded {gru_model_file}")
    historical_dataset = base.metafeatures.gauge_meta_features(gru_model, k,  mkt.import_market_data(ticker, interval, k))
    historical_dataset['ATR'] = fracdiff.get_atr(historical_dataset['Close'], k)

    # Prepare for Meta-Training
    _, _, _, meta_train, meta_test = mkt.create_datasets(historical_dataset)
    _, X_test, y_test, probs = train_gauge_meta_ensemble(meta_train, meta_test)
    physics_test = historical_dataset.loc[X_test.index, ['Ö', 'Öd', 'Ödd', 'ATR','E_High', 'E_Low', 'Close']]
    equity_curve, cash, longs, shorts, winner_longs, loser_longs, winner_shorts, loser_shorts = simulate_trading_no_hedge(y_test, physics_test)
    stats = create_backtest_stats(ticker, equity_curve, cash, longs, shorts, winner_longs, winner_shorts, loser_longs, loser_shorts)
    return stats

if __name__ == '__main__':
    tickers_file = sys.argv[1]    
    k = 14
    tickers = [(k, ticker, "1d") for ticker in np.loadtxt(tickers_file, dtype=str)]
    with Pool(processes=4) as pool:
        result = pool.map(back_test, tickers)
    
    output_file = os.path.join(os.getcwd(), "test-results", f"report-stocksgauge.csv")
    os.remove(output_file) if os.path.exists(output_file) else None
    with open(output_file, 'w') as f: #
        print(
           "Ticker,Initial Capital,Final Capital,Total Return (%),Max Drawdown (%),Volatility (per step),Sharpe Ratio,Number of Steps,Peak Equity,Final Drawdown,Long Trades,Short Trades,Winner Longs,Winner Shorts,Loser Longs,Loser Shorts,", 
        file=f)    
        for stats in result:    
            if result is None:
                continue            
            print(f"{stats['Ticker']},{stats['Initial Capital']:.2f},{stats['Final Capital']:.2f},{stats['Total Return (%)']:.2f},{stats['Max Drawdown (%)']:.2f},{stats['Volatility (per step)']:.4f},{stats['Sharpe Ratio']:.4f},{stats['Number of Steps']},{stats['Peak Equity']:.2f},{stats['Final Drawdown (%)']:.2f},{stats['Long Trades']},{stats['Short Trades']},{stats['Winner Longs']},{stats['Winner Shorts']},{stats['Loser Longs']},{stats['Loser Shorts']}", file=f)