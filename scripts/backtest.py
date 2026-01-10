import sys
from multiprocessing import Pool
from qf.backtesting import simulate_trading_ds
from qf.backtesting import simulate_trading_pd
from qf.backtesting import simulate_trading_wd
import qf.market as mkt
import os
import tensorflow as tf
from qf.nn import fracdiff
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
keras = tf.keras
    
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
    historical_dataset = mkt.import_market_data(ticker, interval, k)
    historical_dataset['ATR'] = fracdiff.get_atr(historical_dataset['Close'], k)
    historical_dataset.dropna(inplace=True)
    y_test = historical_dataset['Close'].pct_change().shift(-1)
    y_test.dropna(inplace=True)    
    physics_test = historical_dataset.loc[y_test.index, ['Ö', 'Öd', 'Ödd', 'ATR','E_High', 'E_Low', 'Close', 'P↑', 'P↓', 'W', 'Wd','Ds']]

    equity_curve, cash, longs, shorts, winner_longs, winner_shorts, loser_longs, loser_shorts, transaction_log = simulator_function(y_test, physics_test)    
    stats = create_backtest_stats(ticker, equity_curve, cash, longs, shorts, winner_longs, winner_shorts, loser_longs, loser_shorts)
    stats['transaction_log'] = transaction_log
    return stats
    
simulators = {"wd": simulate_trading_wd, "pd": simulate_trading_pd, "ds": simulate_trading_ds}

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