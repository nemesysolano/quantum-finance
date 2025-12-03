import qf.market as mkt
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from qf.quantum.estimators import quantum_energy_levels, quantum_lambda
from qf.stats.distributions import empirical_distribution
import sys
from scipy.interpolate import make_smoothing_spline

if __name__ == '__main__': # 
    mkt.import_market_all_data()
    # ticker = sys.argv[1]
    # train, val, test = mkt.read_train_val_test(ticker)
    # price = 'High'
    # yesterday_price = train[price].shift(1)
    # today_price = train[price]

    # time_series = (today_price / yesterday_price).dropna()
    # distribution = empirical_distribution(time_series)
    
    # λ = quantum_lambda(distribution['X'], distribution['P'])
    # min_price = train[price].min()
    # max_price = train[price].max()
    # energy_levels = np.array(quantum_energy_levels(λ, min_price, max_price, 2))
    # print(energy_levels)
    # mpf.plot(train, type='candle', style='yahoo', title=ticker, ylabel='Price ($)', volume=False, hlines=energy_levels.tolist())