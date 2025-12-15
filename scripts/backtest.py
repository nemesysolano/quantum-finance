from sklearn.linear_model import LinearRegression
from qf import market as mkt
from qf.nn.trainers import meta_trainer_run
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import sys
from qf.nn.trainers import extract_meta_features , load_base_models
import pandas as pd
import numpy as np

from qf.nn.trainers import get_limits
base_model_names = ('pricevol', 'priceangle', 'gauge')
MAX_EQUITY_RISK_PCT = 0.01
# class SmaCross(Strategy):
#     def init(self):
#         price = self.data.Close
#         self.ma1 = self.I(SMA, price, 10)
#         self.ma2 = self.I(SMA, price, 20)

#     def next(self):
#         if crossover(self.ma1, self.ma2):
#             self.buy()
#         elif crossover(self.ma2, self.ma1):
#             self.sell()

def create_quantum_strategy(linear_model, dataset):
    class Quantum(Strategy):
        def init(self):
            super().init()
            self.linear_model = linear_model
            self.counter = 0
            self.dataset = dataset
        def next(self): # pred_pct = raw_predictions_pct[i] 

            if not self.position:
                dataset = self.dataset
                t = dataset.index[self.counter]
                X = ((dataset['corrected_pv'][t], dataset['corrected_ang'][t], dataset['corrected_g'][t]),)
                direction = np.sign(self.linear_model.predict(X))[0]
                risk_dollars, reward_dollars = get_limits(dataset['Close'][t], (dataset['E_Low'][t], dataset['E_High'][t]), direction)
                dollar_risk_limit = self.equity * MAX_EQUITY_RISK_PCT
                share_count = np.floor(dollar_risk_limit / risk_dollars)
                entry_price = dataset['Close'][t]
                max_affordable_shares = np.floor(self.equity / dataset['Close'][t])
                share_count = min(share_count, max_affordable_shares)

                if direction > 0:
                    sl_limit = entry_price - risk_dollars
                    tp_limit = entry_price + reward_dollars
                    self.buy(stop = sl_limit*1.01, sl = sl_limit, tp = tp_limit, size=share_count)
                    self.trading = True
                elif direction < 0:
                    sl_limit = entry_price + risk_dollars
                    tp_limit = entry_price - reward_dollars                        
                    self.sell(stop= sl_limit*0.99, sl = sl_limit, tp = tp_limit, size=share_count)

            self.counter += 1
            
    return Quantum

def percentual_difference(a, b):
    return (b - a) / ((a + b) / 2)

def predictions(model, X):
    return model.predict(X)

if __name__ == "__main__":
    ticker = sys.argv[1]
    _, _, _, meta_train_data, test_data = mkt.read_datasets(ticker)
    base_models = load_base_models(ticker)
    meta_X_train = extract_meta_features(meta_train_data, base_models)
    meta_prices = meta_train_data['Close'].values[-len(meta_X_train)-1:]
    meta_y_train = np.diff(meta_prices) / meta_prices[:-1]     
    
    meta_model = LinearRegression()
    meta_model.fit(meta_X_train, meta_y_train)
    
    test_X_meta = extract_meta_features(test_data, base_models)
    test_data['corrected_pv'] = pd.Series(test_X_meta[:,0], index=test_data.index[-len(test_X_meta):])
    test_data['corrected_ang'] = pd.Series(test_X_meta[:,1], index=test_data.index[-len(test_X_meta):])
    test_data['corrected_g'] = pd.Series(test_X_meta[:,2], index=test_data.index[-len(test_X_meta):])
    test_data.dropna(inplace=True)
    
    strategy = create_quantum_strategy(meta_model, test_data)
    bt = Backtest(test_data, strategy, cash=100_000, commission=.002)
    stats = bt.run()
    print(stats)
    