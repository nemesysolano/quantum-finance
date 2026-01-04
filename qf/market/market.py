import time
import pandas as pd
import yfinance as yf
import os
import re
from datetime import datetime, timedelta

from qf import context
from qf.market.augmentation import add_bar_inbalance, add_boundary_energy_levels, add_breaking_gap, add_directional_probabilities, add_price_time_angles, add_price_volume_differences, add_price_volume_oscillator, add_probability_differences, add_quantum_lambda, add_scrodinger_gauge, add_scrodinger_gauge_acceleration, add_scrodinger_gauge_differences, add_swing_ratio, add_wavelet_differences, add_wavelets
base_meta_border = 0.80

def read_csv(path):
    historical_data = pd.read_csv(path, parse_dates=True, date_format='%Y-%m-%d %H:%M:%S', index_col='Date')
    return historical_data

def remove_timezone_from_json_dates(file_path, interval):
    if not os.path.exists(file_path):
        return
    if interval == '1d':
        with open(file_path, 'r') as f:
            source = f.read()
            modified = re.sub(r'\d{2}:\d{2}:\d{2}.\d{2}:\d{2}\s*', '', source) # 2015-12-23 00:00:00+00:00        
        with open(file_path, 'w') as f:
            f.writelines(modified)


def import_market_data(symbol, quantization_level, interval, lookback_periods = 14):    
    module_dir = os.path.dirname(__file__)
    data_dir = os.path.join(module_dir, 'data')
    output_path = os.path.join(data_dir, f"{symbol}.csv")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    start_time = time.time()
    if not os.path.exists(output_path):
        ticker = yf.Ticker(symbol)     

        if interval != '1d':
            end_date = datetime.now()
            start_date = end_date - timedelta(days=59)
            historical_data = ticker.history(interval=interval, start=start_date, end=end_date)
        else:              
            historical_data = ticker.history("10y")  
        
        historical_data.index.name = 'Date'        
        historical_data.to_csv(output_path)        
        remove_timezone_from_json_dates(output_path, interval)
        historical_data = read_csv(output_path)

        add_breaking_gap(historical_data, 0.01)
        add_swing_ratio(historical_data)
        add_directional_probabilities(historical_data)
        add_price_volume_oscillator(historical_data)
        add_price_time_angles(historical_data)
        add_wavelets(historical_data)
        add_price_volume_differences(historical_data)
        add_probability_differences(historical_data)
        add_wavelet_differences(historical_data)
        add_quantum_lambda(ticker, historical_data, lookback_periods)
        add_boundary_energy_levels(historical_data)
        add_scrodinger_gauge(historical_data)
        add_scrodinger_gauge_differences(historical_data)
        add_scrodinger_gauge_acceleration(historical_data)
        add_bar_inbalance(historical_data)
        historical_data.to_csv(output_path)

    return read_csv(output_path)


def import_market_all_data(quantization_level, interval, lookback_periods):    
    module_dir = os.path.dirname(__file__)
    stock_listing_file = os.path.join(module_dir, 'stocks.txt')

    with open(stock_listing_file, 'r') as f:
        symbols = [line.strip() for line in f]
    
    for symbol in symbols:
        start_time = time.time()
        try:
            print(f"Importing data for {symbol}...")
            import_market_data(symbol, quantization_level, interval, lookback_periods)
            print(f"Data for {symbol} imported successfully.")
        except Exception as e:
            output_path = os.path.join(module_dir, 'data', f"{symbol}.csv")
            if os.path.exists(output_path):
                os.remove(output_path)
            print(f"Failed to import data for {symbol}: {e}")
        print(f"{symbol} Time elapsed: {time.time() - start_time} seconds")
  
def read_datasets(symbol):
    module_dir = os.path.dirname(__file__)
    data_dir = os.path.join(module_dir, 'data')
    output_path = os.path.join(data_dir, f"{symbol}.csv")
    return create_datasets(read_csv(output_path))

def create_datasets(dataset): 
    n = len(dataset)
    # Define split points for a 65/15/10/10 split
    train_end = int(n * 0.65)
    val_end = int(n * 0.80)   # 65% + 15%
    test_end = int(n * 0.90)  # 80% + 10%

    # Create contiguous, non-overlapping datasets
    base_train = dataset[0:train_end]
    base_val = dataset[train_end:val_end]
    base_test = dataset[val_end:test_end]

    # As per the logic, meta_train is an overlap with base_test
    meta_train = base_test.copy()
    meta_trade = dataset[test_end:]

    return base_train, base_val, base_test, meta_train, meta_trade
