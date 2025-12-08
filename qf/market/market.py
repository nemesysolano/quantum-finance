import pandas as pd
import yfinance as yf
import os
import re
import numpy as np
from typing import Callable, Union

from qf.market.augmentation import add_breaking_gap, add_cosine_and_sine_for_price_time_angles, add_directional_probabilities, add_fast_swing_ratio, add_fast_trend_run, add_last_opposite, add_price_volume_strength_oscillator, add_relative_volume, add_slow_swing_ratio, add_slow_trend_run, add_structural_direction

def read_csv(path):
    historical_data = pd.read_csv(path, parse_dates=True, date_format='%Y-%m-%d', index_col='Date')
    return historical_data

def remove_timezone_from_json_dates(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, 'r') as f:
        source = f.read()
        modified = re.sub(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:(?:\+|-)\d{2}:\d{2})', r'\1', source)

    with open(file_path, 'w') as f:
        f.writelines(modified)

def add_low_high_average(historical_data):
    historical_data['Low_High_Avg'] = (historical_data['Low'] + historical_data['High']) / 2

def import_market_data(symbol):    
    module_dir = os.path.dirname(__file__)
    data_dir = os.path.join(module_dir, 'data')
    output_path = os.path.join(data_dir, f"{symbol}.csv")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(output_path):
        ticker = yf.Ticker(symbol)        
        historical_data = ticker.history(period="10y", interval="1d")  
        add_low_high_average(historical_data)
        historical_data.to_csv(output_path)        

        remove_timezone_from_json_dates(output_path)
        historical_data = read_csv(output_path)
        add_relative_volume(ticker, historical_data)
        add_structural_direction(historical_data)
        add_slow_trend_run(historical_data)
        add_breaking_gap(historical_data)
        add_fast_trend_run(historical_data)
        add_fast_swing_ratio(historical_data)
        add_last_opposite(historical_data)
        add_slow_swing_ratio(historical_data)
        add_directional_probabilities(historical_data)
        add_price_volume_strength_oscillator(historical_data, "High")
        add_price_volume_strength_oscillator(historical_data, "Close")
        add_price_volume_strength_oscillator(historical_data, "Low")       
        add_cosine_and_sine_for_price_time_angles(historical_data) 
        historical_data.to_csv(output_path)
        return historical_data


    return read_csv(output_path)

def import_market_all_data():    
    module_dir = os.path.dirname(__file__)
    stock_listing_file = os.path.join(module_dir, 'stocks.txt')

    with open(stock_listing_file, 'r') as f:
        symbols = [line.strip() for line in f]
    
    for symbol in symbols:
        import_market_data(symbol)

def read_base_train_val_test(symbol):
    module_dir = os.path.dirname(__file__)
    data_dir = os.path.join(module_dir, 'data')
    output_path = os.path.join(data_dir, f"{symbol}.csv")
    return create_base_train_val_test(read_csv(output_path))
    

def create_base_train_val_test(dataset):
    base_length = int(len(dataset)*0.8)
    historical_data = dataset[0: base_length]
    n = len(historical_data)
    train = historical_data[0:int(n*0.7)]
    val = historical_data[int(n*0.7):int(n*0.9)]
    test = historical_data[int(n*0.9):]
    return train, val, test