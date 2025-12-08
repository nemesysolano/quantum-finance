
import pandas as pd
import numpy as np
import qf.market as mkt
import datetime 

if __name__ == "__main__":

    # Fetch data for a particular stock
    stock_data = mkt.import_market_data('AAPL')
    start = datetime.datetime(2020, 1, 1)
    end = datetime.datetime(2022, 1, 1)
    stock_data = stock_data[start:end]
    print(stock_data.head())