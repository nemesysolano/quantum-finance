import qf.market as mkt
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from qf.nn.trainers import base_trainer
from qf.nn.trainers import meta_trainer
import qf.nn.models.base as base
import qf.nn as nn
import tensorflow as tf
import argparse
import sys

model_trainers = {
    'base': base_trainer,
    'meta': meta_trainer
}

base_model_factories = {
    'probdiff': base.probdiff,
    'pricevol': base.pricevoldiff,
    'wavelets': base.wavelets,
    'gauge': base.gauge,
    'barinbalance': base.barinbalance
}


if __name__ == '__main__': # 
    parser = argparse.ArgumentParser()
    if sys.argv[1] == 'import':
        mkt.import_market_all_data()
        exit()

    tf.random.set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('trainer', type=str, choices=[key for key in model_trainers.keys()], help='Trainer.')
    parser.add_argument('ticker', type=str, help='Ticker symbol in NYSE')    
    parser.add_argument("--model", type=str, choices=[key for key in base_model_factories.keys()], help='The model to use for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--patience', type=int, default=50, help='Number of epochs with no improvement after which training will be stopped.')
    parser.add_argument('--lookback', type=int, default=14, help='Lookback period required for certain indicators (like RSI).')
    parser.add_argument('--l2_rate', type=float, default=1e-4, help='Number of epochs with no improvement after which training will be stopped.')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Number of epochs with no improvement after which training will be stopped.')
    parser.add_argument('--scale_features', type=str, default='no', choices=['yes', 'no'])
    parser.add_argument('--backtest', type=str, default='sgd', choices=['sgd', 'linear', 'combined'])
    parser.add_argument('--quantization_level', type=float, default=1e2)
    parser.add_argument('--interval', type=str, default='1d', choices=['1d', '15m'])
    args = parser.parse_args()
    trainer = model_trainers[args.trainer]
    trainer(args)
