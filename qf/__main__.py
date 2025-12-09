import qf.market as mkt
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from qf.quantum.estimators import quantum_energy_levels, quantum_lambda
from qf.stats.distributions import empirical_distribution
import qf.nn.models.base as base
import qf.nn.models.meta as meta
import tensorflow as tf
import argparse


model_trainers = {
    'base': meta.base_trainer,
    'meta': meta.meta_trainer
}

base_model_factories = {
    'prob': base.probdiff,
    'pricevol': base.pricevoldiff,
    'priceangle': base.priceangle
}

if __name__ == '__main__': # 
    parser = argparse.ArgumentParser()
    parser.add_argument('trainer', type=str, choices=[key for key in model_trainers.keys()], help='Trainer.')
    parser.add_argument('ticker', type=str, help='Ticker symbol in NYSE')    
    parser.add_argument("--model", type=str, choices=[key for key in base_model_factories.keys()], help='The model to use for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--patience', type=int, default=50, help='Number of epochs with no improvement after which training will be stopped.')
    parser.add_argument('--lookback', type=int, default=14, help='Lookback period required for certain indicators (like RSI).')
    parser.add_argument('--l2_rate', type=float, default=1e-6, help='Number of epochs with no improvement after which training will be stopped.')
    parser.add_argument('--dropout_rate', type=float, default=0.20, help='Number of epochs with no improvement after which training will be stopped.')
    parser.add_argument('--scale_features', type=str, default='yes', choices=['yes', 'no'])

    args = parser.parse_args()
    trainer = model_trainers[args.trainer]
    trainer(args)

