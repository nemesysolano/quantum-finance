import sys
from multiprocessing import Pool
import qf.market as mkt
import os
import tensorflow as tf
from qf.nn import fracdiff
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
keras = tf.keras
from typing import NamedTuple

class Transaction(NamedTuple):
    ticker: str
    trade_id: int
    entry_index: int
    exit_index: int
    duration: int
    side: int
    entry_price: float
    exit_price: float
    pl: float # profit or loss
    tp_price: float
    sl_price: float
    exit_reason: int # -1 stop loss, 0 bar close, -1 take profit

# Suggested Stress-Test Version
def dynamic_slippage(atr_pct, base_median_bps=1.0, base_sigma=0.8):
    """
    Simulates realistic friction. 1.0 bps = 0.01% of price.
    """
    noise = np.random.lognormal(mean=np.log(base_median_bps), sigma=base_sigma)
    # Scale cost by volatility (ATR)
    turbulence_factor = np.clip(atr_pct / 0.008, 1.0, 8.0) 
    return (noise * turbulence_factor) / 10000

def random_slippage(): # just a random nummber between 0.01 and 0.10
    return np.random.uniform(0.01, 0.10)
    
def apply_integer_nudge(price, dist, is_tp, is_long):
    """
    Adjusts the target distance to avoid clustering exactly on integer levels.
    """
    target_price = price + dist if (is_long and is_tp) or (not is_long and not is_tp) else price - dist
    nudge = 0.0001 # Small offset to push past the integer
    
    # If the target is very close to an integer, nudge it
    if abs(target_price - round(target_price)) < 0.001:
        if (is_long and is_tp) or (not is_long and not is_tp):
            dist += nudge # Push TP further or SL wider
        else:
            dist -= nudge # Pull SL tighter or TP closer
    return dist
