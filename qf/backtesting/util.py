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

def dynamic_slippage(atr_pct, base_median_bps=0.01, base_sigma=0.5):
    """
    Generates a log-normal slippage distribution scaled by ATR%.
    """
    noise = np.random.lognormal(mean=np.log(base_median_bps), sigma=base_sigma)
    turbulence_factor = np.clip(atr_pct / 0.008, 1.0, 8.0) 
    return (noise * turbulence_factor) / 10000

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
