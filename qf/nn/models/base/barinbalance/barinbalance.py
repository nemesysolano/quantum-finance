import numpy as np
import pandas as pd
import tensorflow as tf
keras = tf.keras
regularizers = keras.regularizers
layers = keras.layers
models = keras.models


def create_inputs(df: pd.DataFrame, window_size: int) -> np.ndarray:
    """
    Creates input features for the Bar Imbalance Difference Forecast.
    
    The model uses a fixed lookback window of k bars of past 
    bar imbalance differences (Θd).
    """
    if 'Θd' not in df.columns:
        raise ValueError("DataFrame must contain 'Θd' column. Run add_tick_imbalance first.")
    
    series = df['Θd'].values
    inputs = []
    
    # We use a sliding window of size 'window_size' (k)
    # The inputs are Θd(t-1) to Θd(t-k)
    for i in range(window_size, len(series)):
        inputs.append(series[i-window_size:i])
        
    return np.array(inputs)

def create_targets(df: pd.DataFrame, window_size: int) -> np.ndarray:
    """
    Creates the prediction target for the Bar Imbalance Difference Forecast.
    
    The target is the bar imbalance difference Θd(τ) at time τ.
    """
    if 'Θd' not in df.columns:
        raise ValueError("DataFrame must contain 'Θd' column.")
    
    series = df['Θd'].values
    targets = []
    
    # The target corresponds to the value immediately following the input window
    for i in range(window_size, len(series)):
        targets.append(series[i])
        
    return np.array(targets)

def create_model(k, l2_rate, dropout_rate):
    """
    Drafts the baseline neural network model for the Bar Imbalance Difference Forecast.
    
    This model is designed to be subsequently improved using quantum mechanics.
    """

    l2_reg = regularizers.l2(l2_rate) if l2_rate > 0 else None

    model = models.Sequential([
        # Input shape is (window_size,) as defined in the README feature table
        layers.Input(shape=(k,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout_rate), 
        layers.Dense(32, activation='tanh'),
        layers.Dense(1)  # Single output for the forecasted Θd(t)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model