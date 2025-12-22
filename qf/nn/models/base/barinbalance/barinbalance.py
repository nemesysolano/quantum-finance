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
    bar imbalance differences (Id).
    """
    if 'Id' not in df.columns:
        raise ValueError("DataFrame must contain 'Id' column. Run add_bar_inbalance first.")
    
    series = df['Id'].values
    inputs = []
    
    # We use a sliding window of size 'window_size' (k)
    # The inputs are Id(t-k) to Id(t-1)
    for i in range(window_size, len(series)):
        inputs.append(series[i-window_size:i])
        
    return np.array(inputs)

def create_targets(df: pd.DataFrame, window_size: int) -> np.ndarray:
    """
    Creates the prediction target for the Bar Imbalance Difference Forecast.
    
    The target is the bar imbalance difference Id(t) at time t.
    """
    if 'Id' not in df.columns:
        raise ValueError("DataFrame must contain 'Id' column.")
    
    series = df['Id'].values
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

    model = keras.Sequential([
        # --- Hidden Layer 1 ---
        layers.Dense(128, kernel_regularizer=l2_reg, input_shape=(k,)),
        layers.BatchNormalization(),       
        layers.Activation('relu'),
        layers.Dropout(dropout_rate),

        # --- Hidden Layer 2 ---
        layers.Dense(64, kernel_regularizer=l2_reg),
        layers.BatchNormalization(), 
        layers.Activation('relu'),
        layers.Dropout(dropout_rate), 
        
        # --- Hidden Layer 2 ---
        layers.Dense(16, kernel_regularizer=l2_reg),
        layers.Activation('relu'),

        layers.Dense(1, activation='tanh')
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    return model
