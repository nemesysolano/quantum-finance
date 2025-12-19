import tensorflow as tf
from tensorflow import keras
layers = keras.layers
regularizers = keras.regularizers
import numpy as np
import pandas as pd

def create_model(k, l2_rate,  dropout_rate):
    """
    Creates the baseline DNN model for REGRESSION of the Price-Volume Difference (Y_d).

    k: lookback window size (e.g., 14).
    l2_rate: The strength of the L2 regularization.
    dropout_rate: The fraction of neurons to drop during training.
    """
    
    # Input Shape: k time steps * 1 feature (Y_d)
    input_shape = (k * 1,)
    
    # Define L2 regularization object
    l2_reg = regularizers.l2(l2_rate) if l2_rate > 0 else None
    
    model = keras.Sequential([
        # --- Hidden Layer 1 ---
        layers.Dense(64, kernel_regularizer=l2_reg, input_shape=input_shape),
        layers.BatchNormalization(), 
        layers.Activation('tanh'),
        layers.Dropout(dropout_rate), 
        
        # --- Output Layer ---
        # The target Y_d(t) is a signed value with no known strict bounds, 
        # so a linear activation (default) is used for regression.
        layers.Dense(1, activation='tanh')
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    return model

def calculate_gauge_series(historical_data: pd.DataFrame) -> pd.Series:
    """
    Helper to calculate the Schrödinger Gauge Ö(t) based on README definitions.
    """
    # Assuming E_Low (E^n1) and E_High (E^n2) are pre-calculated in the dataframe
    # as per the 'Schrödinger Gauge' section of the README.
    E_Low = historical_data['E_Low']
    E_High = historical_data['E_High']
    close = historical_data['Close']
    
    # Formula: Ö(t) = 2 * (Ö↑ - Ö↓) / (Ö↑ + Ö↓)
    # where Ö↑ = E_High - close and Ö↓ = close - E_Low
    numerator = (E_High - close) - (close - E_Low)
    denominator = (E_High - close) + (close - E_Low)
    
    return 2 * numerator / (denominator + 1e-9)

def create_inputs(historical_data: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Forecasts Schrödinger Gauge Difference Ö_d(t) from last k differences.
    Input Features: [Ö_d(t-1), Ö_d(t-2), ..., Ö_d(t-k)]
    """
    # 1. Calculate the absolute gauge
    gauge = calculate_gauge_series(historical_data)
    
    # 2. Calculate the Gauge Difference: Ö_d(t) = Ö(t) - Ö(t-1)
    gauge_diff = gauge.diff().rename('Ö')
    
    # 3. Create the lookback features for the difference
    features = []
    for i in range(1, k + 1):
        features.append(gauge_diff.shift(i).rename(f'Ö_d{i}'))
    
    # 4. Concatenate and drop NaNs (k lags + 1 for the initial diff)
    input_df = pd.concat(features, axis=1).dropna()
    return input_df

def create_targets(historical_data: pd.DataFrame, k: int) -> pd.Series:
    """
    Prediction Target: Schrödinger Gauge Difference Ö_d(t) at time t.
    Aligned with the inputs from create_inputs.
    """
    # 1. Calculate the absolute gauge and its difference
    gauge = calculate_gauge_series(historical_data)
    gauge_diff = gauge.diff().rename('Od')
    
    # 2. Align with create_inputs by dropping the same initial rows
    # We drop k + 1 rows: k for the lookback and 1 for the diff() operation
    return gauge_diff.iloc[k+1:]