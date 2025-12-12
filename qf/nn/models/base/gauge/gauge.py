import tensorflow as tf
from tensorflow import keras
layers = keras.layers
regularizers = keras.regularizers
import numpy as np
import pandas as pd

def create_model(k, l2_rate, dropout_rate):
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
        layers.Activation('relu'),
        layers.Dropout(dropout_rate), 
        
        # --- Output Layer ---
        # The target Y_d(t) is a signed value with no known strict bounds, 
        # so a linear activation (default) is used for regression.
        layers.Dense(1, activation='tanh')
    ])
    
    model.compile(
        optimizer='adam',
        loss='mae',
        metrics=['mae']
    )
    return model

def create_inputs(historical_data: pd.DataFrame, k):
    E_Low = historical_data['E_Low']
    E_High = historical_data['E_High']
    close = historical_data['Close']
    # Calculate the distance ratio as the symmetric percentage difference
    # between the distance to the high energy level and the distance to the low energy level.
    numerator = E_High - close
    denominator = close - E_Low
    distance_ratio = 2 * (numerator - denominator) / (numerator + denominator + 1e-9)
    historical_data['Ö'] = distance_ratio
    
    # Create the lookback features [Ö(t-1), Ö(t-2), ..., Ö(t-k)] for each time t.
    features = []
    for i in range(1, k + 1):
        features.append(historical_data['Ö'].shift(i))
    
    # Concatenate the shifted series into a DataFrame and drop rows with NaNs.
    input_df = pd.concat(features, axis=1).dropna()
    return input_df

def create_targets(historical_data, k):
    """
    Create the target values, which are the Schrödinger Gauge values Ö(t).
    This function ensures the targets Y(t) are aligned with the input features 
    X(t) = [Ö(t-1), Ö(t-2), ..., Ö(t-k)].
    """
    # Calculate the Schrödinger Gauge Ö(t) as the target.
    E_Low = historical_data['E_Low']
    E_High = historical_data['E_High']
    close = historical_data['Close']
    numerator = E_High - close
    denominator = close - E_Low
    distance_ratio = 2 * (numerator - denominator) / (numerator + denominator + 1e-9)
    historical_data['Ö'] = distance_ratio

    # The create_inputs function drops the first `k` rows because of the lookback window.
    # To align the targets, we must also discard the first `k` values of the Ö series.
    # This ensures that the first input sample (features from t=0 to t=k-1) corresponds 
    # to the first target (at t=k).
    aligned_targets = historical_data['Ö'][k:]
    return aligned_targets.values