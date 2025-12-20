import tensorflow as tf
from tensorflow import keras
layers = keras.layers
regularizers = keras.regularizers
import numpy as np
import pandas as pd

def create_sequences(data, k):
    """Utility function to create k-bar lookback sequences."""
    n_samples = len(data) - k
    if n_samples <= 0:
        return np.empty((0, k))
    
    sequences = np.zeros((n_samples, k))
    for i in range(n_samples):
        # Sequence is from t-k to t-1
        sequences[i] = data[i:i+k]
    return sequences

def create_model(k: int, l2_rate: float, dropout_rate: float):
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
    
    # Compile the model
    # MSE (Mean Squared Error) is the standard loss for regression.
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae'] # Mean Absolute Error for interpretability
    )
    
    return model

def create_targets(historical_data: pd.DataFrame, k: int):
    """
    Creates the prediction target for the Price-Volume Difference Forecast.
    
    The target is the Price-Volume Difference Yd at time t.
    We align the target by dropping the first k rows to match the input sequences.
    
    Args:
        historical_data (pd.DataFrame): Dataframe containing 'Yd'.
        k (int): Lookback window size.
        
    Returns:
        np.array: Target values Yd(t).
    """
    # The target is the current Yd
    # We drop the first k rows because they won't have complete input sequences
    y = historical_data['Yd'].values[k:]
    return y.reshape(-1, 1)

def create_inputs(historical_data: pd.DataFrame, k: int):
    """
    Creates the input features for the Price-Volume Difference Forecast.
    
    Features: A sequence of the last k Price-Volume Differences [Yd(t-1), ..., Yd(t-k)].
    
    Args:
        historical_data (pd.DataFrame): Dataframe containing 'Yd'.
        k (int): Lookback window size.
        
    Returns:
        np.array: 2D array of lagged sequences with shape (samples, k).
    """
    data = historical_data['Yd'].values
    n_samples = len(data) - k
    
    if n_samples <= 0:
        return np.empty((0, k))
    
    # Initialize the input matrix
    X = np.zeros((n_samples, k))
    
    for i in range(n_samples):
        # Feature vector at index i (representing time t) 
        # consists of data from i to i + k - 1 (representing t-k to t-1)
        X[i] = data[i : i + k]
        
    return X