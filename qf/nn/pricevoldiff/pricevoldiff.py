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
        layers.Dense(1, activation='linear')
    ])
    
    # Compile the model
    # MSE (Mean Squared Error) is the standard loss for regression.
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae'] # Mean Absolute Error for interpretability
    )
    
    return model

def create_inputs(historical_data: pd.DataFrame, k: int = 14) -> np.ndarray:
    """
    Extracts k-bar sequences for the Price-Volume Difference (Y_d) feature 
    and concatenates them into a flattened input array of shape (N_samples, k * 1).

    The input feature at time τ is Y_d(τ) = Y(τ) - Y(τ-1), where Y is the 
    Price-Volume Oscillator.

    Args:
        historical_data (pd.DataFrame): DataFrame containing the 'Y' (Price-Volume Oscillator) column.
        k (int): The lookback window size. Default is 14.

    Returns:
        np.ndarray: A 2D array of flattened features ready for the model.
    """
    # The oscillator value Y(t) must be pre-calculated and present in the DataFrame.
    # The input feature is the difference of this oscillator: Y_d(τ) = Y(τ) - Y(τ-1).
    
    df = historical_data.copy()

    # 1. Calculate the Price-Volume Difference Y_d
    # Y_d(t) = Y(t) - Y(t-1)
    # We use .diff() which automatically calculates Y(t) - Y(t-1)
    df['Y_D'] = df['Y_Close'].diff() 

    # Drop rows with NaNs. The first row of 'Y_D' will be NaN due to .diff().
    df = df.dropna(subset=['Y_D'])

    if len(df) < k:
        return np.empty((0, k * 1))

    # Extract the NumPy array for sequence generation
    # Y_D_data is the sequence of Y_d(τ) values.
    Y_D_data = df['Y_D'].values

    # Create k-bar sequences for the Y_D feature.
    # A sequence ending at index i has Y_D[i-k] to Y_D[i-1].
    Y_D_sequences = create_sequences(Y_D_data, k)

    # The final input is just the Y_D_sequences, ready for the Dense layer.
    return Y_D_sequences

def create_targets(historical_data: pd.DataFrame, k: int = 14) -> np.ndarray:
    """
    Creates the regression target (today's Y_d) aligned to yesterday's input features.

    The target is Y_d(t), where the input sequence ends at t-1.

    Args:
        historical_data (pd.DataFrame): DataFrame containing the 'Y' (Price-Volume Oscillator) column.
        k (int): The lookback window size. Default is 14.

    Returns:
        np.ndarray: A 1D array of prediction targets.
    """
    df = historical_data.copy()
    
    # 1. Calculate the Price-Volume Difference Y_d
    # Y_d(t) = Y(t) - Y(t-1)
    df['Y_D'] = df['Y_Close'].diff() 

    # Drop rows with NaNs. The first row of 'Y_D' will be NaN.
    df = df.dropna(subset=['Y_D'])

    if len(df) <= k:
        return np.empty((0,))

    # The target for an input sequence ending at t-1 is the value at t.
    # The Y_D array starts at the index where the first valid difference is calculated.
    # The sequence generation in `create_inputs` uses the first N-k elements.
    # The target array must use the elements from index k to the end.
    return df['Y_D'].values[k:]