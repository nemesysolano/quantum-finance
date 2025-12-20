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

def create_model(k, l2_rate, dropout_rate): 
    """
    Creates the baseline DNN model for BINARY CLASSIFICATION (-1 or +1).

    k: lookback window size (e.g., 14).
    l2_rate: The strength of the L2 regularization.
    dropout_rate: The fraction of neurons to drop during training.
    """
    
    # Input Shape: k time steps * 1 feature (P_D)
    # The last feature set used was Pd (P_up - P_down)
    input_shape = (k * 1,)
    
    # Define L2 regularization object
    l2_reg = regularizers.l2(l2_rate) if l2_rate > 0 else None
    
    model = keras.Sequential([
        # --- Hidden Layer 1 ---
        layers.Dense(64, kernel_regularizer=l2_reg, input_shape=input_shape),
        layers.BatchNormalization(), 
        layers.LeakyReLU(alpha=0.1),
        layers.Dropout(dropout_rate), 
        
        layers.Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    return model

    
    return model

def create_inputs(historical_data: pd.DataFrame, k: int = 14) -> np.ndarray:
    """
    Extracts k-bar sequences for the P_D feature
    and concatenates them into a flattened input array of shape (N_samples, k * 1).

    Args:
        historical_data (pd.DataFrame): DataFrame containing 'P↑' and 'P↓' columns.
        k (int): The lookback window size. Default is 14.

    Returns:
        np.ndarray: A 2D array of flattened features ready for the model.
    """
    # Feature set for the k*1 input model
    feature_cols = ['P↑', 'P↓']

    # Drop rows with NaNs from feature calculation and select features.
    df = historical_data[feature_cols].dropna()

    if len(df) < k:
        return np.empty((0, k * 1))

    # Extract the NumPy arrays for sequence generation
    P_D_data = (df['P↓']-df['P↑']).values  # The new differential probability feature

    # Create k-bar sequences for each feature
    P_D_sequences = create_sequences(P_D_data, k)

    # The final input is just the P_D_sequences
    return P_D_sequences

def create_targets(historical_data: pd.DataFrame, k: int = 14) -> np.ndarray:
    """
    Creates the regression target (today's P_D) aligned to yesterday's input features.

    Args:
        historical_data (pd.DataFrame): DataFrame containing at least a 'Close' column.
        k (int): The lookback window size. Default is 14.

    Returns:
        np.ndarray: A 1D array of prediction targets.
    """
    feature_cols = ['P↑', 'P↓']

    # Drop rows with NaNs to align with create_baseline_inputs
    df = historical_data[feature_cols].dropna()

    if len(df) <= k:
        return np.empty((0,))

    # Target is today's P_D value, using inputs from t-k to t-1.
    prop_diff = df['P↑'] - df['P↓']

    # The target for a sequence ending at t-1 is the value at t.
    # We must skip the first k values to align with the sequences.
    return prop_diff.values[k:]