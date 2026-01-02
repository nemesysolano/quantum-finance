import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np
import pandas as pd

def create_model(k, l2_rate,  dropout_rate):
    
    # Input Shape: k time steps * 1 feature (Y_d)
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

def create_targets(historical_data, k=14):
    """
    Creates the prediction target for the Schrödinger Gauge Difference Forecast.
    Target: Öd(t) 
    
    Args:
        historical_data (pd.DataFrame): Dataframe containing 'Öd'.
        k (int): Lookback window (kept for interface compatibility).
        
    Returns:
        np.array: Target values for the current time step.
    """
    # We extract the 'Öd' column. 
    # In the training pipeline, this will be aligned with inputs from t-1.
    # Drop the first k rows to align with inputs (which lose k rows due to shifting)
    return historical_data['Öd'].iloc[k:].values

def create_inputs(historical_data, k=14):
    """
    Creates the input features using a lagged window to prevent leakage.
    Features: [Öd(t-1), Öd(t-2), ..., Öd(t-k)]
    
    Args:
        historical_data (pd.DataFrame): Dataframe containing 'Öd'.
        k (int): Lookback window size.
        
    Returns:
        np.array: 2D array of lagged gauge differences.
    """
    # Create a list of Series, each shifted by an incrementing lag.
    # shift(1) ensures the model never sees the target Öd(t) during training.
    lags = []
    for i in range(1, k + 1):        
        shifted = historical_data['Öd'].shift(i)
        lags.append(shifted)
    
    # Concatenate lags column-wise
    df_inputs = pd.concat(lags, axis=1)
    
    # Drop rows with NaNs (introduced by shifting)
    df_inputs = df_inputs.dropna()
    
    # Return as a numpy array for the Neural Network
    return df_inputs.values