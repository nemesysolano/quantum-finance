import tensorflow as tf
from tensorflow import keras
layers = keras.layers
regularizers = keras.regularizers
import numpy as np
import pandas as pd

def create_model(l2_rate: float, dropout_rate: float, k: int): # k is ignored, but is included int argument list to keep interface compatibility
    """
    Creates the DNN model for REGRESSION of the Probability Difference (P_d)
    based on the 8 Price-Time Angle features.

    l2_rate: The strength of the L2 regularization.
    dropout_rate: The fraction of neurons to drop during training.
    """
    # Input Shape: 4 angles * (cos + sin) = 8 features
    input_shape = (8,)
    
    # Define L2 regularization object
    l2_reg = regularizers.l2(l2_rate) if l2_rate > 0 else None
    
    model = keras.Sequential([
        # --- Input and Hidden Layer 1 ---
        # The input is the flattened 8 features
        layers.Dense(64, kernel_regularizer=l2_reg, input_shape=input_shape),
        layers.BatchNormalization(), 
        layers.Activation('relu'),
        layers.Dropout(dropout_rate), 
        
        # --- Output Layer ---
        # P_d(t) is bounded in [-1, 1], so tanh activation is used for the regression output.
        layers.Dense(1, activation='tanh')
    ])
    
    # Compile the model
    # MSE (Mean Squared Error) is the standard loss for regression.
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae'] # Mean Absolute Error
    )
    
    return model

def create_inputs(historical_data: pd.DataFrame, k: int) -> np.ndarray:  # k is ignored, but is included int argument list to keep interface compatibility
    """
    Extracts the 8 pre-calculated sine and cosine features for the Price-Time Angles.

    Args:
        historical_data (pd.DataFrame): DataFrame containing the pre-calculated 
                                        sine and cosine columns.

    Returns:
        np.ndarray: A 2D array of the 8-feature input vectors, shape (N_samples, 8).
    """
    # The 8 feature columns, already calculated in the DataFrame
    feature_cols = [
        'cos_θ1', 'sin_θ1', 'cos_θ2', 'sin_θ2', 
        'cos_θ3', 'sin_θ3', 'cos_θ4', 'sin_θ4'
    ]

    # Ensure all required columns exist and drop rows with NaNs
    df = historical_data.copy()
    
    # Select the features and drop any rows where feature calculation resulted in NaN
    df = df.dropna(subset=feature_cols)

    if df.empty:
        return np.empty((0, 8))

    # Extract the NumPy array of the feature columns
    X = df[feature_cols].values

    return X

def create_targets(historical_data: pd.DataFrame, k: int) -> np.ndarray:  # k is ignored, but is included int argument list to keep interface compatibility
    """
    Creates the regression target, which is the Probability Difference P_d(t).

    The target is P_d(t) = P↑(t) - P↓(t).

    Args:
        historical_data (pd.DataFrame): DataFrame containing the 'P↑' and 'P↓' columns.

    Returns:
        np.ndarray: A 1D array of prediction targets.
    """
    # The target components
    target_cols = ['P↑', 'P↓']
    
    # Ensure all required columns exist and drop rows with NaNs
    df = historical_data.copy()
    df = df.dropna(subset=target_cols)

    if df.empty:
        return np.empty((0,))

    # Calculate the Probability Difference target
    # P_d(t) = P↑(t) - P↓(t)
    P_D_targets = (df['P↑'] - df['P↓']).values
    
    return P_D_targets