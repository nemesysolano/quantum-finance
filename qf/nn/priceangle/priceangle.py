import tensorflow as tf
from tensorflow import keras
layers = keras.layers
regularizers = keras.regularizers
import numpy as np
import pandas as pd

def create_model(k:int, l2_rate: float, dropout_rate: float):
    """
    Creates the DNN model mapping current Price-Time Angles to Probability Difference Pd(t).
    
    Target: Pd(t) = P↑(t) - P↓(t) (signed value in range [-1, 1]).
    Inputs: 8 trig features [cos θ1, sin θ1, ..., sin θ4].
    """
    input_shape = (8,)
    l2_reg = regularizers.l2(l2_rate) if l2_rate > 0 else None
    
    model = keras.Sequential([
        # --- Hidden Layer ---
        layers.Dense(64, kernel_regularizer=l2_reg, input_shape=input_shape),
        layers.BatchNormalization(), 
        layers.Activation('relu'),
        layers.Dropout(dropout_rate), 
        
        # --- Output Layer ---
        # Pd(t) is bounded in [-1, 1], so tanh activation is appropriate.
        layers.Dense(1, activation='tanh')
    ])
    
    # Standard regression compile
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def create_inputs(historical_data: pd.DataFrame, k:int) -> np.ndarray:
    """
    Extracts the 8 pre-calculated trig features for the Price-Time Angles.
    Aligns rows with potential target NaNs.
    """
    feature_cols = [
        'cos_θ1', 'sin_θ1', 'cos_θ2', 'sin_θ2', 
        'cos_θ3', 'sin_θ3', 'cos_θ4', 'sin_θ4'
    ]
    target_cols = ['P↑', 'P↓']
    
    # Align rows by dropping NaNs present in either features or probability targets
    df = historical_data[feature_cols + target_cols].dropna()
    
    return df[feature_cols].values

def create_targets(historical_data: pd.DataFrame, k:int) -> np.ndarray:
    """
    Creates target Pd(t) = P↑(t) - P↓(t) aligned with inputs.
    """
    feature_cols = [
        'cos_θ1', 'sin_θ1', 'cos_θ2', 'sin_θ2', 
        'cos_θ3', 'sin_θ3', 'cos_θ4', 'sin_θ4'
    ]
    target_cols = ['P↑', 'P↓']
    
    # Align rows by dropping NaNs present in either features or probability targets
    df = historical_data[feature_cols + target_cols].dropna()
    
    # Calculate Pd(t)
    pd_diff = df['P↑'] - df['P↓']
    
    return pd_diff.values