import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np
import pandas as pd

def create_model(k, l2_rate, dropout_rate):
    # The input shape is now (timesteps, features)
    # k = number of time steps (lookback)
    # 3 = Position (Ö), Velocity (Öd) and Acceleration (Ödd)
    input_shape = (k, 3)
    
    l2_reg = regularizers.l2(l2_rate) if l2_rate > 0 else None
    
    model = keras.Sequential([
        layers.Flatten(input_shape=input_shape),
        # --- Hidden Layer 1 ---
        layers.Dense(64, kernel_regularizer=l2_reg),
        layers.BatchNormalization(), 
        layers.LeakyReLU(alpha=0.1),
        layers.Dropout(dropout_rate), 
        
        layers.Dense(1, activation='tanh')
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
    return historical_data['Öd'].iloc[k:].values

def create_inputs(historical_data, k=14):
    lags = []
    # Loop for Position, Velocity and Acceleration
    for col in ['Ö', 'Öd', 'Ödd']:
        for i in range(k, 0, -1):
            lags.append(historical_data[col].shift(i))
    
    df_inputs = pd.concat(lags, axis=1)
    
    # Drop rows with NaN values from shifting
    values = df_inputs.iloc[k:].values
    
    # --- ADD THIS RESHAPE STEP ---
    # Current shape: (Samples, 3 * k)
    # We want: (Samples, Features, Timesteps) -> then transpose or reshape to (Samples, Timesteps, Features)
    # The most direct way given your loop order (all Ö then all Öd then all Ödd):
    
    samples = values.shape[0]
    # We split the columns into 3 groups of k
    p = values[:, :k]       # Position lags k..1
    v = values[:, k:2*k]    # Velocity lags k..1
    a = values[:, 2*k:]     # Acceleration lags k..1
    
    # Stack them to get (Samples, k, 3)
    # np.dstack stacks along the third axis, resulting in (Samples, k, 3)
    reshaped_values = np.dstack((p, v, a))
    
    return reshaped_values