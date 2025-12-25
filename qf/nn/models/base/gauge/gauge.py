import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np

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
    
    Target: Öd(τ) at time τ.
    
    Args:
        historical_data (pd.DataFrame): Dataframe containing 'Öd'.
        k (int): Lookback window (unused, kept for compatibility).
        
    Returns:
        np.array: Target values for the current time step.
    """
    # The target is the current gauge difference Öd(t)
    return historical_data['Öd'].values

def create_inputs(historical_data, k=14):
    """
    Creates the input features for the Schrödinger Gauge Difference Forecast.
    
    Features: Last k schrödinger gauge differences [Öd(t-1), Öd(t-2), ..., Öd(t-k)].
    
    Args:
        historical_data (pd.DataFrame): Dataframe containing 'Öd'.
        k (int): Lookback window size.
        
    Returns:
        np.array: 2D array of lagged gauge differences.
    """
    # Features consist of the last k gauge differences
    # We create a sequence of lags from t-1 back to t-k
    # shift(i) moves data down by i, so row t contains value from t-i
    lags = [historical_data['Öd'].shift(i) for i in range(1, k + 1)]
    
    X = np.column_stack(lags)
    X = np.nan_to_num(X)
    
    # Reshape from (samples, k) to (samples, k, 1) to match Input(shape=(k, 1))
    return np.expand_dims(X, axis=-1)
    