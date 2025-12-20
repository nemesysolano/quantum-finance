import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
regularizers = keras.regularizers
import numpy as np
import pandas as pd

def create_model(k: int, l2_rate= 1e-4, dropout_rate = 0.01):
    """
    Creates the DNN model for Wavelet Forecasting.

    Target: Percent change in close price: (Close(t) - Close(t-1)) / Close(t-1).
    Inputs: Sequence of k past wavelets [W(t-1), W(t-2), ..., W(t-k)].
    """
    # The input shape is now defined by the lookback window size k
    input_shape = (k,) 
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


def _create_aligned_data(wavelet_diffs: pd.Series, k: int) -> pd.DataFrame:
    """
    Internal helper to create a synchronized DataFrame of lagged features and a target.

    For each time t, it creates:
    - Features: [Wd(t-1), Wd(t-2), ..., Wd(t-k)]
    - Target: Wd(t)

    It then aligns them by dropping rows with NaN values that result from the shifting.

    Args:
        wavelet_diffs (pd.Series): Series of wavelet differences ('Wd').
        k (int): The lookback window size.

    Returns:
        pd.DataFrame: A DataFrame with k feature columns and one 'target' column,
                      with NaN rows removed.
    """
    features = []
    for i in range(1, k + 1):
        features.append(wavelet_diffs.shift(i))

    df_combined = pd.concat([*features, wavelet_diffs.rename('target')], axis=1)
    df_combined.dropna(inplace=True)
    return df_combined

def create_inputs(historical_data: pd.DataFrame, k: int = 14) -> np.ndarray:
    """
    Creates input features for the Wavelet Difference Forecast model.
    The model uses a lookback window of k past wavelet differences (Wd(t-1) to Wd(t-k)).
    
    Args:
        historical_data (pd.DataFrame): DataFrame containing the 'Wd' (Wavelet Difference) column.
        k (int): The lookback window size.
        
    Returns:
        np.ndarray: A 2D array of shape (samples, k).
    """
    if 'Wd' not in historical_data.columns:
        raise ValueError("The 'Wd' (Wavelet Difference) column must be calculated before creating inputs.")

    wavelet_diffs = historical_data['Wd']
    
    # Create aligned data and return only the feature columns
    df_combined = _create_aligned_data(wavelet_diffs, k)

    return df_combined.iloc[:, :-1].values

def create_targets(historical_data: pd.DataFrame, k: int = 14) -> np.ndarray:
    """
    Creates the prediction target for the Wavelet Difference Forecast model.
    The target is the wavelet difference at time t: Wd(t).
    
    Args:
        historical_data (pd.DataFrame): DataFrame with 'Wd' column.
        k (int): The lookback window size used in create_inputs to ensure alignment.
        
    Returns:
        np.ndarray: A 1D array of target values.
    """
    if 'Wd' not in historical_data.columns:
        raise ValueError("The 'Wd' column must be in historical_data.")
        
    wavelet_diffs = historical_data['Wd']

    # Create aligned data and return only the target column
    df_combined = _create_aligned_data(wavelet_diffs, k)

    return df_combined['target'].values