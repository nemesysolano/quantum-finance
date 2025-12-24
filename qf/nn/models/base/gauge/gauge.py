import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, constraints
import numpy as np

@tf.keras.utils.register_keras_serializable(package="QuantumTrading")
class FractionalDiffLayer(layers.Layer):
    """
    Implements the Differentiated Time Series formula:
    x_hat(t) = sum_{i=0}^k w_i * x(t-i)
    where w_i = -w_{i-1} * (d - i + 1) / i
    """
    def __init__(self, k, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        # Initialize d as a trainable parameter in [0, 1]
        self.d = self.add_weight(
            name="d_order",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(0.5),
            constraint=constraints.MinMaxNorm(min_value=0.0, max_value=1.0),
            trainable=True
        )

    def call(self, inputs):
        # inputs shape: (batch, k, 1)
        # 1. Generate weights based on current d
        w = [tf.ones_like(self.d)] # w_0 = 1
        for i in range(1, self.k):
            next_w = -w[-1] * (self.d - float(i) + 1.0) / float(i)
            w.append(next_w)
        
        # Stack weights and reshape to (1, k, 1) for broadcasting
        # This replaces the faulty tf.stack + tf.transpose logic
        weights_tensor = tf.reshape(tf.stack(w), (1, self.k, 1))
        
        # 2. Perform the summation: sum(w_i * x(t-i))
        # Multiplication broadcasts weights_tensor across the batch dimension
        weighted_inputs = inputs * weights_tensor
        diff_series = tf.reduce_sum(weighted_inputs, axis=1)
        
        return diff_series
    
    def get_config(self):
        config = super().get_config()
        config.update({"k": self.k})
        return config

def create_model(k, l2_rate, dropout_rate):
    """
    Reimplemented model for Schrödinger Gauge Difference Forecast.
    Integrates FractionalDiffLayer to find optimal differentiation order d.
    """
    l2_reg = regularizers.l2(l2_rate) if l2_rate > 0 else None
    
    # --- Input Stage ---
    # Shape: (k lookback steps, 1 feature)
    raw_input = layers.Input(shape=(k, 1), name="price_history")
    
    # --- Stage 1: Fractional Differentiation (Optimal d Training) ---
    # Transforms non-stationary input into stationary memory-preserved series
    stationary_features = FractionalDiffLayer(k=k, name="frac_diff_stage")(raw_input)
    
    # --- Stage 2: Deep Feed-Forward Network ---
    x = layers.Dense(128, kernel_regularizer=l2_reg)(stationary_features)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(64, kernel_regularizer=l2_reg)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(32, kernel_regularizer=l2_reg)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    
    # Output: Predicted Schrödinger Gauge Difference Öd(τ)
    output = layers.Dense(1, activation='linear', name="target_forecast")(x)
    
    model = keras.Model(inputs=raw_input, outputs=output, name="Quantum_FracDiff_Model")
    
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
    