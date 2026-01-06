import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

features = [
    'Ö', 'Öd', 'Ödd',              # Raw Physics
    'GRU_Forecast',                # Expert Opinion (Continuous)
    'GRU_Sign',                    # Expert Opinion (Binary)
    'Kinematic_Alignment',         # Derived Physics
    'Wall_Proximity',              # Derived Physics
    'Reversal_Risk'                # Derived Physics
]
target = 'Target_Return'

class GaugeMetaFeatures:
    def __init__(self, gru_model, lookback):
        """
        Args:
            gru_model: The trained Keras/TensorFlow model (GRU or Dense).
            lookback (int): The window size 'k' used during GRU training.
        """
        self.model = gru_model
        self.k = lookback

    def create_meta_dataset(self, df):
        """
        Main pipeline to transform raw Gauge data into a Meta-Model dataset.
        
        Args:
            df (pd.DataFrame): Dataframe with 'Ö', 'Öd', 'Ödd', 'Close'.
            
        Returns:
            pd.DataFrame: Cleaned dataset ready for Meta-Model training/inference.
        """
        # 1. Generate the "Expert Opinion" (GRU Predictions)
        print(f"Generating GRU forecasts for {len(df)} rows...")
        df_meta = self.add_gru_predictions(df.copy())
        
        # 2. Add Physics Interaction Features (The "Logic" from our chat)
        df_meta = self.add_physics_interactions(df_meta)
        
        # 3. Create the Target (Future Return)
        # We predict Return at t+1 based on information at t
        df_meta['Target_Return'] = df_meta['Close'].pct_change().shift(-1)
        
        # 4. Clean up (Drop NaNs from lookback and shifting)
        df_ready = df_meta.dropna()
        
        print(f"Meta-Dataset created. Shape: {df_ready.shape}")
        return df_ready

    def add_gru_predictions(self, df):
        """
        Internal method to generate predictions using the pre-trained GRU.
        Constructs the (Samples, k, 3) 3D tensor on the fly.
        """
        # Ensure we have the raw inputs
        req_cols = ['Ö', 'Öd', 'Ödd']
        if not all(c in df.columns for c in req_cols):
            raise ValueError(f"Dataframe missing required columns: {req_cols}")

        # Construct 3D input tensor using rolling window
        # We need to turn the dataframe into (N, k, 3)
        data_values = df[req_cols].values
        X_list = []
        
        # We start from index k to have enough history
        # Note: This loop can be slow for massive datasets; strictly for accuracy here
        # For production, utilize stride_tricks or TimeseriesGenerator
        for i in range(self.k, len(data_values) + 1):
            # Grab the window from i-k to i
            window = data_values[i-self.k : i]
            X_list.append(window)
            
        X_3d = np.array(X_list) # Shape: (Samples, k, 3)
        
        # Generate Predictions
        # These are predictions for time t (aligned with the last row of the window)
        preds = self.model.predict(X_3d, verbose=0)
        
        # Align predictions to the Dataframe
        # The first k rows have no prediction (insufficient history)
        # We create a full-length array filled with NaNs first
        full_preds = np.full((len(df),), np.nan)
        
        # Fill the valid slots. 
        # Since X_list starts at index k (which corresponds to the k-th row in df),
        # we align preds[0] with df.iloc[k-1] or df.iloc[k]? 
        # Standard: Input t-k...t predicts t+1? 
        # YOUR LOGIC: Input t-k...t predicts Öd(t+1) direction.
        # So the prediction generated at index 't' belongs to index 't'.
        full_preds[self.k-1:] = preds.flatten()
        
        df['GRU_Forecast'] = full_preds
        
        # We also extract the "Conviction" (Magnitude)
        df['GRU_Conviction'] = df['GRU_Forecast'].abs()
        
        # And the "Sign" (Direction)
        df['GRU_Sign'] = np.sign(df['GRU_Forecast'])
        
        return df

    def add_physics_interactions(self, df):
        """
        Adds 'Smart' features that help the Meta-Model understand context.
        """
        # 1. The "Kinematic State": Is force aligned with velocity?
        # +1 if accelerating in direction of motion, -1 if decelerating
        df['Kinematic_Alignment'] = np.sign(df['Öd']) * np.sign(df['Ödd'])
        
        # 2. The "Wall Proximity": How close are we to boundaries?
        # High value = Near boundary (danger zone)
        df['Wall_Proximity'] = df['Ö'].abs()
        
        # 3. The "Contrarian Trigger": High Position + Negative Acceleration
        # 1 = Potential Top, -1 = Potential Bottom, 0 = Normal
        # We use a threshold of 0.8 for 'High'
        conditions = [
            (df['Ö'] > 0.8) & (df['Ödd'] < 0),  # Top Reversal Warning
            (df['Ö'] < -0.8) & (df['Ödd'] > 0)  # Bottom Reversal Warning
        ]
        choices = [-1, 1] # Sell signal, Buy signal
        df['Reversal_Risk'] = np.select(conditions, choices, default=0)
        
        # 4. The "Conflict": Does the Model disagree with the current trend?
        # 1 = Disagreement (Model says Up, Price moving Down), 0 = Agreement
        # Note: Öd is current velocity. GRU_Sign is predicted future velocity.
        df['Model_Divergence'] = (np.sign(df['Öd']) != df['GRU_Sign']).astype(int)
        
        return df
    
def gauge_meta_features(gru_model, lookback, historica_dataset):
    features = GaugeMetaFeatures(gru_model, lookback)
    features_dataset = features.create_meta_dataset(historica_dataset)
    return features_dataset

def train_gauge_meta_ensemble(meta_train, meta_test):
    """
    Trains XGBoost to filter GRU signals based on Physics context.
    """
    # Features engineered in the previous step
    
    # Target: Binary direction (1 for Price Up, 0 for Price Down)
    meta_X = lambda df: df[features]
    meta_y = lambda df: (df[target].shift(-1) > 0).astype(int)

    # Chronological Split (No shuffling for time-series!)
    X_train, X_test = meta_X(meta_train), meta_X(meta_test)
    y_train, y_test = meta_y(meta_train), meta_y(meta_test)
    
    # Drop the last row to remove the NaN introduced by shift(-1)
    X_train, y_train = X_train[:-1], y_train[:-1]

    # XGBoost setup for financial 'noise' reduction
    # We use a shallow max_depth to ensure we don't overfit to specific dates
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate probability
    probs = model.predict_proba(X_test)[:, 1]
    return model, X_test, y_test, probs