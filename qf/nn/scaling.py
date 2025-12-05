from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

def scale_features(X_train, X_val, X_test):
    X_scaler = StandardScaler()    
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_val_scaled = X_scaler.transform(X_val)    
    X_test_scaled = X_scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, X_scaler


def scale_targets(Y_train, Y_val, Y_test):
    pass