from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

def scale_features(X_train, X_val, X_test, scale_features = True):
    if scale_features:
        X_scaler = StandardScaler()    
        X_train_scaled = X_scaler.fit_transform(X_train)
        X_val_scaled = X_scaler.transform(X_val)    
        X_test_scaled = X_scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled, X_scaler
    else:
        return X_train, X_val, X_test, None

def scale_targets(Y_train, Y_val, Y_test):
    pass