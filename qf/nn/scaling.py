from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

def scale_features(X_train, X_val, X_test, scale_features = True):
    print(X_train.shape)
    
    if scale_features:
        if len(X_train.shape) == 2:
            X_scaler = StandardScaler()    
            X_train_scaled = X_scaler.fit_transform(X_train)
            X_val_scaled = X_scaler.transform(X_val)    
            X_test_scaled = X_scaler.transform(X_test)
            return X_train_scaled, X_val_scaled, X_test_scaled, X_scaler
        elif len(X_train.shape) == 3:
            X_scaler = StandardScaler()
            X_train_scaled = X_scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
            X_val_scaled = X_scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
            X_test_scaled = X_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
            return X_train_scaled, X_val_scaled, X_test_scaled, X_scaler
    else:
        return X_train, X_val, X_test, None

def scale_targets(Y_train, Y_val, Y_test):
    pass