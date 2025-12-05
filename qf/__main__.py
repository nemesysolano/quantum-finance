import qf.market as mkt
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from qf.quantum.estimators import quantum_energy_levels, quantum_lambda
from qf.stats.distributions import empirical_distribution
import sys
from scipy.interpolate import make_smoothing_spline
import qf.nn as nn
import qf.nn.propdiff as probdiff
import tensorflow as tf
import os

if __name__ == '__main__': # 
    ticker = sys.argv[1]
    k = 14
    l2_rate = 1e-6
    dropout_rate = 0.20

    historical_data = mkt.import_market_data(ticker)
    X_train, X_val, X_test = mkt.create_train_val_test(probdiff.create_inputs(historical_data, k))
    X_train_scaled, X_val_scaled, X_test_scaled, _ = nn.scale_features(X_train, X_val, X_test)
    Y_train, Y_val, Y_test= mkt.create_train_val_test(probdiff.create_targets(historical_data, k))    

    baseline_model = nn.propdiff.create_model(k, l2_rate, dropout_rate)
    checkpoint_filepath = os.path.join(os.getcwd(), 'models', f'{ticker}.keras')

    # UPDATED CALLBACK: Monitoring validation accuracy and setting mode='max'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=True,
        monitor='val_mse', # Monitor the classification metric
        mode='min' # We want to MAXIMIZE the accuracy
    )

    # UPDATED CALLBACK: Monitoring validation accuracy and setting mode='max'
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_mse', # Monitor the classification metric
        patience=10, # Early stopping patience
        mode='min', # We want to MAXIMIZE the accuracy
        restore_best_weights=True
    )

    # Train the model
    baseline_model.fit(
        X_train_scaled, 
        Y_train,
        epochs=25,
        batch_size=32, 
        validation_data=(X_val_scaled, Y_val),
        callbacks = [model_checkpoint_callback, early_stopping_callback]
    )

    # Load the best model
    best_model = tf.keras.models.load_model(checkpoint_filepath)

    # Evaluate the model on the test set
    # The output from evaluate() is (loss, accuracy) since the model is compiled with metrics=['accuracy']
    mse, mae = best_model.evaluate(X_test_scaled, Y_test, verbose=0) 
    
    # Predict and calculate direction match
    Y_pred_raw = best_model.predict(X_test_scaled).flatten()

    # Sign function converts predictions in [-1, 1] to -1 or +1
    Y_pred = np.int32(np.sign(Y_pred_raw)) 
    Y_expected = np.int32(np.sign(Y_test.flatten()))

    # Filters to findout Y_pred's signums and Y_expected's match.
    different = Y_pred != Y_expected
    matching  = Y_pred == Y_expected

    # Calculate matching rate and non-matching rates (they are complementary);
    matching_pct = np.count_nonzero(matching) / len(Y_pred)
    different_pct = np.count_nonzero(different) / len(Y_pred)

    output_file = os.path.join(os.getcwd(), "test-results", f"report.md")
    mode = 'a' if os.path.exists(output_file) else 'w'
    with open(output_file, mode) as f:
        if mode == 'w':
            print("## Report ##", file=f)
            print("## Model Setup ##", file=f)
            print(f"1. k={k}", file=f)
            print(f"2. l2_rate={l2_rate}", file=f)
            print(f"3. dropout_rate={dropout_rate}", file=f)
            print("## Results Table ##", file=f)
            print("|Ticker|MSE|MAE|Match %|Diff %|", file=f) # Updated table headers
            print("|---|---|---|---|---|", file=f)

        # Updated output line to reflect classification metrics
        print(f"|{ticker}|{mse:.4f}|{mae:.4f}|{matching_pct:.4f}|{different_pct:.4f}|", file=f)