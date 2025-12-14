from sklearn.linear_model import LogisticRegression
import qf.market as mkt
import numpy as np
import mplfinance as mpf
from qf.quantum.estimators import quantum_energy_levels, quantum_lambda
from qf.stats.distributions import empirical_distribution
import qf.nn as nn
import tensorflow as tf
import os
from qf.nn.models.base import priceangle, pricevoldiff, probdiff, gauge
from sklearn.linear_model import LinearRegression
layers = tf.keras.layers
regularizers = tf.keras.regularizers

base_model_names = ('prob', 'pricevol', 'priceangle', 'gauge')
# -- base trainer module
def base_trainer(args):
    ticker = args.ticker.upper()
    patience = args.patience
    model_factory_name = args.model
    epochs = args.epochs
    lookback = args.lookback
    scale_features = args.scale_features == 'yes'
    model_factory = nn.base_model_factories[model_factory_name]
     
    k = 8 if lookback < 8 or lookback > 30 else lookback
    l2_rate = args.l2_rate
    dropout_rate = args.dropout_rate

    historical_data = mkt.import_market_data(ticker)
    X_train, X_val, X_test, _, _ = mkt.create_datasets(model_factory.create_inputs(historical_data, k))
    X_train_scaled, X_val_scaled, X_test_scaled, _ = nn.scale_features(X_train, X_val, X_test, scale_features)
    Y_train, Y_val, Y_test, _, _= mkt.create_datasets(model_factory.create_targets(historical_data, k))    

    baseline_model = model_factory.create_model(k, l2_rate, dropout_rate)
    checkpoint_filepath = os.path.join(os.getcwd(), 'models', f'{ticker}-{model_factory_name}.keras')

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=True,
        monitor='val_mse',
        mode='min'
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_mse',
        patience=patience,
        mode='min',
        restore_best_weights=True
    )

    baseline_model.fit(
        X_train_scaled, 
        Y_train,
        epochs=epochs,
        batch_size=32, 
        validation_data=(X_val_scaled, Y_val),
        callbacks = [model_checkpoint_callback, early_stopping_callback]
    )

    best_model = tf.keras.models.load_model(checkpoint_filepath)
    mse, mae = best_model.evaluate(X_test_scaled, Y_test, verbose=0) 
    
    Y_pred_raw = best_model.predict(X_test_scaled).flatten()
    Y_pred = np.int32(np.sign(Y_pred_raw)) 
    Y_expected = np.int32(np.sign(Y_test.flatten()))

    matching  = Y_pred == Y_expected
    different = Y_pred != Y_expected
    matching_pct = np.count_nonzero(matching) / len(Y_pred)
    different_pct = np.count_nonzero(different) / len(Y_pred)

    output_file = os.path.join(os.getcwd(), "test-results", f"report-{model_factory_name}.md")
    mode = 'a' if os.path.exists(output_file) else 'w'
    with open(output_file, mode) as f:
        if mode == 'w':
            print("## Report ##", file=f)
            print("## Model Setup ##", file=f)
            print(f"1. k={k}", file=f)
            print(f"2. l2_rate={l2_rate}", file=f)
            print(f"3. dropout_rate={dropout_rate}", file=f)
            print(f"4. scale features={args.dropout_rate}", file=f)
            print("## Results Table ##", file=f)
            print("|Ticker|MSE|MAE|Match %|Diff %|", file=f)
            print("|---|---|---|---|---|", file=f)

        print(f"|{ticker}|{mse:.4f}|{mae:.4f}|{matching_pct:.4f}|{different_pct:.4f}|", file=f)

#--- meta trainer module
