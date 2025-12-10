from sklearn.linear_model import LogisticRegression
import qf.market as mkt
import numpy as np
import mplfinance as mpf
from qf.quantum.estimators import quantum_energy_levels, quantum_lambda
from qf.stats.distributions import empirical_distribution
import qf.nn as nn
import tensorflow as tf
import os
from qf.nn.models.base import priceangle, pricevoldiff, probdiff
from sklearn.linear_model import LinearRegression
layers = tf.keras.layers
regularizers = tf.keras.regularizers

base_model_names = ('prob', 'pricevol', 'priceangle')
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
def load_base_models(args):
    ticker = args.ticker.upper()
    base_model_path = lambda name: os.path.join(os.getcwd(), 'models', f'{ticker}-{name}.keras')    
    if not  os.path.exists(base_model_path):
        for name in base_model_names:
            args.model = name
            base_trainer(args)

    return tuple([tf.keras.models.load_model(base_model_path(name)) for name in base_model_names])

class CompoundModel:
    def __init__(self, prob_model, pricevol_model, priceangle_model):
        self.prob_model = prob_model
        self.pricevol_model = pricevol_model
        self.priceangle_model = priceangle_model

    def __call__(self, pricevol_inputs, prob_inputs, priceangle_inputs):
        pricevol_prediction = self.pricevol_model.predict(pricevol_inputs)
        prob_prediction = self.prob_model.predict(prob_inputs)
        priceangle_prediction = self.priceangle_model.predict(priceangle_inputs)
        
        # NOTE: Inverting Prob and Angle outputs due to contrarian nature
        stacked = np.hstack([
            pricevol_prediction, 
            -1.0 * prob_prediction, 
            -1.0 * priceangle_prediction
        ])

        return stacked

def trim_to_smallest(pricevol_inputs, prob_inputs, priceangle_inputs):
    min_len = min(len(pricevol_inputs), len(prob_inputs), len(priceangle_inputs))
    return  (pricevol_inputs[:min_len], prob_inputs[:min_len], priceangle_inputs[:min_len])

def create_inputs(historical_data, k, compound_model):
    (pricevol_inputs, prob_inputs, priceangle_inputs) =  trim_to_smallest(
        pricevoldiff.create_inputs(historical_data, k), 
        probdiff.create_inputs(historical_data, k),  
        priceangle.create_inputs(historical_data, k)
    )
    return compound_model(pricevol_inputs, prob_inputs, priceangle_inputs)

def create_targets(historical_data, min_len):
    """
    Calculates the PERCENTAGE difference between next close (t+1) and current close (t).
    Formula: (C(t+1) - C(t)) / C(t)
    """
    df = historical_data[:min_len].copy()
    # pct_change gives (C(t) - C(t-1))/C(t-1)
    # shift(-1) moves it to t, representing the NEXT bar's movement relative to current
    df['Cd'] = df['Close'].pct_change().shift(-1)
    
    # We drop NaNs to ensure alignment
    targets = df['Cd'].dropna().values
    return targets
    
def meta_trainer(args):
    (_, _, _, meta_train, meta_trade) = mkt.read_datasets(args.ticker)
    (prob_model, pricevol_model, priceangle_model) = load_base_models(args)
    compound_model = CompoundModel(prob_model, pricevol_model, priceangle_model)
    k = args.lookback    
    
    # Generate Inputs
    train_inputs_raw = create_inputs(meta_train, k, compound_model)
    trade_inputs_raw = create_inputs(meta_trade, k, compound_model)
    
    # Generate Targets (Percentage Change)
    # Note: create_targets drops the last row (NaN from shift), so we must trim inputs
    train_targets = create_targets(meta_train, len(train_inputs_raw) + 1) # +1 buffer for shift
    trade_targets = create_targets(meta_trade, len(trade_inputs_raw) + 1)
    
    # Align Inputs to Targets
    train_inputs = train_inputs_raw[:len(train_targets)]
    trade_inputs = trade_inputs_raw[:len(trade_targets)]
    
    # Train Meta-Learner (Linear Regression on % Change)
    meta_model = LinearRegression()
    meta_model.fit(train_inputs, train_targets)
    
    trade_predict = meta_model.predict(trade_inputs)    

    predict_sign = np.sign(trade_predict)
    expected_sign = np.sign(trade_targets)

    matching = predict_sign == expected_sign
    different = predict_sign != expected_sign
    matching_pct = np.count_nonzero(matching) / len(predict_sign)
    different_pct = np.count_nonzero(different) / len(predict_sign)

    output_file = os.path.join(os.getcwd(), "test-results", f"report-meta.md")
    mode = 'a' if os.path.exists(output_file) else 'w'
    with open(output_file, mode) as f:
        if mode == 'w':
            print("|ticker|Match %|Diff %|", file=f)
            print("|---|---|---|", file=f)

        print(f"|{args.ticker}|{matching_pct:.4f}|{different_pct:.4f}|", file=f)