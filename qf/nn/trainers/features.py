import qf.nn.models.base.pricevoldiff as pv_lib
import qf.nn.models.base.probdiff as pd_lib
import qf.nn.models.base.wavelets as wav_lib
import qf.nn.models.base.gauge as gauge_lib
import qf.nn.models.base.barinbalance as bar_lib
import numpy as np

def extract_meta_features(historical_data, models, k=14):
    """Stacks predictions from all base models as features."""
    m_pv, m_ang, m_g, m_pd, m_bar = models
    X_pv = pv_lib.create_inputs(historical_data, k)
    X_wav = wav_lib.create_inputs(historical_data, k) 
    X_g = gauge_lib.create_inputs(historical_data, k)
    X_pd = pd_lib.create_inputs(historical_data, k)
    X_bar = bar_lib.create_inputs(historical_data, k)

    min_samples = min(len(X_pv), len(X_wav), len(X_g))
    X_pv, X_wav, X_g, X_pd, X_bar = X_pv[-min_samples:], X_wav[-min_samples:], X_g[-min_samples:], X_pd[-min_samples:], X_bar[-min_samples:]
    
    y_pv = m_pv.predict(X_pv, verbose=0).flatten()
    y_wav = m_ang.predict(X_wav, verbose=0).flatten()
    y_g = m_g.predict(X_g, verbose=0).flatten()
    y_pd = m_pd.predict(X_pd, verbose=0).flatten()
    y_bar = m_bar.predict(X_bar, verbose=0).flatten()

    return np.column_stack([y_pv, y_wav, y_g, y_pd, y_bar])

def get_limits(close_t, energy_levels, direction, risk_pct=0.01):    
    risk_dollars, reward_dollars = 0, 0
    if isinstance(energy_levels, tuple):
        e_low, e_high = energy_levels
        if direction == 1:
            risk_dollars = close_t - e_low
            reward_dollars = e_high - close_t
        elif direction == -1:
            risk_dollars = e_high - close_t
            reward_dollars = close_t - e_low
    
    risk_dollars = min(risk_dollars, close_t * risk_pct)
    reward_dollars = max(risk_dollars * 3, reward_dollars)
    return risk_dollars, reward_dollars