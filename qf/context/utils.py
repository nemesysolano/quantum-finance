import numpy as np

def bounded_percentage_difference(a, b):
    """
    Calculates Δ%(a, b) = (b - a) / (|a| + |b|)
    """
    # Using np.abs to handle both scalars and potential numpy arrays
    denominator = np.abs(a) + np.abs(b)
    
    # Handle the non-zero requirement mentioned in README to avoid division by zero
    if isinstance(denominator, np.ndarray):
        denominator[denominator == 0] = np.nan 
    elif denominator == 0:
        return 0.0
        
    return (b - a) / denominator

def serial_difference(series, k=1):
    """
    Calculates Δ(x(t), k) = Δ%(x(t-k), x(t))
    If k=1, it calculates the difference from the immediate previous element.
    """
    if len(series) <= k:
        return None # Not enough data points for lookback k
    
    # x(t) is the last element, x(t-k) is the element k steps back
    a = series[-(k+1)]
    b = series[-1]
    
    return bounded_percentage_difference(a, b)

def squared_serial_difference(series, k=1):
    """
    Calculates Δ²(x(t), k) = [Δ(x(t), k)]²
    """
    diff = serial_difference(series, k)
    if diff is None:
        return None
        
    return diff ** 2