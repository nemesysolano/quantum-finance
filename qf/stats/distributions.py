from collections import namedtuple
import pandas as pd
import numpy as np
from scipy import stats
from qf.stats.normalizers import dequantize, quantize
NormalDistribution = namedtuple('NormalDistribution', ['mean', 'std'])
from qf.context import default_quantization_level
def normal_distribution(time_series, alpha = 0.05):
    _, p_value = stats.normaltest(time_series)
    if p_value > alpha:
        return NormalDistribution(np.mean(time_series), np.std(time_series))    
    return None

def empirical_distribution(time_series, quantization_level = default_quantization_level):
    unique, counts = np.unique(quantize(time_series, quantization_level), return_counts=True)
    return pd.DataFrame({'X': dequantize(unique, quantization_level), 'P': counts / np.sum(counts), 'C': counts})
