import numpy as np
from qf.context import default_quantization_level

def quantize(time_series, quantization_level = default_quantization_level):
    quantized = (time_series * quantization_level).astype(np.int64)
    return quantized

def dequantize(quantized_time_series, quantization_level = default_quantization_level):
    return quantized_time_series / quantization_level