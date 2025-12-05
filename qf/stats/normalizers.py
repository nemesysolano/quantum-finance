import numpy as np
from qf.context import default_quantization_delta

def quantize(time_series, quantization_level = default_quantization_delta):
    quantized = (time_series * default_quantization_delta).astype(np.int64)
    return quantized

def dequantize(quantized_time_series, quantization_level = default_quantization_delta):
    return quantized_time_series / default_quantization_delta