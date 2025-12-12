import numpy as np
import pandas as pd
def quantum_lambda(r, f):
    max_f_index = f.idxmax()
    loc = f.index.get_loc(max_f_index)

    # r value before r    
    r_prev = r.loc[loc-1]
    f_prev = f.loc[loc-1]
    
    # r value after r
    r_after = r.loc[loc+1]
    f_after = f.loc[loc+1]

    q_l = np.abs(
        (np.power(r_prev, 2) * f_prev - np.power(r_after ,  2) * f_after) /  
        (np.power(r_after, 4) * f_after - np.power(r_prev, 4) * f_prev) 
    )
    
    return q_l

def quantum_energy_level(l, n):
    k_n = np.cbrt((1.1924 + 33.2383*n +  56.2169 * np.power(n,2))/ (1 + 43.6196*n))
    p = -np.power(2*n + 1, 2)
    q = -l * np.power(2*n + 1, 3) * np.power(k_n, 3)
    sub_sqrt = np.sqrt(np.power(q, 2) / 4 + np.power(p,3) / 27)
    half_q = q / 2
    E = np.cbrt(-half_q + sub_sqrt) + np.cbrt(-half_q - sub_sqrt)
    return E

def quantum_energy_levels(l, minimum, maximum):
    k = 1
    E1 = quantum_energy_level(l, k)    
    E0 = E1

    while E1 < minimum:        
        k += 1
        E0 = E1
        E1 = quantum_energy_level(l, k)

    E = [E0]        
    while E1 < maximum:        
        k += 1
        E1 = quantum_energy_level(l, k)
        E.append(E1)

    return E

def maximum_energy_level(x, l, max_n=100000):
    """
    Finds the highest energy level E^(n) that is strictly less than x.
    """
    last_E = -np.inf
    # Start from n=0 to calculate energy levels
    for n in range(max_n):
        E_n = quantum_energy_level(l, n)
        if E_n < x:
            last_E = E_n
        else:
            # Energy levels are monotonically increasing, so we can stop
            # once we find a level greater than or equal to x.
            break
    # Return the last found level that was less than x, or nan if none were.
    return last_E if last_E > -np.inf else np.nan

def minimum_energy_level(x, l, max_n=100000):
    """
    Finds the lowest energy level E^(n) that is strictly greater than x.
    """
    # Start from n=0 to calculate energy levels
    for n in range(max_n):
        E_n = quantum_energy_level(l, n)
        if E_n > x:
            return E_n
    # If no level is found greater than x within max_n, return nan.
    return np.nan
