import pickle
import numpy as np
import scipy.constants as const # Used for the speed of light (c)

def calibration_from_range(rx_iq, t, W, R_cal):
    """
    Calibrates the RX signal using Bandwidth and Calibration Range.
    """
    with open('rx_calibration.pkl', 'rb') as f:
        a_IF_c = pickle.load(f)
        
    c = const.c # Speed of light in a vacuum (approx 3e8 m/s)
    
    # Compute the correction factor: exp((4 * pi * j * W * R_cal * t) / c)
    correction_factor = np.exp((1j * 4 * np.pi * W * R_cal * t) / c)
    
    # Apply the calibration equation
    a_calibrated = (rx_iq / a_IF_c) * correction_factor
    
    return a_calibrated