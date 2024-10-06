# utils.py

import numpy as np
from filterpy.kalman import KalmanFilter

def initialize_kalman_filter(initial_value, measurement_noise=1.0, process_noise=1.0):
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[initial_value], [0]])  # Initial state (value and its rate)
    kf.F = np.array([[1, 1],
                     [0, 1]])  # State transition matrix
    kf.H = np.array([[1, 0]])  # Measurement function
    kf.P *= 1000.0  # Covariance matrix
    kf.R = measurement_noise  # Measurement noise
    kf.Q = np.array([[process_noise, 0],
                     [0, process_noise]])  # Process noise
    return kf

def exponential_smooth(current_value, previous_value, alpha=0.05):
    if previous_value is None:
        return current_value
    return previous_value + alpha * (current_value - previous_value)
