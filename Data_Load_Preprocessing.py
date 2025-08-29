import numpy as np
import pyabf
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.interpolate import interp1d
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import os # Imported for file existence checks

## --- Data Loading Functions --- ##

def import_ABF_trace(file_path):
    """Import ABF file and extract sweep data
    sweep[:,0] is the time data
    sweep[:,1] is the current data
    Assumes 1 trace in the ABF file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ABF file not found at path: {file_path}")
    abf = pyabf.ABF(file_path)
    print(abf) #Prints information about the ABF file
    sample_rate = abf.dataRate
    raw_sweep = np.column_stack((abf.sweepX, abf.sweepY)) #Stack the sweepX and sweepY arrays horizontally
    return raw_sweep, sample_rate

def invert_trace(data):
    '''Invert the data'''
    data[:,1] = -data[:,1]
    return data

## --- Data Processing Functions --- ##

def low_pass_filter(data_signal, cutoff_freq, sample_rate, order=5):
    '''Standard Low Pass Butterworth Filter'''
    nyquist = 0.5 * sample_rate # Nyquist frequency is half the sampling rate
    if cutoff_freq >= nyquist:
        print(f"Warning: Cutoff frequency ({cutoff_freq} Hz) is >= Nyquist frequency ({nyquist} Hz). Skipping filter.")
        return data_signal
    normal_cutoff = cutoff_freq / nyquist # normalised cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    try:
        y = filtfilt(b, a, data_signal) # Apply the filter to the data using filtfilt
        return y
    except ValueError as e:
        print(f"Error applying filtfilt (likely due to data length): {e}. Returning original data.")
        return data_signal


def sample_time(data, desired_sample_time_us, sample_rate):
    '''Downsample the data to the desired sample time'''
    original_sample_time_us = 1e6 / sample_rate
    downsample_factor = int(round(desired_sample_time_us / original_sample_time_us))
    if downsample_factor <= 0:
        downsample_factor = 1
        print("Warning: Desired sample time is less than original. No downsampling performed.")
    elif downsample_factor >= len(data):
         downsample_factor = 1
         print("Warning: Downsample factor is larger than data length. No downsampling performed.")

    print(f"Downsampling by factor: {downsample_factor}")
    data_downsample = data[::downsample_factor]
    new_sample_rate = sample_rate / downsample_factor
    return data_downsample, new_sample_rate

def baseline_moving_average(y, window_size=10001):
    """Estimate a baseline using a moving average."""
    if window_size > len(y):
        print(f"Warning: Window size ({window_size}) > data length ({len(y)}). Using data length as window size.")
        window_size = len(y)
    if window_size < 1:
        window_size = 1
    if window_size % 2 == 0:
        window_size += 1 # Ensure odd window size
        print(f"Adjusted window size to odd number: {window_size}")

    cumsum = np.cumsum(np.insert(y, 0, 0))
    baseline = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    pad_size = (window_size - 1) // 2 # Pad the result to match the length of the input signal
    baseline = np.pad(baseline, (pad_size, pad_size), mode='edge')
    return baseline

def baseline_als(y, lam=1e6, p=0.01, niter=10):
    """
    Estimate a baseline using Asymmetric Least Squares Smoothing using sparse matrices.

    Parameters:
      y (array-like): The input signal (1D array).
      lam (float): Smoothness parameter. Higher values enforce a smoother baseline. 1e6 - 1e8
      p (float): Asymmetry parameter. Lower values favor fitting below the data. 0.001 - 0.01
      niter (int): Number of iterations for the reweighting procedure. 10 - 20

    Returns:
      baseline (np.array): The fitted baseline.
    """
    L = len(y) # Construct the sparse second-order difference matrix D with shape (L-2, L)
    if L == 0: return np.array([])
    D = diags([1, -2, 1], [0, 1, 2], shape=(L-2, L), format='csr') # Use CSR format
    DTD = D.T @ D # Precompute D.T @ D (still sparse)
    w = np.ones(L)
    for i in range(niter):
        W = diags(w, 0, format='csr') # Use CSR format to construct a sparse diagonal weight matrix W
        A = W + lam * DTD # The system to solve: (W + lam * D.T@D) * baseline = W * y
        try:
            baseline = spsolve(A, w * y)
        except Exception as e:
            print(f"Error in spsolve during baseline_als iteration {i+1}: {e}. Returning previous baseline or mean.")
            if 'baseline' in locals(): return baseline # Return last successful baseline
            else: return np.full_like(y, np.mean(y)) # Fallback

        w = p * (y > baseline) + (1 - p) * (y < baseline) # Update weights: points above the baseline get weight p, below get weight (1-p)
    return baseline

def baseline_SG(y, window_length = 1001, polyorder = 2):
    """ Fit a baseline using a Savitzky-Golay filter."""
    if window_length > len(y):
        print(f"Warning: SG filter window length ({window_length}) > data length ({len(y)}). Adjusting window length.")
        window_length = len(y)
    if window_length % 2 == 0: window_length -=1 # Must be odd
    if window_length < polyorder + 1:
         print(f"Warning: SG filter window length ({window_length}) must be > polyorder ({polyorder}). Adjusting polyorder.")
         polyorder = window_length - 1
         if polyorder < 1: # Should not happen if window_length >= 1
             print("Error: Cannot apply SG filter with current parameters.")
             return np.full_like(y, np.mean(y)) # Fallback

    if window_length <= 0: # Final check
        print("Error: SG filter window length is non-positive. Returning mean.")
        return np.full_like(y, np.mean(y)) # Fallback

    baseline = savgol_filter(y, window_length, polyorder)
    return baseline

def interpolate_baseline(x_down, baseline_down, x_full, method='linear'):
    """Interpolates a downsampled baseline back to full resolution."""
    if len(x_down) < 2:
        print("Warning: Not enough points for interpolation. Returning constant baseline.")
        return np.full_like(x_full, np.mean(baseline_down) if len(baseline_down)>0 else 0)
    try:
        interp_func = interp1d(x_down, baseline_down, kind=method, fill_value="extrapolate", bounds_error=False)
        return interp_func(x_full)
    except ValueError as e:
        print(f"Interpolation failed: {e}. Returning constant baseline.")
        return np.full_like(x_full, np.mean(baseline_down) if len(baseline_down)>0 else 0)

def baseline_offset(baseline, data_signal, manual_offset=0):
    """ Add an offset to the baseline. Typically not needed for normalized data."""
    baseline -= manual_offset # Add a manual offset to the baseline
    return baseline

def normalise_trace(data_array, baseline_method='als', current_sample_rate = 125000.00, **kwargs):
    """ Normalize the signal by estimating and subtracting the baseline."""
    y = data_array[:, 1]
    x = data_array[:, 0]

    # Option to downsample for baseline calculation and then interpolate
    downsample_baseline = kwargs.get('downsample_baseline', False)
    if downsample_baseline:
        print("Downsampling for baseline calculation...")
        baseline_sample_time = kwargs.get('baseline_sample_time_us', 30)
        data_baseline_down, _ = sample_time(data_array, baseline_sample_time, current_sample_rate)
        y_down = data_baseline_down[:,1]
        x_down = data_baseline_down[:,0]

        if baseline_method == 'als':
            baseline_down = baseline_als(y_down, **{k: v for k,v in kwargs.items() if k not in ['downsample_baseline', 'baseline_sample_time_us','manual_offset']})
        elif baseline_method == 'sg':
            baseline_down = baseline_SG(y_down, **{k: v for k,v in kwargs.items() if k not in ['downsample_baseline', 'baseline_sample_time_us','manual_offset']})
        elif baseline_method == 'moving_average':
             baseline_down = baseline_moving_average(y_down, **{k: v for k,v in kwargs.items() if k not in ['downsample_baseline', 'baseline_sample_time_us','manual_offset']})

        baseline = interpolate_baseline(x_down, baseline_down, x)

    else:
        # Select baseline method
        if baseline_method == 'als':
            baseline = baseline_als(y, **{k: v for k,v in kwargs.items() if k not in ['downsample_baseline', 'baseline_sample_time_us','manual_offset']})
        elif baseline_method == 'sg':
            baseline = baseline_SG(y, **{k: v for k,v in kwargs.items() if k not in ['downsample_baseline', 'baseline_sample_time_us','manual_offset']})
        elif baseline_method == 'moving_average':
            baseline = baseline_moving_average(y, **{k: v for k,v in kwargs.items() if k not in ['downsample_baseline', 'baseline_sample_time_us','manual_offset']})
        else:
            raise ValueError("Invalid baseline method specified. Choose 'als', 'sg', or 'moving_average'.")

    # --- Apply manual offset AFTER baseline calculation ---
    manual_offset = kwargs.get('manual_offset', 0)
    baseline = baseline_offset(baseline, y, manual_offset=manual_offset) # Pass signal for potential future STDV use

    # Subtract baseline
    normalized_y = y - baseline
    normalized_data = data_array.copy()
    normalized_data[:, 1] = normalized_y
    return normalized_data, baseline