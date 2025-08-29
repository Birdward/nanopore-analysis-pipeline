import numpy as np

# Helper function for steady-state detection
def _detect_steady_state(event_signal, noise_std, 
                         min_duration_points_for_detection, 
                         plateau_slope_thresh_factor, 
                         plateau_ratio_threshold):
    """
    Detects if an event signal exhibits steady-state behavior.
    
    Parameters:
        event_signal (np.ndarray): The signal values for the event.
        noise_std (float): Standard deviation of the baseline noise.
        min_duration_points_for_detection (int): Minimum number of points the event_signal must have.
        plateau_slope_thresh_factor (float): Factor to multiply noise_std to get slope threshold.
        plateau_ratio_threshold (float): Minimum fraction of derivative points below slope threshold.
        
    Returns:
        tuple: (bool_steady_state_detected, str_reason)
    """
    duration_points = len(event_signal)

    # Event needs to have enough points to calculate a meaningful derivative and ratio
    if duration_points < min_duration_points_for_detection:
        return False, f"Signal too short ({duration_points} points, min {min_duration_points_for_detection}) for steady state detection"

    # Derivative calculation requires at least 2 points in event_signal
    if duration_points <= 1: 
        return False, f"Signal too short ({duration_points} points, needs >1) for derivative calculation"

    derivative = np.abs(np.diff(event_signal))
    
    # As a safeguard:
    if len(derivative) == 0:
        return False, "Derivative calculation resulted in empty array (event likely too short)"

    # Threshold for what constitutes a plateau (low derivative)
    # This is the maximum slope magnitude to be considered part of a plateau
    plateau_slope_limit = noise_std * plateau_slope_thresh_factor
    
    num_plateau_points = np.sum(derivative < plateau_slope_limit)
    percentage_plateau = num_plateau_points / len(derivative)
    
    reason = f"Plateau points: {percentage_plateau*100:.1f}% (threshold: {plateau_ratio_threshold*100:.1f}%)"
    if percentage_plateau > plateau_ratio_threshold:
        return True, reason + " -> Steady state detected"
    else:
        return False, reason + " -> No significant steady state"


def classify_events(data, event_boundaries, noise_std, noise_mean, sample_rate,
                    short_event_duration_ms=0.5, snr_threshold=4.0, default="ADEPT",
                    min_cusum_duration_ms=0.032,
                    steady_state_min_duration_points=5,
                    steady_state_plateau_slope_factor=0.7,
                    steady_state_plateau_ratio_threshold=0.3):
    """
    Classify events as ADEPT (short/spike) or CUSUM (long/step) based on duration, SNR, and steady-state behavior.
    
    Parameters:
        data (np.ndarray): 2D numpy array. First column is time (in seconds), second column is signal.
        event_boundaries (list): List of tuples (start_index, end_index) for each event.
                                 'end_index' is exclusive, as in Python slicing.
        noise_std (float): Standard deviation of the baseline noise.
        noise_mean (float): Mean of the baseline noise. (Note: Currently not used in the core
                            classification logic if data is assumed to be baseline-subtracted for SNR calc).
        sample_rate (float): Sampling rate in Hz. Used for accurate duration calculation.
        short_event_duration_ms (float): Threshold in milliseconds. Events with duration <= this value
                                         are primarily considered for ADEPT, otherwise CUSUM.
        snr_threshold (float): Signal-to-noise ratio threshold. Events with SNR < this value favor ADEPT.
        default (str): Default classification ("ADEPT" or "CUSUM") applied initially.
        min_cusum_duration_ms (float): Absolute minimum duration in ms for an event to potentially be
                                       classified as CUSUM. Events shorter than this are always ADEPT.
        steady_state_min_duration_points (int): Minimum number of data points an event must have
                                                to perform steady-state detection.
        steady_state_plateau_slope_factor (float): Factor multiplied by noise_std to determine the
                                                   slope threshold for identifying a plateau region.
        steady_state_plateau_ratio_threshold (float): Minimum fraction of an event's derivative points
                                                      that must be part of a plateau to classify
                                                      the event as having a steady state.

    Returns:
        tuple: (adept_events, cusum_events)
               Each is a list of dictionaries:
               {'event_idx': int, 'start': int, 'end': int, 'reasons': str (semicolon-separated)}
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Data must be a 2D numpy array with at least two columns (time, signal).")

    data_signal = data[:, 1]
    time_vec = data[:, 0]
    adept_events = []
    cusum_events = []

    if noise_std <= 0:
        # Setting to a small epsilon (1e-9) to avoid division by zero/issues.")
        noise_std = 1e-9  # Use a small positive epsilon

    for i, (start, end) in enumerate(event_boundaries):
        # Skip invalid or zero-length boundaries
        if start >= end:
            print(f"Debug: Skipping event {i} due to invalid/empty boundaries: start={start}, end={end}")
            continue

        event_signal = data_signal[start:end]
        duration_points = len(event_signal) # Same as end - start

        # This should not happen if start >= end is checked, but as an extra safeguard
        if duration_points == 0:
            print(f"Debug: Skipping event {i} due to zero points despite start < end: start={start}, end={end}")
            continue

        # --- Calculate event duration in ms ---
        # The total duration covered by 'duration_points' samples.
        duration_s = 0.0
        if sample_rate > 0:
            duration_s = duration_points / sample_rate
        elif duration_points > 1 : # Fallback if sample_rate is invalid, use time_vec span
            duration_s = time_vec[end - 1] - time_vec[start]
            # Event {i}: sample_rate invalid (<=0). Duration calculated from time_vec span, might be inaccurate for total extent

        duration_ms = duration_s * 1000

        # --- Calculate event statistics (SNR) ---
        # Assumes event_signal is relative to a baseline of 0 (i.e., baseline-subtracted).
        event_max_abs_deviation = np.max(np.abs(event_signal))
        snr = event_max_abs_deviation / noise_std if noise_std > 0 else float('inf')

        # --- Decision Logic ---
        reasons = []
        
        # 1. Initial decision based on `default` parameter
        current_decision_is_cusum = (default.upper() == "CUSUM")
        reasons.append(f"Initial: {default.upper()}")

        # 2. Primary classification based on `short_event_duration_ms`
        if duration_ms > short_event_duration_ms:
            if not current_decision_is_cusum: # It was ADEPT by default
                reasons.append(f"Duration ({duration_ms:.3f}ms > {short_event_duration_ms}ms) -> CUSUM")
            else: # It was CUSUM by default
                reasons.append(f"Duration ({duration_ms:.3f}ms > {short_event_duration_ms}ms) confirms CUSUM")
            current_decision_is_cusum = True
        else:  # duration_ms <= short_event_duration_ms
            if current_decision_is_cusum: # It was CUSUM by default
                reasons.append(f"Duration ({duration_ms:.3f}ms <= {short_event_duration_ms}ms) -> ADEPT")
            else: # It was ADEPT by default
                reasons.append(f"Duration ({duration_ms:.3f}ms <= {short_event_duration_ms}ms) confirms ADEPT")
            current_decision_is_cusum = False
        
        # 3. Steady-state detection (favors CUSUM, can override ADEPT from duration rule)
        steady_state_detected, ss_reason = _detect_steady_state(
            event_signal, noise_std,
            steady_state_min_duration_points,
            steady_state_plateau_slope_factor,
            steady_state_plateau_ratio_threshold
        )
        reasons.append(f"SteadyState: {ss_reason}")
        if steady_state_detected:
            if not current_decision_is_cusum: # If previous rules decided ADEPT
                reasons.append("Steady state detected -> Override to CUSUM")
                current_decision_is_cusum = True

        # 4. Low SNR (favors ADEPT, can override CUSUM from duration/steady state)
        reasons.append(f"SNR: {snr:.2f} (Thresh: {snr_threshold})")
        if snr < snr_threshold:
            if current_decision_is_cusum: # If previous rules decided CUSUM
                reasons.append(f"Low SNR -> Override to ADEPT")
                current_decision_is_cusum = False

        # 5. Absolute override for very short events (must be ADEPT)
        if duration_ms < min_cusum_duration_ms:
            if current_decision_is_cusum: # If it was somehow still CUSUM
                reasons.append(f"Very short ({duration_ms:.3f}ms < {min_cusum_duration_ms}ms) -> Force ADEPT")
            elif not reasons[-1].startswith("Very short"): # Add specific reason if not already made ADEPT by this rule
                 reasons.append(f"Very short ({duration_ms:.3f}ms < {min_cusum_duration_ms}ms) confirms ADEPT")
            current_decision_is_cusum = False
        
        # --- Assign to appropriate algorithm list ---
        event_info = {
            'event_idx': i,
            'start': start,
            'end': end,
            'reasons': "; ".join(reasons)
        }
        if current_decision_is_cusum:
            cusum_events.append(event_info)
        else:
            adept_events.append(event_info)

    return adept_events, cusum_events