import numpy as np
from scipy.optimize import curve_fit
import warnings

# Heaviside step function
def heaviside(x):
    """Heaviside step function"""
    return np.where(x >= 0, 1.0, 0.0)

# General multistate ADEPT model function
def adept_multistate_model(t, *params):
    """
    General multistate ADEPT model, adjusted for inverted traces where events are current drops.
    Assumes amplitude parameters (a_j) represent the positive magnitude of the current drop.
    
    Parameters:
        t (np.ndarray): Time vector.
        params: List of parameters in the format [i_0, a_1, mu_1, tau_1, a_2, mu_2, tau_2, ...].
                i_0 is the baseline current (typically high for inverted traces).
                For each state j: a_j is the positive amplitude (magnitude) of the drop, 
                                 mu_j is time delay, tau_j is time constant.
    
    Returns:
        np.ndarray: Modeled current values.
    """
    i_0 = params[0]  # Baseline current
    num_states = (len(params) - 1) // 3
    
    # Start with baseline current
    i_t = i_0 + np.zeros_like(t)
    
    # Subtract contribution from each state for inverted traces
    for j in range(num_states):
        idx = 1 + j * 3
        # a_j is assumed to be the positive magnitude of the drop
        a_j, mu_j, tau_j = params[idx:idx+3] 
        # Ensure tau_j is positive to prevent numerical issues
        tau_j = abs(tau_j)
        
        # Protect against numerical issues
        t_rel = np.maximum(t - mu_j, 0)  # Time since transition (clipped at 0)
        
        # Calculate state contribution (magnitude of change)
        with np.errstate(over='ignore', under='ignore'):
            # This term represents the magnitude of the change from baseline due to state j
            state_magnitude_change = a_j * (1 - np.exp(-(t_rel / tau_j))) * heaviside(t - mu_j)
            
        # Replace any NaN or Inf values with 0
        state_magnitude_change = np.nan_to_num(state_magnitude_change, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Subtract the magnitude change from the baseline for inverted traces
        i_t -= state_magnitude_change 
    
    return i_t

# Simplified 2-state ADEPT model
def adept_2state_model(t, i_0, a, mu_1, mu_2, tau_rise, tau_fall, beta_rise=1.0, beta_fall=1.0):
    """
    Improved 2-state ADEPT model with separate time constants for rise and fall phases
    and shape parameters for non-ideal behavior.
    
    Parameters:
        t (np.ndarray): Time vector.
        i_0 (float): Baseline current.
        a (float): Amplitude of the state.
        mu_1 (float): First time shift parameter.
        mu_2 (float): Second time shift parameter.
        tau_rise (float): Time constant for entry phase.
        tau_fall (float): Time constant for exit phase.
        beta_rise (float): Shape parameter for entry phase (default=1.0 for standard exponential).
        beta_fall (float): Shape parameter for exit phase (default=1.0 for standard exponential).
    
    Returns:
        np.ndarray: Modeled current values.
    """
    # Ensure tau values are positive
    tau_rise = abs(tau_rise)
    tau_fall = abs(tau_fall)
    
    # Protect against numerical issues
    t1 = np.maximum(t - mu_1, 0)  # Time since first transition (clipped at 0)
    t2 = np.maximum(t - mu_2, 0)  # Time since second transition (clipped at 0)
    
    # First term: entry into the pore with shape parameter
    term1 = (1 - np.exp(-((t1 / tau_rise)**beta_rise))) * heaviside(t - mu_1)

    # Second term: exit from the pore with shape parameter
    term2 = (np.exp(-((t2 / tau_fall)**beta_fall)) - 1) * heaviside(t - mu_2)
    
    # Compute the result with checks for numerical stability
    result = i_0 + a * (term1 + term2)
    
    # Ensure no NaN or infinity values
    result = np.nan_to_num(result, nan=i_0)
    
    return result


def adept_analyze_event(data, event_idx, start, end, num_states=1, use_2state_model=False, initial_params=None, system_tau=None, debug=False):
    """
    Analyze a short event using the ADEPT approach.
    Models the event as an exponentially rising and falling pulse,
    which is appropriate for fast transitions that don't reach steady state.
    
    Parameters:
        data (np.ndarray): The nanopore data, where data[:,1] contains the current values.
        event_idx (int): Index of the event for tracking purposes.
        start (int): Start index of the event.
        end (int): End index of the event.
        num_states (int, optional): Number of states to fit in the model. Default is 1.
        use_2state_model (bool, optional): Whether to use the simplified 2-state model. Default is False.
        initial_params (list, optional): Initial parameter guesses for fitting. 
                                        For multistate: [i_0, a_1, mu_1, tau_1, a_2, mu_2, tau_2, ...]
                                        For 2-state: [i_0, a, mu_1, mu_2, tau_rise, tau_fall, beta_rise, beta_fall]
        system_tau (float, optional): System's characteristic relaxation time. If not provided, estimated from data.
        debug (bool, optional): Whether to print debugging information. Default is False.
        
    Returns:
        dict: Event parameters including fitted model parameters.
    """

    if num_states == 1:
        use_2state_model = True

    data_signal = data[:, 1]
    event_signal = data_signal[start:end]
    event_length = len(event_signal)

    # Calculate time values
    time_vec = data[:, 0]
    event_time = time_vec[start:end]
    event_start_time = time_vec[start]
    event_end_time = time_vec[end-1]  # Use end-1 to get the last valid index
    event_duration = event_end_time - event_start_time
    
    # Convert to relative time for fitting (starting from 0)
    rel_time = event_time - event_start_time

    # For normalized data, the baseline is approximately 0
    baseline_level = 0.0
    
    # Find peak amplitude and its location
    peak_idx = np.argmax(np.abs(event_signal - baseline_level))
    peak_current = event_signal[peak_idx]
    peak_time = time_vec[start + peak_idx]
    normalized_position = peak_idx / event_length
    
    # Calculate event area relative to baseline
    event_area = np.sum(event_signal - baseline_level)
    
    # Check for asymmetry in rise and fall times
    rise_time = peak_idx
    fall_time = event_length - peak_idx
    asymmetry = abs(rise_time - fall_time) / event_length
    
    # Estimate rise and fall time constants separately
    tau_rise = None
    tau_fall = None
    
    # Estimate rise time constant (time to reach 63.2% of peak value)
    if peak_current > baseline_level:  # upward peak
        rise_curve = event_signal[:peak_idx+1] - baseline_level
        target_value = 0.632 * (peak_current - baseline_level)
        rise_indices = np.where(rise_curve >= target_value)[0]
        
        # Look at falling phase - time to decay to 36.8% of peak
        fall_curve = event_signal[peak_idx:] - baseline_level
        fall_target = 0.368 * (peak_current - baseline_level)
        fall_indices = np.where(fall_curve <= fall_target)[0]
        
    else:  # downward peak
        rise_curve = baseline_level - event_signal[:peak_idx+1]
        target_value = 0.632 * (baseline_level - peak_current)
        rise_indices = np.where(rise_curve >= target_value)[0]
        
        # For falling phase of downward peak
        fall_curve = baseline_level - event_signal[peak_idx:]
        fall_target = 0.368 * (baseline_level - peak_current)
        fall_indices = np.where(fall_curve <= fall_target)[0]
    
    # Calculate tau_rise (entry time constant)
    if len(rise_indices) > 0:
        tau_rise = rel_time[rise_indices[0]]
    else:
        # Fallback estimate based on rise time portion
        tau_rise = rel_time[peak_idx] / 3.0
    
    # Calculate tau_fall (exit time constant)
    if len(fall_indices) > 0 and fall_indices[0] < len(event_signal) - peak_idx:
        # Convert to absolute index in the event time array
        fall_idx = peak_idx + fall_indices[0]
        tau_fall = rel_time[fall_idx] - rel_time[peak_idx]
    else:
        # Fallback estimate based on fall time portion
        tau_fall = (rel_time[-1] - rel_time[peak_idx]) / 3.0
    
    # Check for extreme values or estimation failures
    if tau_rise <= 0 or not np.isfinite(tau_rise):
        tau_rise = event_duration / 10.0
    
    if tau_fall <= 0 or not np.isfinite(tau_fall):
        tau_fall = event_duration / 10.0
    
    # Shape parameter estimation - look at curve shape to determine non-ideality
    beta_rise = 1.0
    beta_fall = 1.0
    
    try:
        # For rise phase
        if peak_idx > 3:  # Need enough points for shape estimation
            # Compare actual rise to ideal exponential with estimated tau
            t_rise = rel_time[:peak_idx+1]
            ideal_rise = (1 - np.exp(-(t_rise / tau_rise)))
            actual_rise = (event_signal[:peak_idx+1] - baseline_level) / (peak_current - baseline_level)
            
            # Calculate residuals between actual and ideal
            rise_residuals = actual_rise - ideal_rise
            
            # If residuals are mostly positive, beta_rise < 1 (slower rise)
            # If residuals are mostly negative, beta_rise > 1 (faster rise)
            if np.mean(rise_residuals) > 0:
                beta_rise = 0.7  # Slower rise than exponential
            else:
                beta_rise = 1.3  # Faster rise than exponential
        
        # For fall phase
        if fall_time > 3:  # Need enough points for shape estimation
            # Compare actual fall to ideal exponential with estimated tau
            t_fall = rel_time[peak_idx:] - rel_time[peak_idx]
            ideal_fall = np.exp(-(t_fall / tau_fall))
            
            # Normalize actual fall curve
            if peak_current != baseline_level:
                actual_fall = (event_signal[peak_idx:] - baseline_level) / (peak_current - baseline_level)
                
                # Calculate residuals
                fall_residuals = actual_fall - ideal_fall
                
                # Determine beta_fall from residuals
                if np.mean(fall_residuals) > 0:
                    beta_fall = 0.7  # Slower decay
                else:
                    beta_fall = 1.3  # Faster decay
    except Exception as e:
        if debug:
            print(f"Shape parameter estimation failed: {e}")
        # Default values if shape parameter estimation fails
        beta_rise = 1.0
        beta_fall = 1.0
    
    # Ensure shape parameters are within reasonable bounds
    beta_rise = min(max(beta_rise, 0.3), 3.0)
    beta_fall = min(max(beta_fall, 0.3), 3.0)
    
    if debug:
        print(f"Estimated parameters: tau_rise={tau_rise:.2e}, tau_fall={tau_fall:.2e}, beta_rise={beta_rise:.2f}, beta_fall={beta_fall:.2f}")
    
    # Setup for model fitting
    # For 2-state model
    if use_2state_model:
        if initial_params is None:
            # Estimate parameters for 2-state model
            a_est = peak_current - baseline_level
            mu_1_est = 0.0  # Start of the event
            mu_2_est = rel_time[peak_idx]  # Time of peak as estimate for second transition
            
            # Use our estimated parameters
            initial_params = [baseline_level, a_est, mu_1_est, mu_2_est, tau_rise, tau_fall, beta_rise, beta_fall]
        
        model_func = adept_2state_model
    
    # For general multistate model
    else:
        if initial_params is None:
            # Start with baseline
            initial_params = [baseline_level]
            
            # Identifying potential state transitions
            # Look for significant changes in the derivative of the signal
            event_diff = np.diff(event_signal)
            threshold = np.std(event_diff) * 2
            potential_transitions = np.where(np.abs(event_diff) > threshold)[0]
            
            # Group nearby transitions
            if len(potential_transitions) > 0:
                grouped_transitions = [[potential_transitions[0]]]
                for i in range(1, len(potential_transitions)):
                    if potential_transitions[i] - potential_transitions[i-1] > tau_rise * 5:
                        grouped_transitions.append([potential_transitions[i]])
                    else:
                        grouped_transitions[-1].append(potential_transitions[i])
                
                # Take the average position for each group
                transition_points = [int(np.mean(group)) for group in grouped_transitions]
                
                # Limit to specified number of states
                transition_points = transition_points[:num_states]
            else:
                # Fallback: evenly spaced transitions
                step = event_length // (num_states + 1)
                transition_points = [step * (i + 1) for i in range(num_states)]
            
            # For each detected state, add parameters: amplitude, mu, tau
            for trans_idx in transition_points:
                a_est = event_signal[trans_idx] - baseline_level
                mu_est = rel_time[trans_idx]
                tau_est = tau_rise  # Use rise time as default time constant
                
                initial_params.extend([a_est, mu_est, tau_est])
        
        # Create model function with appropriate number of states
        model_func = adept_multistate_model
    
    # Perform curve fitting with bounds
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if use_2state_model:
                # Set bounds for the 2-state model with separate time constants and shape parameters
                bounds = ([0, -np.inf, 0, 0, 1e-6, 1e-6, 0.3, 0.3], 
                          [np.inf, np.inf, rel_time[-1], rel_time[-1], np.inf, np.inf, 3.0, 3.0])

                popt, pcov = curve_fit(model_func, rel_time, event_signal, p0=initial_params, 
                                       bounds=bounds, maxfev=10000)
                
                # Extract fitted parameters
                fitted_params = {
                    'baseline_current': popt[0],
                    'amplitude': popt[1],
                    'mu_1': popt[2],
                    'mu_2': popt[3],
                    'tau_rise': popt[4],
                    'tau_fall': popt[5],
                    'beta_rise': popt[6],
                    'beta_fall': popt[7],
                    'model': '2-state'
                }
                
                if debug:
                    print("2-state fit successful")
                    print(f"Fitted parameters: {popt}")
                
            else:
                # More complex bounds for multistate model
                num_params = len(initial_params)
                lower_bounds = [0]  # baseline >= 0
                upper_bounds = [np.inf]  # baseline < infinity
                
                # For each state: amplitude can be any value, mu must be within event time, tau must be positive
                for i in range(1, num_params, 3):
                    lower_bounds.extend([-np.inf, 0, 1e-6])
                    upper_bounds.extend([np.inf, rel_time[-1], np.inf])
                
                popt, pcov = curve_fit(model_func, rel_time, event_signal, p0=initial_params,
                                       bounds=(lower_bounds, upper_bounds), maxfev=10000)
                
                # Extract fitted parameters
                fitted_params = {'baseline_current': popt[0], 'model': f'{num_states}-state'}
                for j in range(num_states):
                    idx = 1 + j * 3
                    fitted_params[f'amplitude_{j+1}'] = popt[idx]
                    fitted_params[f'mu_{j+1}'] = popt[idx+1]
                    fitted_params[f'tau_{j+1}'] = popt[idx+2]
                
                if debug:
                    print(f"{num_states}-state fit successful")
        
        # Calculate fitted values
        if use_2state_model:
            fitted_values = adept_2state_model(rel_time, *popt)
        else:
            fitted_values = adept_multistate_model(rel_time, *popt)
        
        # Calculate goodness of fit metrics
        residuals = event_signal - fitted_values
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((event_signal - np.mean(event_signal))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Calculate reduced chi-squared
        n = len(event_signal)
        p = len(popt)

        if (n-p) > 0: 
            reduced_chi_squared = ss_res / (n - p)
        
        # Add goodness of fit metrics to fitted parameters
        fitted_params['r_squared'] = r_squared
        fitted_params['reduced_chi_squared'] = reduced_chi_squared
        
        # For visualization, store the fitted values
        fitted_params['fitted_values'] = fitted_values
        
    except Exception as e:
        # If fitting fails, fallback to basic analysis and try simpler model
        if debug:
            print(f"Fitting failed: {str(e)}")
            
        if not use_2state_model:
            if debug:
                print("Trying 2-state model as fallback...")
            try:
                # Try with 2-state model instead
                a_est = peak_current - baseline_level
                mu_1_est = 0.0
                mu_2_est = rel_time[peak_idx]
                simple_params = [baseline_level, a_est, mu_1_est, mu_2_est, tau_rise, tau_fall, 1.0, 1.0]
                
                bounds = ([0, -np.inf, 0, 0, 1e-6, 1e-6, 0.3, 0.3], 
                          [np.inf, np.inf, rel_time[-1], rel_time[-1], np.inf, np.inf, 3.0, 3.0])

                popt, pcov = curve_fit(adept_2state_model, rel_time, event_signal, p0=simple_params, 
                                       bounds=bounds, maxfev=5000)
                
                fitted_params = {
                    'baseline_current': popt[0],
                    'amplitude': popt[1],
                    'mu_1': popt[2],
                    'mu_2': popt[3],
                    'tau_rise': popt[4],
                    'tau_fall': popt[5],
                    'beta_rise': popt[6],
                    'beta_fall': popt[7],
                    'model': '2-state-fallback'
                }
                
                fitted_values = adept_2state_model(rel_time, *popt)
                
                # Calculate goodness of fit metrics
                residuals = event_signal - fitted_values
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((event_signal - np.mean(event_signal))**2)
                r_squared = 1 - (ss_res / ss_tot)
                
                fitted_params['r_squared'] = r_squared
                fitted_params['fitted_values'] = fitted_values
                
                if debug:
                    print("Fallback 2-state model succeeded")
            except Exception as fallback_error:
                if debug:
                    print(f"Fallback also failed: {str(fallback_error)}")
                fitted_params = {
                    'baseline_current': baseline_level,
                    'model': 'fitting_failed',
                    'fitting_error': str(e) + " | Fallback error: " + str(fallback_error)
                }
        else:
            fitted_params = {
                'baseline_current': baseline_level,
                'model': 'fitting_failed',
                'fitting_error': str(e)
            }
    
    # Combine original event parameters with fitted parameters
    event_dict = {
        'event_idx': event_idx,
        'event_start_idx': start,
        'event_end_idx': end,
        'analysis_method': 'ADEPT',
        'event_type': 'pulse',
        'dwell_time': event_length,
        'dwell_time_ms': event_duration * 1000,  # Convert to milliseconds
        'peak_current': peak_current,
        'carrier_current': peak_current, # For compatibility with the results display
        'baseline_current': baseline_level,
        'normalized_position': normalized_position,
        'peak_time': peak_time,
        'event_area': event_area,
        'asymmetry': asymmetry,
        'fitted_params': fitted_params
    }
    
    return event_dict