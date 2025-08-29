import numpy as np
from scipy.optimize import curve_fit

def gaussian(x, A, mu, sigma):
    """ Gaussian function with amplitude A, mean mu, and standard deviation sigma."""
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def stdv_gaus(data, window = 20):
    '''Calculate the standard deviation and mean of the noise peak by fitting a Gaussian function'''
    if data.ndim == 2:
        currents = data[:,1]
    elif data.ndim == 1:
        currents = data
    else:
        raise ValueError("Input data must be 1D or 2D numpy array.")

    if len(currents) == 0:
        print("Warning: Empty data array provided to stdv_gaus. Returning defaults.")
        return 0.01, 0.0, 1.0, 0.0, 0.1 # Return some defaults

    # 1. Create histogram
    counts, bin_edges = np.histogram(currents, bins='auto') # Use automatic binning
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    if len(counts) == 0:
        print("Warning: Histogram generation failed in stdv_gaus. Returning defaults.")
        # Estimate from overall data as fallback
        mu_est = np.mean(currents)
        sigma_est = np.std(currents)
        if sigma_est == 0: sigma_est = 0.01 # Avoid zero std
        return sigma_est, mu_est, 1.0, mu_est - window*sigma_est, mu_est + window*sigma_est

    # 2. Identify the largest peak
    peak_idx = np.argmax(counts)
    peak_center = bin_centers[peak_idx]
    peak_count = counts[peak_idx]

    # 3. Define a window around the peak to fit (use provided window as multiplier of estimated std)
    # Estimate std dev directly first for window setting
    initial_sigma_est = np.std(currents)
    if initial_sigma_est == 0: initial_sigma_est = np.std(currents[currents != peak_center]) # Try excluding peak
    if initial_sigma_est == 0 or not np.isfinite(initial_sigma_est): initial_sigma_est = 0.1 # Absolute fallback

    fit_left = peak_center - window * initial_sigma_est
    fit_right = peak_center + window * initial_sigma_est

    # Ensure window is within data range
    min_current = np.min(currents)
    max_current = np.max(currents)
    fit_left = max(fit_left, min_current)
    fit_right = min(fit_right, max_current)

    fit_mask = (bin_centers >= fit_left) & (bin_centers <= fit_right)
    x_fit = bin_centers[fit_mask]
    y_fit = counts[fit_mask]

    if len(x_fit) < 3:
        print(f"Warning: Not enough bins ({len(x_fit)}) in the fit range [{fit_left:.2f}, {fit_right:.2f}] for Gaussian fit. Using overall statistics.")
        mu_fit = np.mean(currents)
        sigma_fit = np.std(currents)
        if sigma_fit == 0: sigma_fit = 0.01
        A_fit = peak_count
        return sigma_fit, mu_fit, A_fit, fit_left, fit_right

    # 4. Fit a Gaussian
    # Use strong initial guesses
    p0 = [peak_count, peak_center, initial_sigma_est]
    try:
        popt, pcov = curve_fit(gaussian, x_fit, y_fit, p0=p0, maxfev=5000)
        A_fit, mu_fit, sigma_fit = popt
        # Ensure sigma is positive
        sigma_fit = abs(sigma_fit)
        if sigma_fit == 0: sigma_fit = initial_sigma_est # Fallback if fit yields zero sigma
        return abs(sigma_fit), mu_fit, abs(A_fit), fit_left, fit_right

    except RuntimeError:
        print("Gaussian fit did not converge. Using overall statistics.")
        mu_fit = np.mean(currents)
        sigma_fit = np.std(currents)
        if sigma_fit == 0: sigma_fit = 0.01
        A_fit = peak_count
        return sigma_fit, mu_fit, A_fit, fit_left, fit_right
    except Exception as e:
        print(f"Error during Gaussian fit: {e}. Using overall statistics.")
        mu_fit = np.mean(currents)
        sigma_fit = np.std(currents)
        if sigma_fit == 0: sigma_fit = 0.01
        A_fit = peak_count
        return sigma_fit, mu_fit, A_fit, fit_left, fit_right

def threshold (data_normalised, threshold_multiplier = 6, SO_multiplier = 1.3, window = 20):
    '''Calculate the threshold for the noise peak of normalized data'''
    # Fit a gaussian to the noise peak to extract parameters
    # For normalized data, the mean (mu_fit) should be near 0
    sigma_fit, mu_fit, A_fit, fit_left, fit_right = stdv_gaus(data_normalised[:,1], window)

    # Thresholds are calculated relative to the fitted mean (should be close to 0)
    threshold_value = mu_fit + threshold_multiplier * sigma_fit # Threshold for event start
    SO_value = mu_fit + SO_multiplier * sigma_fit # Threshold for signal overlap (event end)

    print(f"Noise characteristics: Mean={mu_fit:.4f}, StdDev={sigma_fit:.4f}")
    print(f"Calculated Thresholds: Event Start > {threshold_value:.4f}, Event End < {SO_value:.4f}")
    return threshold_value, SO_value, sigma_fit, mu_fit, A_fit, fit_left, fit_right

def detect_events(data: np.ndarray, threshold: float):
    """Detect events in a NORMALISED trace using a simple thresholding method."""
    # Assumes baseline is 0 after normalisation. Events are positive deviations.
    if data.ndim == 2:
        trace = data[:, 1]
    elif data.ndim == 1:
        trace = data
    else:
         raise ValueError("Input data must be 1D or 2D numpy array.")

    # Detect upward deviations: an event occurs when trace > threshold
    event_mask = trace > threshold

    # Return all indices above threshold (simpler, boundaries determined later)
    event_indices = np.where(event_mask)[0]

    print(f"Found {len(event_indices)} data points exceeding threshold {threshold:.3f}")
    return event_indices


def determine_event_boundaries_SO(data: np.ndarray, event_indices: np.ndarray, SO_value: float):
    """
    Determine the start and end indices of events using the Signal Overlap (SO) value.
    Groups contiguous indices above threshold and expands boundaries until signal drops below SO_value.
    """
    if event_indices.size == 0:
        return []

    if data.ndim == 2:
        trace = data[:, 1]
    elif data.ndim == 1:
        trace = data
    else:
        raise ValueError("Input data must be 1D or 2D numpy array.")

    event_boundaries = []
    visited_indices = np.zeros(len(trace), dtype=bool)

    for idx in event_indices:
        if visited_indices[idx]:
            continue

        # Find contiguous block of event indices starting from idx
        current_block = []
        q = [idx]
        visited_indices[idx] = True
        min_idx_in_block = idx
        max_idx_in_block = idx

        while q:
            curr = q.pop(0)
            current_block.append(curr)
            min_idx_in_block = min(min_idx_in_block, curr)
            max_idx_in_block = max(max_idx_in_block, curr)

            # Check neighbors 
            for neighbor_offset in [-1, 1]:
                neighbor_idx = curr + neighbor_offset
                if 0 <= neighbor_idx < len(trace) and trace[neighbor_idx] > SO_value and not visited_indices[neighbor_idx]:
                     # Only group if neighbor is also likely part of an event (above SO)
                    if neighbor_idx in event_indices: 
                        visited_indices[neighbor_idx] = True
                        q.append(neighbor_idx)


        # Find the highest peak within this block
        if not current_block: # Should not happen if idx was in event_indices
             continue
        peak_idx_local = np.argmax(trace[current_block])
        peak_idx = current_block[peak_idx_local]

        # Expand boundaries from the peak using SO_value
        # Find start (searching left from peak)
        start_idx = peak_idx
        while start_idx > 0 and trace[start_idx - 1] > SO_value:
            start_idx -= 1

        # Find end (searching right from peak)
        end_idx = peak_idx
        while end_idx < len(trace) - 1 and trace[end_idx + 1] > SO_value:
            end_idx += 1

        # Mark all indices within the final boundaries as visited
        # This prevents overlaps and redundant processing
        visited_indices[start_idx:end_idx+1] = True

        event_boundaries.append((start_idx, end_idx))

    # Sort events by start time
    event_boundaries.sort(key=lambda x: x[0])

    # Filter out potentially overlapping events (conservative approach)
    if not event_boundaries:
        return []

    final_events = [event_boundaries[0]]
    for i in range(1, len(event_boundaries)):
        current_start, current_end = event_boundaries[i]
        last_start, last_end = final_events[-1]

        # If current event starts after the last one ends, add it
        if current_start > last_end:
            final_events.append((current_start, current_end))

    print(f"Refined to {len(final_events)} distinct events using SO boundaries.")
    return final_events


def determine_event_boundaries_FWHM(data: np.ndarray, event_indices: np.ndarray, fraction: float = 2.0):
    """Determine the start and end indices of events using the Fraction of Width at Half Maximum (FWHM) method."""

    if event_indices.size == 0:
        return [], []

    if data.ndim == 2:
        trace = data[:, 1]
    elif data.ndim == 1:
        trace = data
    else:
        raise ValueError("Input data must be 1D or 2D numpy array.")

    event_boundaries = []
    fwhm_points_plot = []
    visited_indices = np.zeros(len(trace), dtype=bool)
    threshold = np.min(trace[event_indices]) # Use the detection threshold as minimum height

    for idx in event_indices:
        if visited_indices[idx]:
            continue

        # Find contiguous block of event indices (above threshold)
        current_block = []
        q = [idx]
        visited_indices[idx] = True

        while q:
            curr = q.pop(0)
            current_block.append(curr)

            for neighbor_offset in [-1, 1]:
                neighbor_idx = curr + neighbor_offset
                if 0 <= neighbor_idx < len(trace) and trace[neighbor_idx] > threshold and not visited_indices[neighbor_idx]:
                     # Check if neighbor is above threshold
                     if neighbor_idx in event_indices: # Check if it was detected initially
                        visited_indices[neighbor_idx] = True
                        q.append(neighbor_idx)

        if not current_block: continue

        # Find the highest peak in this continuous event
        peak_idx_local = np.argmax(trace[current_block])
        peak_idx = current_block[peak_idx_local]
        peak_value = trace[peak_idx] # Peak height relative to 0 baseline

        # Calculate fractional maximum relative to baseline (0)
        fractional_max = peak_value / fraction

        # Find left intersection (searching left from peak)
        left_idx = peak_idx
        while left_idx > 0 and trace[left_idx - 1] > fractional_max:
            left_idx -= 1

        # Find right intersection (searching right from peak)
        right_idx = peak_idx
        while right_idx < len(trace) - 1 and trace[right_idx + 1] > fractional_max:
            right_idx += 1

        # Add boundaries and points for plotting
        event_boundaries.append((left_idx, right_idx))
        fwhm_points_plot.append((fractional_max, left_idx))
        fwhm_points_plot.append((fractional_max, right_idx))

        # Mark indices within this FWHM boundary as visited (less aggressive than SO)
        visited_indices[left_idx:right_idx+1] = True


    # Sort events by start time
    event_boundaries.sort(key=lambda x: x[0])

    # Filter out potentially overlapping events (conservative approach)
    if not event_boundaries:
        return [], []

    final_events = [event_boundaries[0]]
    # Also filter fwhm_points 
    final_fwhm_points = []
    if len(event_boundaries) > 0:
        start_f, end_f = event_boundaries[0]
        final_fwhm_points.extend([p for p in fwhm_points_plot if start_f <= p[1] <= end_f])

    for i in range(1, len(event_boundaries)):
        current_start, current_end = event_boundaries[i]
        last_start, last_end = final_events[-1]

        if current_start > last_end:
            final_events.append((current_start, current_end))
            # Add corresponding FWHM points
            final_fwhm_points.extend([p for p in fwhm_points_plot if current_start <= p[1] <= current_end])

    print(f"Refined to {len(final_events)} distinct events using FWHM boundaries.")
    return final_events, final_fwhm_points
