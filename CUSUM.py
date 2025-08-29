import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

def cusum_detector(data, k=0.5, h=3, min_distance=3, adaptive_threshold=True, window_size=50, noise_std=None):
    """
    Enhanced CUSUM algorithm based on OpenNanopore concepts.
    Detects change points in potentially noisy data using adaptive or fixed thresholds.
    """
    n = len(data)
    if n < 2: return []

    # Initialize CUSUM statistics
    s_pos = np.zeros(n)
    s_neg = np.zeros(n)

    # Use provided noise std deviation if available, otherwise estimate
    if noise_std is None:
        # Estimate from the whole segment - might be biased by levels, but fallback
        est_std = np.std(data)
        if est_std < 1e-6: est_std = 1e-6 # Avoid zero std
        noise_std = est_std
        print(f"Warning: noise_std not provided to cusum_detector, estimated as {noise_std:.4f}")

    # Ensure noise_std is positive
    if noise_std <= 0: noise_std = 1e-6

    # Adaptive thresholding setup
    if adaptive_threshold and n > window_size and window_size > 0:
        # Calculate rolling std dev - more robust against level changes than mean
        # Using pandas for efficient rolling calculation if available, else numpy loop
        try:
            import pandas as pd
            rolling_std_pd = pd.Series(data).rolling(window=window_size, center=True, min_periods=max(1, window_size//2)).std()
            # Fill NaN values at edges using backfill/ffill
            rolling_std_pd = rolling_std_pd.bfill().ffill()
            rolling_std = rolling_std_pd.to_numpy()
            # Ensure minimum std dev based on global estimate
            rolling_std = np.maximum(rolling_std, noise_std * 0.5) # Don't let local std drop too low
            # Smooth the rolling std slightly
            rolling_std = gaussian_filter1d(rolling_std, sigma=1)
        except ImportError:
            print("Pandas not found, using numpy for rolling std (slower).")
            adaptive_threshold = False # Fallback to non-adaptive
            rolling_std = np.full(n, noise_std)
    else:
        # Use global noise std if adaptive thresholding is disabled or data too short
        rolling_std = np.full(n, noise_std)

    # --- CUSUM core loop ---
    # k represents the shift to detect in units of std dev (delta/sigma in OpenNanopore)
    # h is the threshold multiplier (related to ARL)
    change_points = []
    last_change_point = -min_distance # Initialize to allow detection near start

    # Initial reference level (use local median for robustness)
    initial_window = data[:min(n, window_size if window_size > 0 else 50)]
    current_reference = np.median(initial_window) if len(initial_window) > 0 else data[0]

    for i in range(1, n):
        local_std = rolling_std[i]
        if local_std <= 0 : local_std = noise_std # Safety check

        # k_norm is the reference value (k * sigma in OpenNanopore notation)
        k_norm = k * local_std

        # Difference from current reference level
        diff = data[i] - current_reference

        # Update CUSUM statistics
        s_pos[i] = max(0, s_pos[i-1] + diff - k_norm)
        s_neg[i] = max(0, s_neg[i-1] - diff - k_norm) # Detect downward shifts (-diff > k_norm)

        # Detection threshold (h * sigma in OpenNanopore notation)
        h_threshold = h * local_std

        detected = False
        if s_pos[i] > h_threshold:
            if i - last_change_point >= min_distance:
                change_points.append(i)
                detected = True
        elif s_neg[i] > h_threshold: # Check negative CUSUM
             if i - last_change_point >= min_distance:
                change_points.append(i)
                detected = True

        if detected:
            last_change_point = i
            # Reset CUSUM stats after detection
            s_pos[i] = 0
            s_neg[i] = 0
            # Update reference level to the median around the detected change point
            ref_win_start = max(0, i - window_size // 4)
            ref_win_end = min(n, i + window_size // 4 + 1)
            if ref_win_end > ref_win_start:
                 current_reference = np.median(data[ref_win_start:ref_win_end])
            else: # Very close to end
                 current_reference = data[i]


    # --- Refinement (Optional) ---
    # The detected point 'i' is when the threshold is crossed. The actual change
    # might have occurred earlier. OpenNanopore uses argmin S(n-1) but that requires
    # tracking S(k), not just G(k). A simpler refinement: look for max gradient near change point.
    refined_points = []
    if change_points:
        refined_points.append(change_points[0]) # Keep first
        for j in range(len(change_points) - 1):
            cp1 = refined_points[-1]
            cp2 = change_points[j+1]
            # Only add if distance constraint met AFTER potential refinement
            if cp2 - cp1 >= min_distance:
                 # --- Simple refinement: find max diff in window before cp2 ---
                 search_start = max(cp1 + 1, cp2 - min_distance) # Look between last refined and current
                 search_end = cp2 + 1
                 if search_end > search_start + 1:
                     local_data = data[search_start:search_end]
                     local_diff = np.abs(np.diff(local_data))
                     if len(local_diff) > 0:
                         refined_idx = search_start + np.argmax(local_diff) + 1 # Point after max diff
                         # Check distance again
                         if refined_idx - cp1 >= min_distance:
                              refined_points.append(refined_idx)
                         else:
                              refined_points.append(cp2) # Use original if refinement violates distance
                     else:
                          refined_points.append(cp2) # Use original if no diff calc possible
                 else:
                      refined_points.append(cp2) # Use original if window too small
            #else: skip cp2 as it's too close to the last refined point

    return sorted(list(set(refined_points))) # Return unique sorted points


def group_levels(data, change_points, min_level_points=5):
    """Groups data into levels based on detected change points."""
    levels = []
    n = len(data)

    # Add start and end points if not already change points
    all_points = sorted(list(set([0] + change_points + [n])))

    # Filter out consecutive duplicates that might arise from adding 0/n
    unique_points = []
    if all_points:
        unique_points.append(all_points[0])
        for i in range(1, len(all_points)):
            if all_points[i] > all_points[i-1]:
                unique_points.append(all_points[i])
    all_points = unique_points

    if len(all_points) < 2: return [] # Not enough points to form a level

    global_std = np.std(data) if n > 0 else 0.1 # Avoid zero std

    for i in range(len(all_points) - 1):
        start_idx = all_points[i]
        end_idx = all_points[i+1]
        duration = end_idx - start_idx

        if duration < min_level_points:
            continue

        segment = data[start_idx:end_idx]

        # Use median for level current (robust to outliers/transients)
        level_current = np.median(segment)
        # Use std dev for noise estimate within the level
        level_std = np.std(segment)
        if level_std < 1e-9: level_std = global_std * 0.1 # Handle flat segments

        # Stability metric (lower is more stable) - std relative to magnitude
        stability = level_std / (abs(level_current) + 1e-9)

        level = {
            'start': start_idx,
            'end': end_idx,
            'points': duration,
            'current': float(level_current),
            'std': float(level_std),
            'mean': float(np.mean(segment)), # Also store mean
            'median': float(level_current),
            'stability': float(stability),
            'dwell_time_ms': None # Placeholder, calculated later if time data exists
        }
        levels.append(level)

    # --- Merge adjacent levels if they are very similar ---
    if len(levels) > 1:
        merged_levels = [levels[0]]
        for i in range(1, len(levels)):
            prev_level = merged_levels[-1]
            curr_level = levels[i]

            # Check similarity based on std devs
            # Merge if difference is less than combined noise estimate
            diff = abs(curr_level['current'] - prev_level['current'])
            merge_threshold = 1.0 * (prev_level['std'] + curr_level['std']) 

            if diff < merge_threshold:
                # Merge curr_level into prev_level
                merged_start = prev_level['start']
                merged_end = curr_level['end']
                merged_segment = data[merged_start:merged_end]
                merged_duration = merged_end - merged_start

                if merged_duration > 0:
                    merged_current = np.median(merged_segment)
                    merged_std = np.std(merged_segment)
                    if merged_std < 1e-9: merged_std = global_std * 0.1
                    merged_stability = merged_std / (abs(merged_current) + 1e-9)

                    # Update the last level in merged_levels
                    merged_levels[-1].update({
                        'end': merged_end,
                        'points': merged_duration,
                        'current': float(merged_current),
                        'std': float(merged_std),
                        'mean': float(np.mean(merged_segment)),
                        'median': float(merged_current),
                        'stability': float(merged_stability)
                    })
                # else: skip potential zero duration merge
            else:
                # No merge, add current level
                merged_levels.append(curr_level)
        levels = merged_levels

    return levels


def identify_carrier_levels(levels, noise_std, min_carrier_fraction=0.05, min_level_separation=2.0):
    """Identifies significant 'carrier' levels based on duration, stability, separation."""
    if not levels: return []

    total_points = sum(level['points'] for level in levels)
    if total_points == 0: return []

    # Min points threshold based on fraction and absolute minimum
    min_threshold_points = max(5, int(min_carrier_fraction * total_points))

    # Filter levels that are long enough
    candidate_levels = [level for level in levels if level['points'] >= min_threshold_points]

    if not candidate_levels:
        # Fallback: return the longest level if no candidates meet fraction threshold
        if levels:
             longest_level = max(levels, key=lambda x: x['points'])
             if longest_level['points'] >= 3: # Ensure at least minimal points
                 return [longest_level]
        return []


    # Sort candidates by duration (longest first) - simpler than complex scoring
    candidate_levels.sort(key=lambda x: x['points'], reverse=True)

    carrier_levels = []
    if candidate_levels:
        # Start with the longest level
        carrier_levels.append(candidate_levels[0])

        # Add other levels if they are distinct enough
        for level in candidate_levels[1:]:
            is_distinct = True
            for carrier in carrier_levels:
                # Separation check based on noise level
                separation_threshold = min_level_separation * max(noise_std, level['std'], carrier['std'])
                if abs(level['current'] - carrier['current']) < separation_threshold:
                    is_distinct = False
                    break

            if is_distinct:
                carrier_levels.append(level)

    # Sort final carriers by current value
    carrier_levels.sort(key=lambda x: x['current'])

    return carrier_levels


def detect_subpeaks(event_data, carrier_level, carrier_levels_all=None, noise_std=0.1, min_peak_height_mult=2.0, min_peak_dist_points=5):
    """Detects sub-peaks relative to a given carrier level."""
    start_idx = carrier_level['start']
    end_idx = carrier_level['end']
    carrier_current = carrier_level['current']
    # Use carrier std if available and reliable, otherwise use global noise std
    level_std = carrier_level.get('std', noise_std)
    if level_std <= 0: level_std = noise_std
    if level_std <= 0: level_std = 0.1 # Absolute fallback

    segment = event_data[start_idx:end_idx]
    segment_len = len(segment)

    if segment_len < min_peak_dist_points + 1: # Need enough points for detection
        return []

    # Data relative to the carrier level median
    relative_segment = segment - carrier_current

    # Find peaks significantly above the carrier level
    # Height is relative to the carrier level (which is 0 in relative_segment)
    height_threshold = min_peak_height_mult * level_std
    # Prominence relative to noise level
    prominence_threshold = level_std * 1.0
    # Min width in points
    min_width = 2

    peak_indices_rel, properties = find_peaks(
        relative_segment,
        height=height_threshold,
        distance=min_peak_dist_points,
        prominence=prominence_threshold,
        width=min_width
    )

    subpeaks = []
    if len(peak_indices_rel) > 0:
        # Get peak properties
        peak_heights = properties['peak_heights']
        prominences = properties['prominences']
        widths = properties['widths']
        left_ips = properties['left_ips'] # Interpolated position
        right_ips = properties['right_ips'] # Interpolated position

        for i, peak_idx_rel in enumerate(peak_indices_rel):
            peak_idx_abs = start_idx + peak_idx_rel
            actual_peak_current = event_data[peak_idx_abs]
            amplitude = actual_peak_current - carrier_current

            # Use width property for dwell time estimate
            # Note: 'widths' from find_peaks is at half-prominence by default
            # Use the interpolated boundaries for start/end
            peak_start_rel = int(np.floor(left_ips[i]))
            peak_end_rel = int(np.ceil(right_ips[i]))
            peak_start_abs = start_idx + peak_start_rel
            peak_end_abs = start_idx + peak_end_rel
            dwell_time_points = peak_end_rel - peak_start_rel + 1

            # Calculate area above carrier
            peak_area = np.sum(segment[peak_start_rel : peak_end_rel+1] - carrier_current)

            # Find carrier index
            carrier_idx = -1
            if carrier_levels_all:
                try:
                    carrier_idx = carrier_levels_all.index(carrier_level)
                except ValueError:
                     # Find closest carrier if exact match fails
                     dists = [abs(c['current'] - carrier_current) for c in carrier_levels_all]
                     if dists: carrier_idx = np.argmin(dists)


            subpeak = {
                'start': peak_start_abs,
                'end': peak_end_abs,
                'peak_idx': peak_idx_abs, # Index of the peak maximum
                'peak_current': float(actual_peak_current),
                'amplitude': float(amplitude), # Relative to carrier
                'dwell_time': dwell_time_points,
                'dwell_time_ms': None, # Placeholder
                'prominence': float(prominences[i]),
                'width': float(widths[i]), # Width at half-prominence
                'peak_area': float(peak_area),
                'carrier_idx': carrier_idx,
                'significance': float(prominences[i] * widths[i] / level_std if level_std > 0 else 0) # Simple significance score
            }
            subpeaks.append(subpeak)

    # Sort subpeaks by position
    subpeaks.sort(key=lambda x: x['peak_idx'])
    return subpeaks


def cusum_analyze_event(data_normalised, event_start_idx, event_end_idx, baseline_current, noise_std):
    """Analyzes a single event using the CUSUM algorithm."""

    if event_start_idx >= event_end_idx:
        print(f"Warning: Skipping CUSUM analysis for event, start >= end ({event_start_idx} >= {event_end_idx}).")
        return {'event_start_idx': event_start_idx, 'event_end_idx': event_end_idx, 'analysis_method': 'CUSUM', 'event_type': 'step_error'}

    event_data = data_normalised[event_start_idx:event_end_idx, 1]
    event_time = data_normalised[event_start_idx:event_end_idx, 0]
    event_length = len(event_data)

    if event_length < 5: # Need minimum points for analysis
         print(f"Warning: Skipping CUSUM analysis for event {event_start_idx}-{event_end_idx}, too short ({event_length} points).")
         return {'event_start_idx': event_start_idx, 'event_end_idx': event_end_idx, 'analysis_method': 'CUSUM', 'event_type': 'step_short'}

    # Adaptive parameter selection (based on OpenNanopore/MOSAIC ideas)
    if event_length < 50:  # Very short - high sensitivity
        k_value = 0.4 # More sensitive to small changes
        h_value = 4.5 # Lower threshold (was 2.5)
        min_dist = 2
        min_level_points = 3
        adaptive_threshold = False # Not enough data for reliable rolling stats
        window_size = 20
    elif event_length < 200: # Medium
        k_value = 0.5 
        h_value = 5.0 # was 3.0
        min_dist = 4
        min_level_points = 5
        adaptive_threshold = True
        window_size = 30
    else:  # Long events - more conservative
        k_value = 0.6 # Less sensitive to noise
        h_value = 5.5 # Higher threshold to avoid spurious breaks (was 3.5)
        min_dist = 5
        min_level_points = max(5, int(event_length * 0.02))
        adaptive_threshold = True
        window_size = min(50, event_length // 4)

    # Detect change points
    change_points = cusum_detector(
        event_data,
        k=k_value,
        h=h_value,
        min_distance=min_dist,
        adaptive_threshold=adaptive_threshold,
        window_size=window_size,
        noise_std=noise_std
    )

    # Group into levels
    levels = group_levels(
        event_data,
        change_points,
        min_level_points=min_level_points
    )

    # Identify carrier levels
    min_carrier_fraction = 0.05 if event_length > 100 else 0.03
    min_level_separation = 2.5 if noise_std < 0.15 else 3.0 # Increase separation for noisier data
    carrier_levels = identify_carrier_levels(
        levels,
        noise_std,
        min_carrier_fraction=min_carrier_fraction,
        min_level_separation=min_level_separation
    )

    # Ensure at least one carrier level if levels were found
    if not carrier_levels and levels:
        # Fallback: choose the longest or most stable level
        longest_level = max(levels, key=lambda x: x['points'])
        # most_stable = min(levels, key=lambda x: x.get('stability', float('inf')))
        carrier_levels = [longest_level]

    # Calculate dwell times in ms for carrier levels
    total_carrier_points = sum(level['points'] for level in carrier_levels)
    dwell_time_s = float(event_time[-1] - event_time[0]) if event_length > 1 else 0.0

    for level in carrier_levels:
        level_duration_s = dwell_time_s * (level['points'] / event_length) if event_length > 0 else 0.0
        level['dwell_time_ms'] = level_duration_s * 1000
        level['dwell_fraction'] = level['points'] / total_carrier_points if total_carrier_points > 0 else 0

    # Detect subpeaks on carrier levels
    all_subpeaks = []
    for i, carrier in enumerate(carrier_levels):
        min_peak_ht = 2.0 if carrier.get('stability', 1.0) > 0.5 else 1.5 # Higher threshold for less stable carriers
        subpeaks = detect_subpeaks(
            event_data,
            carrier,
            carrier_levels_all=carrier_levels,
            noise_std=noise_std,
            min_peak_height_mult=min_peak_ht,
            min_peak_dist_points=max(3, min_dist) # Link subpeak distance to change point distance
        )
        # Add carrier index and calculate dwell time ms
        for sp in subpeaks:
            sp['carrier_idx'] = i
            if sp['end'] < event_length and sp['start'] < sp['end']:
                 sp_duration_s = event_time[sp['end']] - event_time[sp['start']]
                 sp['dwell_time_ms'] = sp_duration_s * 1000
            else:
                 sp['dwell_time_ms'] = (sp['dwell_time'] / (event_length / dwell_time_s)) * 1000 if event_length > 0 and dwell_time_s > 0 else 0 # Estimate based on points

        all_subpeaks.extend(subpeaks)

    # Filter subpeaks by significance if too many
    if len(all_subpeaks) > 10: # Limit number of reported subpeaks
        all_subpeaks.sort(key=lambda x: x.get('significance', 0), reverse=True)
        all_subpeaks = all_subpeaks[:10]
        all_subpeaks.sort(key=lambda x: x['peak_idx']) # Resort by time

    # Calculate overall event metrics
    dwell_time_ms = dwell_time_s * 1000
    event_mean = np.mean(event_data)
    event_std = np.std(event_data)
    event_area = np.sum(baseline_current - event_data) # Area relative to external baseline

    # Weighted average carrier current
    if total_carrier_points > 0:
        carrier_current_mean = sum(c['current'] * c['points'] for c in carrier_levels) / total_carrier_points
    elif levels:
        carrier_current_mean = np.mean([l['current'] for l in levels]) # Avg of all levels if no carriers
    else:
        carrier_current_mean = event_mean # Fallback

    event_params = {
        'event_idx': -1, # Placeholder, filled in main loop
        'event_start_idx': event_start_idx,
        'event_end_idx': event_end_idx,
        'analysis_method': 'CUSUM',
        'event_type': 'step',
        'dwell_time': event_length,
        'dwell_time_ms': dwell_time_ms,
        'carrier_current': float(carrier_current_mean), # Weighted avg carrier
        'event_area': float(event_area),
        'event_max': float(np.max(event_data)),
        'event_min': float(np.min(event_data)),
        'event_mean': float(event_mean),
        'event_median': float(np.median(event_data)),
        'event_std': float(event_std),
        'carrier_levels': carrier_levels,
        'subpeaks': all_subpeaks,
        'change_points': change_points, # Store detected change points
        'baseline_current': float(baseline_current), # Baseline used for area calc
        'noise_std': float(noise_std), # Noise level used for analysis
        'num_levels': len(carrier_levels),
        'num_subpeaks': len(all_subpeaks)
    }

    return event_params