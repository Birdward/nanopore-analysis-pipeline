import numpy as np
from Data_Load_Preprocessing import *
from Event_Detection import *
from Event_classification import *
from ADEPT import *
from CUSUM import *
from Visualisation_Output import *
import matplotlib.pyplot as plt

def main():
    """Main function to execute the nanopore analysis workflow."""

    # --- Configuration ---
    # Input Data
    file_path = '.abf'
    data_load_method = 'abf' # 'abf' or 'numpy'
    numpy_file_path = '.npy' # Use pre-normalized data if available

    # Processing Parameters - Need to optimise for each individual trace
    invert_signal = True # Set True if signal is inverted relative to expectation (False)
    do_filter = True
    cutoff_frequency = 10000 # Hz (adjust based on expected signal vs noise)
    do_downsample = True
    desired_sample_time_us = 10 # microseconds (effective sampling rate = 1/desired_sample_time_us MHz)

    # Baseline & Normalization (less critical if loading pre-normalized data)
    baseline_method = 'als' # 'als', 'sg', 'moving_average'
    baseline_params = {'lam': 1e8, 'p': 0.01, 'niter': 15, 'downsample_baseline': True, 'baseline_sample_time_us' : 30, 'manual_offset' : 0} # ALS example
    # baseline_params = {'window_length': 1001, 'polyorder': 2, 'downsample_baseline': False, 'baseline_sample_time_us' : 30, 'manual_offset' : 0} # SG example
    # baseline_params = {'window_size': 10001, 'downsample_baseline': False, 'baseline_sample_time_us' : 30, 'manual_offset' : 0} # Moving average example

    # Event Detection
    threshold_multiplier = 6 # Times noise std dev for event detection (6)
    SO_multiplier = 1.5   # Times noise std dev for event end (Signal Overlap) (1.5)
    noise_window = 20     # Multiplier for Gaussian fit window (times initial std dev)
    boundary_method = 'SO' # 'SO' or 'FWHM'

    # Event Classification
    short_event_duration_ms = 0.5 # Events shorter than this likely ADEPT (typically 0.5 ms)
    snr_threshold = 4.0           # SNR threshold (currently secondary classifier) (4.0)
    default_classifier = "ADEPT"  # If ambiguous, classify as this (Was "ADEPT")

    # Analysis & Output
    output_csv_file = 'nanopore_analysis_results_test.csv'
    run_interactive_plotter = False 
    plot_initial_steps = False # Plot intermediate steps like histogram, threshold trace? (False)
    Debug = False 
    time_range_plot = (0, 5) # Optional: Limit some plots to a time range (seconds), None for full trace

    # --- Data Loading ---
    print("--- Loading Data ---")
    if data_load_method == 'abf':
        try:
            data_raw, sample_rate = import_ABF_trace(file_path)  
            print(f"Loaded ABF: {len(data_raw)} points, Sample Rate: {sample_rate} Hz")
            if invert_signal:
                print("Inverting signal...")
                data_raw = invert_trace(data_raw)
        except FileNotFoundError:
            print(f"Error: ABF file not found at {file_path}")
            return
        except Exception as e:
            print(f"Error loading ABF file: {e}")
            return
    elif data_load_method == 'numpy':
        try:
            data_normalised = np.load(numpy_file_path)
            print(f"Loaded Numpy array: {data_normalised.shape}")
            if data_normalised.ndim != 2 or data_normalised.shape[1] != 2:
                 raise ValueError("Numpy array must be 2D with shape (n_points, 2) [time, signal]")
            # Estimate sample rate from numpy data if possible
            if len(data_normalised) > 1:
                 time_diff = np.mean(np.diff(data_normalised[:, 0]))
                 if time_diff > 0:
                     sample_rate = 1.0 / time_diff
                     print(f"Estimated sample rate from data: {sample_rate:.2f} Hz")
                 else:
                     sample_rate = 125000 # Assume a default if time is constant or decreasing
                     print(f"Warning: Could not estimate sample rate, assuming {sample_rate} Hz.")
            else:
                 sample_rate = 125000 # Assume a default
                 print(f"Warning: Could not estimate sample rate from short data, assuming {sample_rate} Hz.")
            # Data is already normalized, skip filtering/normalization steps
            data_processed = data_normalised
            baseline = np.zeros_like(data_processed[:,1]) # Baseline is zero
            current_sample_rate = sample_rate # No downsampling applied here

        except FileNotFoundError:
            print(f"Error: Numpy file not found at {numpy_file_path}")
            return
        except Exception as e:
            print(f"Error loading Numpy file: {e}")
            return
    else:
        print("Error: Invalid data_load_method specified.")
        return

    # --- Initial Processing (if starting from raw ABF) ---
    if data_load_method == 'abf':
        print("\n--- Initial Processing ---")
        data_to_process = data_raw.copy()
        current_sample_rate = sample_rate

        # Filtering
        if do_filter:
            print(f"Applying low-pass filter (Cutoff: {cutoff_frequency} Hz)...")
            data_to_process[:, 1] = low_pass_filter(data_to_process[:, 1], cutoff_frequency, current_sample_rate)

        # Downsampling
        if do_downsample:
            print(f"Downsampling to ~{desired_sample_time_us} µs sample time...")
            data_to_process, current_sample_rate = sample_time(data_to_process, desired_sample_time_us, current_sample_rate)
            print(f"New sample rate: {current_sample_rate:.2f} Hz")

        # Baseline Correction & Normalization
        print(f"Calculating baseline ({baseline_method})...")
        try:
            data_processed, baseline = normalise_trace(data_to_process, baseline_method=baseline_method, current_sample_rate = current_sample_rate, **baseline_params)
            print("Normalisation complete.")
        except Exception as e:
            print(f"Error during baseline/normalization: {e}. Proceeding with unnormalized data.")
            data_processed = data_to_process # Fallback
            baseline = np.zeros_like(data_processed[:,1])

    # --- Event Detection ---
    print("\n--- Event Detection ---")
    print("Characterizing noise...")
    noise_params = stdv_gaus(data_processed[:,1], window=noise_window)
    noise_sigma, noise_mu, _, _, _ = noise_params

    if noise_sigma <= 0:
        print("Error: Could not estimate noise standard deviation. Aborting.")
        return

    threshold_value, SO_value, _, _, _, _, _ = threshold(data_processed, threshold_multiplier, SO_multiplier, noise_window)

    print("Detecting points above threshold...")
    event_indices = detect_events(data_processed, threshold_value)

    if len(event_indices) == 0:
        print("No points detected above threshold. No events to analyze.")
        if plot_initial_steps:
            plot_hist(data_processed, threshold_value, SO_value, noise_params)
            plot_trace_threshold(data_processed, threshold_value, SO_value, time_range=time_range_plot)
            plt.show()
        return

    print(f"Determining event boundaries using {boundary_method} method...")
    if boundary_method == 'SO':
        event_boundaries = determine_event_boundaries_SO(data_processed, event_indices, SO_value)
        fwhm_points = None
    elif boundary_method == 'FWHM':
        # Requires fraction parameter, using default 2.0 (Half Max)
        event_boundaries, fwhm_points = determine_event_boundaries_FWHM(data_processed, event_indices, fraction=3.0)
    else:
        print(f"Error: Unknown boundary_method '{boundary_method}'. Using SO.")
        event_boundaries = determine_event_boundaries_SO(data_processed, event_indices, SO_value)
        fwhm_points = None

    if not event_boundaries:
        print("No distinct event boundaries identified.")
        if plot_initial_steps:
            plot_hist(data_processed, threshold_value, SO_value, noise_params)
            plot_trace_threshold(data_processed, threshold_value, SO_value, time_range=time_range_plot)
            plt.show()
        return

    print(f"Identified {len(event_boundaries)} potential events.")

    if plot_initial_steps:
        print("Plotting initial analysis steps...")
        plot_hist(data_processed, threshold_value, SO_value, noise_params)
        plot_trace_threshold(data_processed, threshold_value, SO_value, time_range=time_range_plot)
        plot_event_boundaries(data_processed, event_boundaries[:20], # Plot first 20 events
                                threshold_params=(threshold_value, SO_value, noise_sigma, noise_mu, None, None, None),
                                boundary_method=boundary_method, time_range=time_range_plot)
        plt.show() # Show setup plots before analysis

    # --- Event Classification ---
    print("\n--- Event Classification ---")
    adept_event_info, cusum_event_info = classify_events(
        data_processed, event_boundaries, noise_sigma, noise_mu, current_sample_rate,
        short_event_duration_ms=short_event_duration_ms, snr_threshold=snr_threshold,
        default=default_classifier)
    
    print(f"Classified: {len(adept_event_info)} ADEPT, {len(cusum_event_info)} CUSUM")
    # Print reasons for first few classifications
    for i, info in enumerate(adept_event_info[:2] + cusum_event_info[:2]):
         print(f"  Event {info['event_idx']+1}: {info['reasons']}")

    # --- Event Analysis ---
    print("\n--- Analyzing Events ---")
    all_event_params = []
    processed_count = 0

    # Analyze ADEPT events
    if adept_event_info:
        print(f"Analyzing {len(adept_event_info)} events with ADEPT...")
        for info in adept_event_info:
            processed_count += 1
            print(f" Processing ADEPT Event {processed_count}/{len(adept_event_info)} (Idx: {info['event_idx']+1})")
            event_params = adept_analyze_event(
                data_processed, info['event_idx'], info['start'], info['end'],
                use_2state_model=True, # Defaulting to 2-state model for ADEPT
                debug=Debug # Set True for detailed fit info
            )
            all_event_params.append(event_params)
        print("\nADEPT analysis complete.")

    # Analyze CUSUM events
    if cusum_event_info:
        print(f"Analyzing {len(cusum_event_info)} events with CUSUM...")
        for info in cusum_event_info:
            processed_count += 1
            print(f" Processing CUSUM Event {processed_count}/{len(cusum_event_info)} (Idx: {info['event_idx']+1})")
            # Use noise_mu (should be ~0) as baseline_current for area calc, noise_sigma for analysis
            event_params = cusum_analyze_event(
                data_processed, info['start'], info['end'], noise_mu, noise_sigma
            )
            event_params['event_idx'] = info['event_idx'] # Add original index
            all_event_params.append(event_params)
        print("\nCUSUM analysis complete.")

    # Sort results by original event index
    all_event_params.sort(key=lambda x: x['event_idx'])

    # --- Save Results ---
    print("\n--- Saving Results ---")
    save_event_params_csv(data_processed, all_event_params, output_csv_file)

    # --- Interactive Plotting ---
    if run_interactive_plotter:
        interactive_event_plotter(data_processed, all_event_params)

    print("\n--- Analysis Finished ---")


# --- Execute Main Function ---
if __name__ == "__main__":
    main()