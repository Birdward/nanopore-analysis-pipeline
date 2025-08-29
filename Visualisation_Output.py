import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import csv

from Event_Detection import gaussian
from CUSUM import *


def plot_hist(data_normalised, threshold_val, SO_val, noise_params):
    """Plots histogram of normalised data with Gaussian fit and thresholds."""
    sigma_fit, mu_fit, A_fit, fit_left, fit_right = noise_params
    currents = data_normalised[:, 1]

    counts, bin_edges = np.histogram(currents, bins='auto')
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers, counts, 'x', markersize=4, label='Histogram Bins')
    plt.yscale('log')

    # Plot fitted Gaussian
    if A_fit is not None and sigma_fit is not None and sigma_fit > 0:
        x_fine = np.linspace(min(currents), max(currents), 500)
        fit_curve = gaussian(x_fine, A_fit, mu_fit, sigma_fit)
        plt.plot(x_fine, fit_curve, 'r-', label=f'Gaussian Fit (μ={mu_fit:.2f}, σ={sigma_fit:.2f})')

        # Highlight fit range used
        plt.axvline(fit_left, color='grey', linestyle=':', alpha=0.6, label='Fit Range')
        plt.axvline(fit_right, color='grey', linestyle=':', alpha=0.6)

    # Plot thresholds
    plt.axvline(threshold_val, color='k', linestyle='--', label=f'Threshold ({threshold_val:.2f})')
    plt.axvline(SO_val, color='g', linestyle='--', label=f'Signal Overlap ({SO_val:.2f})')

    plt.xlabel('Normalized Current')
    plt.ylabel('Frequency (log scale)')
    plt.title('Normalized Current Histogram and Noise Fit')
    plt.legend()
    # Adjust xlim for better visibility near baseline
    plt.xlim(mu_fit - 8 * sigma_fit, threshold_val * 1.5 if sigma_fit > 0 else max(currents))
    plt.ylim(bottom=0.5) # Avoid zero in log scale
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def plot_trace_threshold(data, threshold_value, SO_value, time_range=None):
    '''Plot the trace with the threshold lines'''
    if time_range:
        start_t, end_t = time_range
        mask = (data[:, 0] >= start_t) & (data[:, 0] <= end_t)
        data_plot = data[mask]
    else:
        data_plot = data

    if len(data_plot) == 0:
        print("No data to plot in the specified time range.")
        return

    plt.figure(figsize=(15, 5))
    plt.plot(data_plot[:, 0], data_plot[:, 1], label='Normalised Signal', color='blue', linewidth=0.4)
    plt.axhline(y=threshold_value, color='black', linestyle='--', label=f'Threshold ({threshold_value:.2f})')
    plt.axhline(y=SO_value, color='green', linestyle='--', label=f'SO Threshold ({SO_value:.2f})')
    plt.axhline(y=0, color='grey', linestyle=':', alpha=0.7, label='Baseline (0)') # Baseline is 0

    plt.xlabel("Time (s)")
    plt.ylabel("Normalised Current")
    plt.title("Normalised Trace with Detection Thresholds")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if time_range: plt.xlim(time_range)
    plt.tight_layout()

def plot_cusum_analysis(data_normalised, event_dict, ax=None):
    """Plot the event signal with CUSUM analysis results."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))

    start = event_dict.get('event_start_idx', 0)
    end = event_dict.get('event_end_idx', len(data_normalised))
    if start >= end: return ax # Skip if invalid range

    event_time = data_normalised[start:end, 0]
    event_signal = data_normalised[start:end, 1]
    baseline_current = 0.0 # Assumes normalized data

    # Plot original signal
    ax.plot(event_time, event_signal, color='black', linestyle='-', linewidth=0.8, label='Signal', alpha=0.9)

    # Plot baseline
    ax.axhline(y=baseline_current, color='grey', linestyle=':', alpha=0.7, label='Baseline (0)')

    # Plot carrier levels
    carrier_levels = event_dict.get('carrier_levels', [])
    colors = plt.cm.viridis(np.linspace(0, 0.8, max(1, len(carrier_levels)))) # Use a colormap

    for i, level in enumerate(carrier_levels):
        # Indices are relative to event start
        level_start_idx = level.get('start', 0)
        level_end_idx = level.get('end', len(event_time))
        level_current = level.get('current', 0)
        level_std = level.get('std', 0)

        # Ensure indices are within the plotted time range length
        level_start_idx = min(max(level_start_idx, 0), len(event_time) - 1)
        level_end_idx = min(max(level_end_idx, 0), len(event_time))
        if level_end_idx <= level_start_idx: continue # Skip zero/negative duration levels

        level_start_time = event_time[level_start_idx]
        level_end_time = event_time[level_end_idx - 1] # End time of the last point in level

        # Plot the carrier level line segment
        ax.plot([level_start_time, level_end_time], [level_current, level_current],
                color=colors[i], linestyle='-', linewidth=2.5, alpha=0.8,
                label=f'Carrier {i+1}')

        # Add std deviation bands
        if level_std > 0:
            ax.fill_between([level_start_time, level_end_time],
                            [level_current - level_std, level_current - level_std],
                            [level_current + level_std, level_current + level_std],
                            color=colors[i], alpha=0.2, step='post') 

    # Plot subpeaks
    subpeaks = event_dict.get('subpeaks', [])
    for i, peak in enumerate(subpeaks):
        peak_idx_abs = peak.get('peak_idx', -1)
        # Convert absolute index to relative index within the event plot
        peak_idx_rel = peak_idx_abs - start
        carrier_idx = peak.get('carrier_idx', 0)

        if 0 <= peak_idx_rel < len(event_time):
            peak_time = event_time[peak_idx_rel]
            peak_current = peak.get('peak_current', 0)
            color = colors[carrier_idx % len(colors)] if colors.any() else 'red'

            # Mark the peak
            ax.plot(peak_time, peak_current, 'o', color=color, markersize=8, markeredgecolor='black',
                    label='Subpeak' if i == 0 else None) # Label only first subpeak

            # Show peak boundaries if available
            peak_start_abs = peak.get('start', -1)
            peak_end_abs = peak.get('end', -1)
            peak_start_rel = peak_start_abs - start
            peak_end_rel = peak_end_abs - start

            if 0 <= peak_start_rel < len(event_time) and peak_start_rel < peak_end_rel < len(event_time):
                ax.axvspan(event_time[peak_start_rel], event_time[peak_end_rel],
                           alpha=0.2, color=color,
                           label='Subpeak Region' if i == 0 else None)

    # Plot change points
    change_points_rel = event_dict.get('change_points', [])
    for cp_idx_rel in change_points_rel:
        if 0 < cp_idx_rel < len(event_time): # Avoid plotting at very start/end
            ax.axvline(x=event_time[cp_idx_rel], color='k', linestyle=':', alpha=0.6, linewidth=1.0, label='Change Point' if 'Change Point' not in ax.get_legend_handles_labels()[1] else None)

    # Add event information text
    dwell_time_ms = event_dict.get('dwell_time_ms', 0)
    carrier_current_mean = event_dict.get('carrier_current', 0)
    event_area = event_dict.get('event_area', 0)

    info_text = (
        f"Dwell: {dwell_time_ms:.2f} ms\n"
        f"Carriers: {len(carrier_levels)} | Subpeaks: {len(subpeaks)}\n"
        f"Mean Carrier Curr: {carrier_current_mean:.3f}\n"
        # f"Event Area: {event_area:.2f}"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalised Current')
    ax.set_title(f"Event Analysis (CUSUM)") # Title set externally for specific event

    # Clean up legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    return ax


def plot_adept_fit(data_normalised, event_dict, ax=None, debug=False):
    """Plot the original event signal and the ADEPT fit."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))

    start = event_dict.get('event_start_idx', 0)
    end = event_dict.get('event_end_idx', len(data_normalised))
    if start >= end: return ax

    event_time = data_normalised[start:end, 0]
    event_signal = data_normalised[start:end, 1]
    baseline_current = 0.0 # Assumes normalised

    # Plot original signal
    ax.plot(event_time, event_signal, color='black', linestyle='-', linewidth=0.8, label='Signal', alpha=0.9)

    # Plot baseline
    ax.axhline(y=baseline_current, color='grey', linestyle=':', alpha=0.7, label='Baseline (0)')

    # Check for fitting results
    fit_info = event_dict.get('fitted_params', {})
    fitted_values = fit_info.get('fitted_values', None)
    model_type = fit_info.get('model', 'fitting_failed')

    if fitted_values is not None and len(fitted_values) == len(event_time) and model_type != 'fitting_failed':
        ax.plot(event_time, fitted_values, 'r--', linewidth=2, label=f'ADEPT Fit ({model_type})')
        fit_successful = True
    else:
        fit_successful = False
        if debug: print(f"Debug: No valid fitted values for event {event_dict.get('event_idx', -1)+1}")

    # Add fitting parameters text
    r_squared = fit_info.get('r_squared', float('nan'))
    info_text = f"Model: {model_type}\n"
    if fit_successful:
        info_text += f"R²: {r_squared:.3f}\n"
        if model_type == '2-state':
             info_text += (f"τ_rise: {fit_info.get('tau_rise', 0):.2e} s | β_rise: {fit_info.get('beta_rise', 0):.2f}\n"
                          f"τ_fall: {fit_info.get('tau_fall', 0):.2e} s | β_fall: {fit_info.get('beta_fall', 0):.2f}")
        # Add key params for multistate if needed

    dwell_time_ms = event_dict.get('dwell_time_ms', 0)
    peak_current = event_dict.get('peak_current', 0)
    event_area = event_dict.get('event_area', 0)

    event_info = (
        f"Dwell: {dwell_time_ms:.2f} ms\n"
        f"Peak Curr: {peak_current:.3f}\n")

    combined_text = event_info + "\nFit Info:\n" + info_text
    ax.text(0.02, 0.98, combined_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Plot peak marker
    peak_idx_abs = start + int(event_dict.get('normalized_position', 0.5) * event_dict.get('dwell_time', 0))
    if start <= peak_idx_abs < end:
         ax.plot(data_normalised[peak_idx_abs, 0], peak_current, 'x', color='red', markersize=8, label='Peak')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalised Current')
    ax.set_title(f"Event Analysis (ADEPT)")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    return ax


def plot_event_boundaries(data: np.ndarray, event_boundaries: list, threshold_params, boundary_method = 'SO', time_range = None):
    """"Plot events with boundaries (SO or FWHM) in subplots."""
    threshold_val, SO_val, noise_sigma, noise_mu, _, _, _ = threshold_params

    if time_range:
        start_time, end_time = time_range
        event_boundaries = [(s, e) for s, e in event_boundaries
                            if data[s, 0] >= start_time and data[e-1, 0] <= end_time] # Use e-1 for end time check
        if boundary_method == 'FWHM' and 'FWHM_points' in locals(): # Needs FWHM points passed in
             pass # Filter FWHM points if needed

    num_events = len(event_boundaries)
    if num_events == 0:
        print("No events found in the specified range.")
        return
    max_events_per_plot = 16 # Reduced for better visibility

    num_figures = int(np.ceil(num_events / max_events_per_plot))

    for fig_num in range(num_figures):
        start_plot_idx = fig_num * max_events_per_plot
        end_plot_idx = min((fig_num + 1) * max_events_per_plot, num_events)
        current_events = event_boundaries[start_plot_idx:end_plot_idx]

        rows = int(np.ceil(np.sqrt(len(current_events))))
        cols = int(np.ceil(len(current_events) / rows))

        fig, axes = plt.subplots(rows, cols, figsize=(15, 15), squeeze=False) # squeeze=False ensures axes is always 2D

        # Plot each event
        for idx, ((start, end), ax) in enumerate(zip(current_events, axes.flat)):
            if start >= end: continue # Skip invalid events

            margin = max(10, (end - start) // 4) # Adaptive margin
            plot_start = max(0, start - margin)
            plot_end = min(len(data), end + margin)

            time_plot = data[plot_start:plot_end, 0]
            signal_plot = data[plot_start:plot_end, 1]

            ax.plot(time_plot, signal_plot, color='black', linewidth=0.5)

            # Plot boundaries
            event_start_time = data[start, 0]
            event_end_time = data[end-1, 0] # Time of last point
            ax.axvline(event_start_time, color='green', linestyle='--', alpha=0.7, linewidth=1)
            ax.axvline(event_end_time, color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax.axhline(0, color='grey', linestyle=':', alpha=0.5, linewidth=0.5) # Baseline

            # Plot SO or FWHM specific info
            if boundary_method == 'SO':
                ax.axhline(SO_val, color='purple', linestyle=':', alpha=0.6, linewidth=0.8, label=f'SO ({SO_val:.2f})' if idx==0 else None)

            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel('Norm. Current', fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f')) # Format time ticks
            ax.grid(True, alpha=0.2)

        # Hide empty subplots
        for idx in range(len(current_events), rows*cols):
            axes.flat[idx].set_visible(False)

        plt.suptitle(f"Detected Events ({boundary_method} Boundaries) - Set {fig_num + 1}/{num_figures}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout

def plot_all_events_and_select(data: np.ndarray, event_boundaries: list, threshold_params, boundary_method='SO', time_range = None):
    """Plot trace with events highlighted, print details, allow user selection."""
    threshold_val, SO_val, noise_sigma, noise_mu, _, _, _ = threshold_params

    if time_range:
        start_time, end_time = time_range
        plot_mask = (data[:, 0] >= start_time) & (data[:, 0] <= end_time)
        trace_plot = data[plot_mask]
        # Adjust event boundaries indices relative to the start of the time range if needed
        # For simplicity, just filter which events are fully within range
        events_in_range = [(s, e) for s, e in event_boundaries
                           if data[s, 0] >= start_time and data[e-1, 0] <= end_time]
    else:
        trace_plot = data
        events_in_range = event_boundaries

    if len(trace_plot) == 0 or len(events_in_range) == 0:
        print("No data or events to plot in the specified range.")
        return

    # Plot full trace with highlighted events
    plt.figure(figsize=(15, 6))
    plt.plot(trace_plot[:, 0], trace_plot[:, 1], 'blue', linewidth=0.3, label='Signal')

    print("\nEvent Details:")
    for i, (start, end) in enumerate(events_in_range, 1):
        event_start_time = data[start, 0]
        event_end_time = data[end-1, 0]
        # Ensure times are within the plot range if time_range was specified
        if time_range and (event_start_time < start_time or event_end_time > end_time):
            continue
        print(f"Event {i}: Start={event_start_time:.4f} s, End={event_end_time:.4f} s, Indices={start}-{end}")
        plt.axvspan(event_start_time, event_end_time, color='red', alpha=0.2)

    plt.xlabel('Time (s)')
    plt.ylabel('Normalised Current')
    plt.title('Trace with Highlighted Events')
    plt.axhline(y=0, color='grey', linestyle=':', alpha=0.7, label='Baseline (0)')
    plt.axhline(y=threshold_val, color='black', linestyle='--', label=f'Threshold ({threshold_val:.2f})')
    if boundary_method == 'SO':
        plt.axhline(y=SO_val, color='green', linestyle='--', label=f'SO ({SO_val:.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if time_range: plt.xlim(time_range)
    plt.tight_layout()
    plt.show() # Show the overview plot

    # Get user input
    while True:
        try:
            user_input = input(f"Enter event number (1-{len(events_in_range)}) to view in detail, or 'q' to quit: ")
            if user_input.lower() == 'q':
                return
            event_num = int(user_input)
            if 1 <= event_num <= len(events_in_range):
                break
            print("Invalid event number.")
        except ValueError:
            print("Please enter a valid number or 'q'.")

    # Plot selected event
    start, end = events_in_range[event_num-1]
    margin = max(20, (end - start) // 2) # Adaptive margin
    plot_start = max(0, start - margin)
    plot_end = min(len(data), end + margin)

    plt.figure(figsize=(10, 6))
    time_plot = data[plot_start:plot_end, 0]
    signal_plot = data[plot_start:plot_end, 1]
    plt.plot(time_plot, signal_plot, 'black', linewidth=0.6)

    event_start_time = data[start, 0]
    event_end_time = data[end-1, 0]
    plt.axvline(event_start_time, color='green', linestyle='--', label='Start', alpha=0.7)
    plt.axvline(event_end_time, color='red', linestyle='--', label='End', alpha=0.7)
    plt.axhline(0, color='grey', linestyle=':', alpha=0.7, label='Baseline (0)')

    # Add SO threshold or FWHM points
    if boundary_method == 'SO':
        plt.axhline(SO_val, color='purple', linestyle=':', label=f'SO ({SO_val:.2f})', alpha=0.6)

    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Current')
    plt.title(f'Event {event_num} Detail')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show() # Show the detail plot




def interactive_cusum_analysis(data_normalised, event_start_idx, event_end_idx, initial_baseline, initial_noise_std):
    """Interactive visualisation tool for CUSUM analysis with adjustable parameters."""
    plt.close('all') # Close previous interactive plots

    event_data = data_normalised[event_start_idx:event_end_idx, 1]
    event_time = data_normalised[event_start_idx:event_end_idx, 0]
    event_length = len(event_data)

    if event_length < 5:
        print("Event too short for interactive CUSUM analysis.")
        return

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(6, 4, figure=fig)
    ax_plot = fig.add_subplot(gs[0:4, :])
    ax_stats = fig.add_subplot(gs[4, :])
    ax_stats.axis('off')
    slider_axes = [fig.add_subplot(gs[5, i]) for i in range(4)] # k, h, min_dist, min_level_pts

    # Initial parameters (try to guess reasonably)
    k_default = 0.5
    h_default = 3.0
    min_distance_default = max(3, event_length // 50)
    min_level_points_default = max(3, event_length // 100)
    adaptive_threshold_default = True
    window_size_default = min(50, max(10, event_length // 4))

    # Perform initial analysis
    event_dict = cusum_analyze_event(data_normalised, event_start_idx, event_end_idx, initial_baseline, initial_noise_std)

    plot_cusum_analysis(data_normalised, event_dict, ax=ax_plot)

    # Create sliders
    k_slider = Slider(slider_axes[0], 'k (sens)', 0.1, 1.5, valinit=k_default, valstep=0.05)
    h_slider = Slider(slider_axes[1], 'h (thresh)', 1.0, 8.0, valinit=h_default, valstep=0.1)
    min_dist_slider = Slider(slider_axes[2], 'Min Dist', 1, max(10, event_length // 10), valinit=min_distance_default, valstep=1)
    min_points_slider = Slider(slider_axes[3], 'Min Lvl Pts', 2, max(10, event_length // 5), valinit=min_level_points_default, valstep=1)

    # --- Additional parameter window ---
    fig_params = plt.figure(figsize=(10, 3))
    gs_params = GridSpec(1, 3, figure=fig_params)
    window_ax = fig_params.add_subplot(gs_params[0, 0])
    adaptive_ax = fig_params.add_subplot(gs_params[0, 1])
    reset_ax = fig_params.add_subplot(gs_params[0, 2])

    window_slider = Slider(window_ax, 'Window Size', 10, max(100, event_length // 2), valinit=window_size_default, valstep=5)
    adaptive_check = CheckButtons(adaptive_ax, ['Adapt Thresh'], [adaptive_threshold_default])
    reset_button = Button(reset_ax, 'Reset Params')

    stats_text_obj = ax_stats.text(0.01, 0.95, "", fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgrey', alpha=0.5))

    def update_display(current_event_dict):
        # Update plot
        ax_plot.clear()
        plot_cusum_analysis(data_normalised, current_event_dict, ax=ax_plot)
        # Update stats text
        stats = (f"Event: {event_start_idx}-{event_end_idx} ({current_event_dict.get('dwell_time_ms', 0):.2f} ms)\n"
                 f"Carriers: {current_event_dict.get('num_levels', 0)} | Subpeaks: {current_event_dict.get('num_subpeaks', 0)} | Changes: {len(current_event_dict.get('change_points', []))}\n")
        if current_event_dict.get('carrier_levels'):
            stats += "Carriers (Curr | Std | Pts | Frac):\n"
            for i, lvl in enumerate(current_event_dict['carrier_levels']):
                stats += f" {i+1}: {lvl['current']:.3f} | {lvl['std']:.3f} | {lvl['points']} | {lvl.get('dwell_fraction',0)*100:.1f}%\n"
        stats_text_obj.set_text(stats.strip())
        fig.canvas.draw_idle()
        fig_params.canvas.draw_idle()

    update_display(event_dict) # Initial display

    def update(val):
        k_val = k_slider.val
        h_val = h_slider.val
        min_dist_val = int(min_dist_slider.val)
        min_level_pts_val = int(min_points_slider.val)
        adapt_thresh_val = adaptive_check.get_status()[0]
        window_size_val = int(window_slider.val)

        # Rerun analysis with current slider values

        # --- Re-run detection and grouping ---
        temp_event_data = data_normalised[event_start_idx:event_end_idx, 1]
        change_points = cusum_detector(temp_event_data, k=k_val, h=h_val, min_distance=min_dist_val,
                                        adaptive_threshold=adapt_thresh_val, window_size=window_size_val, noise_std=initial_noise_std)
        levels = group_levels(temp_event_data, change_points, min_level_points=min_level_pts_val)
        # Re-identify carriers (use default fractions/separation for simplicity)
        carrier_levels = identify_carrier_levels(levels, initial_noise_std)
         # Re-detect subpeaks
        all_subpeaks = []
        for i, carrier in enumerate(carrier_levels):
             subpeaks = detect_subpeaks(temp_event_data, carrier, carrier_levels_all=carrier_levels, noise_std=initial_noise_std, min_peak_dist_points=min_dist_val)
             for sp in subpeaks: sp['carrier_idx'] = i # Add index
             all_subpeaks.extend(subpeaks)
        all_subpeaks.sort(key=lambda x: x['peak_idx'])


        # Create a temporary dict to update plot (copy relevant parts from original dict)
        temp_dict = event_dict.copy()
        temp_dict.update({
            'change_points': change_points,
            'carrier_levels': carrier_levels,
            'subpeaks': all_subpeaks,
            'num_levels': len(carrier_levels),
            'num_subpeaks': len(all_subpeaks)
        })

        # Update plot and stats
        update_display(temp_dict)

    def reset(event):
        k_slider.reset()
        h_slider.reset()
        min_dist_slider.reset()
        min_points_slider.reset()
        window_slider.reset()
        # Reset checkbox requires checking current status
        if adaptive_check.get_status()[0] != adaptive_threshold_default:
             adaptive_check.set_active(0) # Set active flips state
        update(None) # Trigger update

    # Connect sliders/buttons
    k_slider.on_changed(update)
    h_slider.on_changed(update)
    min_dist_slider.on_changed(update)
    min_points_slider.on_changed(update)
    window_slider.on_changed(update)
    adaptive_check.on_clicked(lambda label: update(None))
    reset_button.on_clicked(reset)

    def on_close_main(event): plt.close(fig_params)
    def on_close_params(event): plt.close(fig)
    fig.canvas.mpl_connect('close_event', on_close_main)
    fig_params.canvas.mpl_connect('close_event', on_close_params)

    fig.suptitle(f"Interactive CUSUM Analysis: Event {event_start_idx}-{event_end_idx}", fontsize=14)
    fig_params.suptitle("CUSUM Parameters", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_params.tight_layout(rect=[0, 0, 1, 0.9])

    # Try to position parameter window
    try:
        main_manager = fig.canvas.manager
        param_manager = fig_params.canvas.manager
        if hasattr(main_manager, 'window') and hasattr(param_manager, 'window'):
             geom = main_manager.window.geometry()
             x, y, dx, dy = geom.getRect()
             param_manager.window.setGeometry(x, y + dy + 50, 500, 150) # Position below main
    except Exception:
        print("Note: Window positioning failed.")

    plt.show()

def run_interactive_cusum_on_event(data_normalised, event_tuple, baseline_current, noise_std):
     """Helper to run interactive CUSUM on a single (start, end) tuple."""
     start, end = event_tuple
     print(f"\n--- Running Interactive CUSUM for Event {start}-{end} ---")
     interactive_cusum_analysis(data_normalised, start, end, baseline_current, noise_std)
     print(f"--- Finished Interactive CUSUM for Event {start}-{end} ---")

def interactive_event_plotter(data_normalised, event_params):
    """Interactive function to select and plot analyzed events (ADEPT or CUSUM)."""
    if not event_params:
        print("No analyzed events to plot.")
        return

    print(f"\n--- Interactive Event Analysis ---")
    print(f"Total analyzed events: {len(event_params)}")
    adept_count = sum(1 for e in event_params if e.get('analysis_method') == 'ADEPT')
    cusum_count = sum(1 for e in event_params if e.get('analysis_method') == 'CUSUM')
    print(f"  ADEPT: {adept_count}, CUSUM: {cusum_count}")

    # Summary stats
    cusum_levels = [e.get('num_levels', 0) for e in event_params if e.get('analysis_method') == 'CUSUM']
    cusum_subpeaks = [e.get('num_subpeaks', 0) for e in event_params if e.get('analysis_method') == 'CUSUM']
    if cusum_count > 0:
        print(f"CUSUM Event Stats:")
        print(f"  Avg Levels: {np.mean(cusum_levels):.1f} ± {np.std(cusum_levels):.1f}")
        print(f"  Avg Subpeaks: {np.mean(cusum_subpeaks):.1f} ± {np.std(cusum_subpeaks):.1f}")
        print(f"  Events w/ Subpeaks: {sum(1 for x in cusum_subpeaks if x > 0)}")

    while True:
        print("\nOptions:")
        print("  1. Plot specific event by index (1 to", len(event_params), ")")
        print("  2. List events (optionally filtered)")
        print("  3. Run Interactive CUSUM Tuner on an event")
        print("  4. Exit")
        choice = input("Enter choice: ").strip()

        try:
            if choice == '1':
                event_num = int(input(f"Enter event index (1-{len(event_params)}): "))
                if 1 <= event_num <= len(event_params):
                    event_to_plot = event_params[event_num - 1]
                    plt.close('all') # Close previous plots
                    fig, ax = plt.subplots(figsize=(12, 7))
                    if event_to_plot['analysis_method'] == 'ADEPT':
                        plot_adept_fit(data_normalised, event_to_plot, ax=ax)
                    elif event_to_plot['analysis_method'] == 'CUSUM':
                        plot_cusum_analysis(data_normalised, event_to_plot, ax=ax)
                    else:
                        print("Unknown analysis method for this event.")
                        continue
                    ax.set_title(f"Event {event_num} ({event_to_plot['event_start_idx']}-{event_to_plot['event_end_idx']}) - {event_to_plot['analysis_method']}")
                    plt.tight_layout()
                    plt.show()
                else:
                    print("Invalid index.")

            elif choice == '2':
                filter_type = input("Filter by method? (a=ADEPT, c=CUSUM, n=None): ").lower()
                max_list = 20
                count = 0
                print("\nEvent List:")
                for i, event in enumerate(event_params):
                    method = event.get('analysis_method', 'Unknown')
                    show = False
                    if filter_type == 'n': show = True
                    elif filter_type == 'a' and method == 'ADEPT': show = True
                    elif filter_type == 'c' and method == 'CUSUM': show = True

                    if show:
                        count += 1
                        start, end = event['event_start_idx'], event['event_end_idx']
                        dwell = event.get('dwell_time_ms', 0)
                        extra = ""
                        if method == 'CUSUM':
                            extra = f"| Lvls: {event.get('num_levels', 0)}, SPks: {event.get('num_subpeaks', 0)}"
                        elif method == 'ADEPT':
                             r2 = event.get('fitted_params', {}).get('r_squared', float('nan'))
                             extra = f"| Fit R²: {r2:.2f}" if not np.isnan(r2) else "| Fit Failed"

                        print(f"  {i+1}: {start}-{end} | {dwell:.2f} ms | {method} {extra}")
                        if count >= max_list:
                            print(f"  ... (showing first {max_list})")
                            break
                if count == 0: print("No events match filter.")


            elif choice == '3':
                 event_num = int(input(f"Enter CUSUM event index (1-{len(event_params)}) to tune: "))
                 if 1 <= event_num <= len(event_params):
                     event_to_tune = event_params[event_num - 1]
                     if event_to_tune['analysis_method'] == 'CUSUM':
                         # Need baseline and noise used for the original analysis
                         # Get them from the event dict if stored, else use initial estimates
                         baseline = event_to_tune.get('baseline_current', 0.0)
                         noise_std = event_to_tune.get('noise_std', 0.1) # Need a reasonable default
                         run_interactive_cusum_on_event(data_normalised,
                                                       (event_to_tune['event_start_idx'], event_to_tune['event_end_idx']),
                                                       baseline, noise_std)
                     else:
                         print("Selected event was not analysed with CUSUM.")
                 else:
                     print("Invalid index.")


            elif choice == '4':
                print("Exiting interactive plotter.")
                break
            else:
                print("Invalid choice.")

        except ValueError:
            print("Invalid input. Please enter a number where expected.")
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()


def save_event_params_csv(data_normalised, event_params, csv_filename):
    """
    Save extracted event parameters (both ADEPT and CUSUM) to a CSV file.
    """
    if not event_params:
        print("No event parameters to save.")
        return

    processed_events_for_csv = []
    time_vec = data_normalised[:, 0]

    # This set will collect all unique ADEPT parameter keys with their unit suffixes
    all_adept_param_keys_discovered = set()

    for i, original_event_dict in enumerate(event_params):
        # Use a copy to avoid modifying the original list of dicts
        processed_event_row = original_event_dict.copy()

        # --- 1. Core Identifiers & Calculated Timing ---
        processed_event_row['event_display_idx'] = i + 1
        # 'event_idx' from original dict is the 0-based pipeline index
        processed_event_row['event_pipeline_idx'] = processed_event_row.get('event_idx', i)


        start_idx = processed_event_row['event_start_idx']
        # Assuming event_end_idx is exclusive (i.e., event data is data[start_idx:event_end_idx])
        end_idx_exclusive = processed_event_row['event_end_idx']

        # Calculate event start and end times in seconds
        if start_idx < len(time_vec) and end_idx_exclusive > start_idx and end_idx_exclusive <= len(time_vec):
            processed_event_row['event_start_time_s'] = time_vec[start_idx]
            # Timestamp of the last point within the event
            processed_event_row['event_end_time_s'] = time_vec[end_idx_exclusive - 1]

            # Use dwell_time_ms from event dict if available, as it's directly calculated
            if 'dwell_time_ms' in processed_event_row and processed_event_row['dwell_time_ms'] is not None:
                processed_event_row['dwell_time_s'] = processed_event_row['dwell_time_ms'] / 1000.0
            else: # Fallback: calculate from start/end times
                # This calculation defines duration as time from first point to last point.
                processed_event_row['dwell_time_s'] = processed_event_row['event_end_time_s'] - processed_event_row['event_start_time_s']
                # If dwell_time_ms was missing, populate it from the calculated dwell_time_s
                if 'dwell_time_ms' not in processed_event_row or processed_event_row['dwell_time_ms'] is None:
                    processed_event_row['dwell_time_ms'] = processed_event_row['dwell_time_s'] * 1000.0
        else:
            # Handle invalid event indices or very short/problematic events
            processed_event_row['event_start_time_s'] = time_vec[start_idx] if start_idx < len(time_vec) else -1.0
            processed_event_row['event_end_time_s'] = processed_event_row['event_start_time_s']
            processed_event_row['dwell_time_s'] = 0.0
            if 'dwell_time_ms' not in processed_event_row or processed_event_row['dwell_time_ms'] is None:
                 processed_event_row['dwell_time_ms'] = 0.0

        # --- 2. Rename/Clarify Existing Fields with Units & Specific Prefixes ---
        if 'dwell_time' in processed_event_row:
            processed_event_row['dwell_time_points'] = processed_event_row.pop('dwell_time')

        # Fields representing current values (suffix _norm for "normalized current units")
        current_fields_map = {
            # Common fields (content depends on analysis_method)
            'carrier_current': 'carrier_current_norm',     # ADEPT: uses peak_current; CUSUM: uses weighted_avg_carrier
            'baseline_current': 'baseline_current_norm',   # ADEPT: uses fitted i_0 (approx); CUSUM: uses noise_mu
            # ADEPT specific current value
            'peak_current': 'adept_peak_current_norm',
            # CUSUM specific current values
            'event_max': 'cusum_max_current_norm',
            'event_min': 'cusum_min_current_norm',
            'event_mean': 'cusum_mean_current_norm',
            'event_median': 'cusum_median_current_norm',
        }
        for old_key, new_key in current_fields_map.items():
            if old_key in processed_event_row:
                processed_event_row[new_key] = processed_event_row.pop(old_key)

        # Event area (typically sum of current values within event * 1 data point interval)
        if 'event_area' in processed_event_row:
            processed_event_row['event_area_norm_current_x_points'] = processed_event_row.pop('event_area')

        # Standard deviation fields
        std_dev_fields_map = {
            # Noise std dev used by the analysis step (ADEPT: residual_std, CUSUM: global_noise_sigma from detection)
            'noise_std': 'event_processing_noise_std_norm',
            # Std dev of current values within a CUSUM event segment
            'event_std': 'cusum_event_internal_current_std_norm'
        }
        for old_key, new_key in std_dev_fields_map.items():
            if old_key in processed_event_row:
                processed_event_row[new_key] = processed_event_row.pop(old_key)

        # CUSUM specific counts (add cusum_ prefix for clarity)
        if processed_event_row.get('analysis_method') == 'CUSUM':
            if 'num_levels' in processed_event_row: # from cusum_analyze_event output
                processed_event_row['cusum_num_carrier_levels'] = processed_event_row.pop('num_levels')
            if 'num_subpeaks' in processed_event_row: # from cusum_analyze_event output
                processed_event_row['cusum_num_subpeaks'] = processed_event_row.pop('num_subpeaks')

        # --- 3. ADEPT Specific: Flatten "fitted_params" with "adept_" Prefix and Unit Suffixes ---
        if processed_event_row.get('analysis_method') == 'ADEPT' and \
           isinstance(processed_event_row.get('fitted_params'), dict):
            adept_fitted_params = processed_event_row.pop('fitted_params') # Remove original
            for param_key, param_value in adept_fitted_params.items():
                if param_key in ['fitted_values', 'param_covariance']: # Exclude complex/large items
                    continue
                
                new_adept_key_with_unit = f"adept_{param_key}"

                # Add unit suffixes for known ADEPT parameters based on their typical meaning
                if param_key in ['mu_1', 'mu_2', 'tau_rise', 'tau_fall'] or \
                   (param_key.startswith('mu_') and 'beta' not in param_key) or \
                   (param_key.startswith('tau_') and 'beta' not in param_key): # Heuristic for time constants
                    new_adept_key_with_unit += "_s" # Time in seconds
                elif param_key in ['i_0', 'a'] or param_key.startswith('a_'): # Heuristic for current amplitudes
                    new_adept_key_with_unit += "_norm" # Normalized current units

                processed_event_row[new_adept_key_with_unit] = param_value
                all_adept_param_keys_discovered.add(new_adept_key_with_unit) # Collect for header
        
        # --- 4. Remove Original Complex Fields Not Suitable for Single CSV Row ---
        # These fields are lists or dicts and are summarised by other fields (e.g., counts)
        for complex_key_to_remove in ['carrier_levels', 'subpeaks', 'change_points', 'fitted_params']:
            if complex_key_to_remove in processed_event_row:
                del processed_event_row[complex_key_to_remove]

        processed_events_for_csv.append(processed_event_row)

    if not processed_events_for_csv:
        print("No events remained after processing for CSV.")
        return

    # --- 5. Determine Final CSV Headers in a Logical Order ---
    # Start with a defined order for the most critical fields.
    ordered_core_fieldnames = [
        'event_display_idx', 'event_pipeline_idx',
        'event_start_time_s', 'event_end_time_s',
        'dwell_time_s', 'dwell_time_ms', 'dwell_time_points',
        'analysis_method', 'event_type',
        'carrier_current_norm',      # Primary current metric (ADEPT uses its peak, CUSUM uses weighted avg)
        'baseline_current_norm',     # Baseline reference for the event (ADEPT uses i0, CUSUM uses noise_mu)
        'event_area_norm_current_x_points', # Sum of current values in event (norm. current * points)
        'event_processing_noise_std_norm'   # Noise context for the analysis (e.g. residuals_std or global_noise_sigma)
    ]

    # Discover all other unique keys found across all processed event dicts
    all_other_keys_found_in_rows = set()
    for row_dict in processed_events_for_csv:
        all_other_keys_found_in_rows.update(row_dict.keys())
    
    # Exclude core fields (already defined) and ADEPT keys (handled separately by all_adept_param_keys_discovered)
    # These remaining keys are typically CUSUM-specific or other general event properties.
    remaining_general_and_cusum_keys = sorted(list(
        all_other_keys_found_in_rows - set(ordered_core_fieldnames) - all_adept_param_keys_discovered
    ))
    
    # Sort discovered ADEPT keys alphabetically for consistent ordering in the CSV
    sorted_adept_specific_keys = sorted(list(all_adept_param_keys_discovered))

    # Combine all parts for the final fieldnames list
    # Order: Core fields, then ADEPT-specific fields, then other CUSUM/general fields
    final_csv_fieldnames = ordered_core_fieldnames + sorted_adept_specific_keys + remaining_general_and_cusum_keys
    
    # Ensure no duplicates in the final list (should be handled by set operations, but as a safeguard)
    seen_final_fields = set()
    unique_final_fieldnames_ordered = []
    for field in final_csv_fieldnames:
        if field not in seen_final_fields:
            unique_final_fieldnames_ordered.append(field)
            seen_final_fields.add(field)
    final_csv_fieldnames = unique_final_fieldnames_ordered

    print(f"CSV Headers: {final_csv_fieldnames}")

    # --- 6. Write to CSV File ---
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=final_csv_fieldnames, extrasaction='ignore')
        writer.writeheader()
        for event_row_to_write in processed_events_for_csv:
            writer.writerow(event_row_to_write)

    print(f"Successfully saved {len(processed_events_for_csv)} events to {csv_filename}")