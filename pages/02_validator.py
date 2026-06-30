import streamlit as st
import mne
import numpy as np
from scipy import signal
import plotly.graph_objects as go
import tempfile
import os

# --- PAGE SETUP ---
st.set_page_config(layout="wide", page_title="fNIRS Filter Validator")
st.title("fNIRS Filter Validation Dashboard")
st.markdown("Interactive tool to visually validate bandpass filter parameters and trim data against raw fNIRS recordings.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a .snirf file", type=["snirf"])

st.sidebar.header("2. Filter Parameters")
apply_filter = st.sidebar.checkbox("Apply Bandpass Filter", value=True)
highpass = st.sidebar.slider("High-pass Cutoff (Hz)", min_value=0.00, max_value=0.10, value=0.01, step=0.01)
lowpass = st.sidebar.slider("Low-pass Cutoff (Hz)", min_value=0.05, max_value=2.00, value=0.08, step=0.01)

# --- LOAD DATA (Cached for speed) ---
@st.cache_data
def load_and_prep_data(file_data):
    if file_data is None:
        return None
        
    try:
        # Create a temporary file on the hard drive
        with tempfile.NamedTemporaryFile(delete=False, suffix='.snirf') as tmp_file:
            tmp_file.write(file_data.getvalue())
            tmp_path = tmp_file.name
            
        # Load the data into MNE using the temporary path
        raw = mne.io.read_raw_snirf(tmp_path, preload=True)
        raw_od = mne.preprocessing.nirs.optical_density(raw)
        raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)
        
        # NOTE: We DO NOT delete the tmp_path here.
        # MNE requires the file to stay on disk to perform background operations like .crop()
        
        # Return all three states of the data
        return {'raw': raw, 'od': raw_od, 'haemo': raw_haemo}
        
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

data_dict = load_and_prep_data(uploaded_file)

if data_dict is None:
    st.info("Please drag and drop a .snirf file into the sidebar to begin.")
else:
    # --- DATA TRIMMING & SELECTION UI ---
    st.sidebar.header("3. Data Trimming & Selection")
    
    # State selection (Raw, OD, Hemoglobin)
    signal_state = st.sidebar.radio("Select Signal State", ["Hemoglobin", "Optical Density (OD)", "Raw Wavelength (WL)"])
    
    # Set the active data object based on selection
    if signal_state == "Hemoglobin":
        active_raw = data_dict['haemo']
        y_axis_title = "Amplitude (Molar)"
    elif signal_state == "Optical Density (OD)":
        active_raw = data_dict['od']
        y_axis_title = "Optical Density (AU)"
    else:
        active_raw = data_dict['raw']
        y_axis_title = "Intensity (AU)"
        
    max_time = float(active_raw.times[-1])
    
    # Dual-handled slider for cropping out motion artifacts at the start/end
    trim_range = st.sidebar.slider("Select Time Range (s)", 
                                   min_value=0.0, 
                                   max_value=max_time, 
                                   value=(0.0, max_time), 
                                   step=1.0)

    # --- APPLY CROP TO A COPY OF THE DATA ---
    working_raw = active_raw.copy().crop(tmin=trim_range[0], tmax=trim_range[1])

    # --- VIEW OPTIONS ---
    st.sidebar.header("4. View Options")
    
    # Dynamically extract base channel names and signal components based on the state
    if signal_state == "Hemoglobin":
        base_chans = list(set([ch.replace(' hbo', '').replace(' hbr', '') for ch in working_raw.ch_names if ' hbo' in ch or ' hbr' in ch]))
        available_types = ["HbO", "HbR", "HbTot"]
        default_types = ["HbO", "HbR"]
    else:
        # Extract base channels (e.g., 'S1_D1') and available wavelengths (e.g., '760', '850')
        base_chans = list(set([ch.rsplit(' ', 1)[0] for ch in working_raw.ch_names if ' ' in ch]))
        available_types = list(set([ch.rsplit(' ', 1)[1] for ch in working_raw.ch_names if ' ' in ch]))
        available_types.sort()
        default_types = available_types

    base_chans.sort()
    
    channel_option = st.sidebar.selectbox("Select Channel to View", ["Grand Average"] + base_chans)
    
    # Multiselect for subtypes (Wavelengths or Hemoglobin types)
    selected_types = st.sidebar.multiselect(f"Select Sub-Signal(s)", available_types, default=default_types)
    
    overlay_raw = st.sidebar.checkbox("Overlay Raw Data", value=True)
    
    # Marker Options
    show_markers = st.sidebar.checkbox("Show Event Markers", value=True)
    if show_markers:
        marker_opacity = st.sidebar.slider("Marker Opacity", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
        override_durations = st.sidebar.checkbox("Override Event Durations")
        if override_durations:
            custom_duration = st.sidebar.number_input("Custom Duration (s)", min_value=0.0, value=10.0, step=1.0)

    if not selected_types:
        st.warning("Please select at least one Sub-Signal from the sidebar to view plots.")
        st.stop()

    # --- DATA EXTRACTION HELPER ---
    times = working_raw.times
    fs = working_raw.info['sfreq']
    
    def get_signal_data(base_ch, sig_type):
        """Helper to extract channels based on user selection and current state"""
        if signal_state == "Hemoglobin":
            if base_ch == "Grand Average":
                chans_o = [ch for ch in working_raw.ch_names if ' hbo' in ch]
                chans_r = [ch for ch in working_raw.ch_names if ' hbr' in ch]
                
                if sig_type == "HbO":
                    return np.mean(working_raw.get_data(picks=chans_o), axis=0)
                elif sig_type == "HbR":
                    return np.mean(working_raw.get_data(picks=chans_r), axis=0)
                else: # HbTot
                    return np.mean(working_raw.get_data(picks=chans_o), axis=0) + np.mean(working_raw.get_data(picks=chans_r), axis=0)
            else:
                ch_o = f"{base_ch} hbo"
                ch_r = f"{base_ch} hbr"
                
                if sig_type == "HbO":
                    return working_raw.get_data(picks=[ch_o])[0, :]
                elif sig_type == "HbR":
                    return working_raw.get_data(picks=[ch_r])[0, :]
                else: # HbTot
                    return working_raw.get_data(picks=[ch_o])[0, :] + working_raw.get_data(picks=[ch_r])[0, :]
        
        else: # Raw WL or OD
            if base_ch == "Grand Average":
                chans = [ch for ch in working_raw.ch_names if ch.endswith(f" {sig_type}")]
                return np.mean(working_raw.get_data(picks=chans), axis=0)
            else:
                ch_name = f"{base_ch} {sig_type}"
                return working_raw.get_data(picks=[ch_name])[0, :]

    # --- FILTER VALIDATION ---
    if apply_filter and highpass >= lowpass:
        st.error("⚠️ **Filter Error:** The High-pass cutoff must be strictly lower than the Low-pass cutoff. Please adjust the sliders.")
        st.stop() 

    # --- BUILD THE PLOTS ---
    col1, col2 = st.columns(2) 
    
    with col1:
        st.subheader(f"Time Domain: {channel_option}")
        fig_time = go.Figure()
        
    with col2:
        st.subheader(f"Frequency Domain (PSD): {channel_option}")
        fig_psd = go.Figure()

    # Color definitions mapping
    color_theme = {
        "HbO": {"main": "#d62728", "faded": "rgba(214, 39, 40, 0.3)"},  # Red
        "HbR": {"main": "#1f77b4", "faded": "rgba(31, 119, 180, 0.3)"}, # Blue
        "HbTot": {"main": "#2ca02c", "faded": "rgba(44, 160, 44, 0.3)"} # Green
    }
    
    # Generic fallback colors for WLs if not Hemoglobin
    fallback_colors = [
        {"main": "#9467bd", "faded": "rgba(148, 103, 189, 0.3)"}, # Purple
        {"main": "#ff7f0e", "faded": "rgba(255, 127, 14, 0.3)"},  # Orange
        {"main": "#8c564b", "faded": "rgba(140, 86, 75, 0.3)"}    # Brown
    ]

    # Loop through each selected component and add it to the plots
    for idx, sig_type in enumerate(selected_types):
        data_raw = get_signal_data(channel_option, sig_type)
        
        # Determine Color
        if sig_type in color_theme:
            c_main = color_theme[sig_type]["main"]
            c_fade = color_theme[sig_type]["faded"]
        else:
            c_main = fallback_colors[idx % len(fallback_colors)]["main"]
            c_fade = fallback_colors[idx % len(fallback_colors)]["faded"]
            
        # Compute raw PSD
        freqs_raw, psd_raw = signal.welch(data_raw, fs, nperseg=1024)
        psd_raw_db = 10 * np.log10(psd_raw)

        # Apply Filter if requested
        if apply_filter:
            sos = signal.butter(4, [highpass, lowpass], btype='bandpass', fs=fs, output='sos')
            data_to_plot = signal.sosfiltfilt(sos, data_raw)
            
            freqs_filt, psd_filt = signal.welch(data_to_plot, fs, nperseg=1024)
            psd_to_plot_db = 10 * np.log10(psd_filt)
        else:
            data_to_plot = data_raw
            psd_to_plot_db = psd_raw_db

        # 1. Populate Time Domain Plot
        if apply_filter and overlay_raw:
            fig_time.add_trace(go.Scatter(x=times, y=data_raw, mode='lines', 
                                          name=f'Raw {sig_type}', line=dict(color=c_fade, width=1)))
            
        line_name = f'Filtered {sig_type}' if apply_filter else f'Raw {sig_type}'
        fig_time.add_trace(go.Scatter(x=times, y=data_to_plot, mode='lines', 
                                      name=line_name, line=dict(color=c_main, width=2)))

        # 2. Populate Frequency Domain (PSD) Plot
        if apply_filter and overlay_raw:
            fig_psd.add_trace(go.Scatter(x=freqs_raw, y=psd_raw_db, mode='lines', 
                                         name=f'Raw {sig_type} PSD', line=dict(color=c_fade, width=1)))
            
        line_name_psd = f'Filtered {sig_type} PSD' if apply_filter else f'Raw {sig_type} PSD'
        fig_psd.add_trace(go.Scatter(x=freqs_raw, y=psd_to_plot_db, mode='lines', 
                                     name=line_name_psd, line=dict(color=c_main, width=2)))

    # --- FINALIZE TIME PLOT (MARKERS & LAYOUT) ---
    with col1:
        if show_markers and len(active_raw.annotations) > 0:
            unique_desc = list(set(active_raw.annotations.description))
            colors = ['#7f7f7f', '#bcbd22', '#17becf', '#e377c2', '#8c564b', '#9467bd'] 
            color_map = {desc: colors[i % len(colors)] for i, desc in enumerate(unique_desc)}
            added_to_legend = set()

            for ann in active_raw.annotations:
                orig_onset = ann['onset']
                
                if override_durations:
                    duration = custom_duration
                else:
                    duration = ann['duration']
                    
                desc = ann['description']
                
                if trim_range[0] <= orig_onset <= trim_range[1]:
                    aligned_onset = orig_onset - trim_range[0]
                    c = color_map[desc]
                    show_leg = desc not in added_to_legend
                    added_to_legend.add(desc)

                    if duration > 0:
                        fig_time.add_vrect(x0=aligned_onset, x1=aligned_onset+duration, fillcolor=c, opacity=marker_opacity, line_width=0, layer="below")
                        if show_leg:
                            fig_time.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color=c, symbol='square', size=12), name=f"Event: {desc}"))
                    else:
                        fig_time.add_vline(x=aligned_onset, line_color=c, line_dash="dash", line_width=2.5, opacity=marker_opacity, layer="below")
                        if show_leg:
                            fig_time.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color=c, dash='dash', width=2), name=f"Marker: {desc}"))

        fig_time.update_layout(xaxis_title="Time (s)", yaxis_title=y_axis_title, 
                               template="plotly_white", legend=dict(x=0.01, y=0.99))
        st.plotly_chart(fig_time, use_container_width=True)

    # --- FINALIZE PSD PLOT (BANDS & LAYOUT) ---
    with col2:
        # Move overlapping text to the bottom using annotation_position="bottom left"
        fig_psd.add_vrect(x0=0.01, x1=0.08, fillcolor="blue", opacity=0.05, line_width=0, annotation_text="Neural", annotation_position="bottom left", annotation_textangle=-90)
        fig_psd.add_vrect(x0=0.05, x1=0.15, fillcolor="orange", opacity=0.1, line_width=0, annotation_text="Mayer Waves", annotation_position="bottom left", annotation_textangle=-90)
        fig_psd.add_vrect(x0=0.2, x1=0.4, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Respiration", annotation_position="bottom left", annotation_textangle=-90)
        fig_psd.add_vrect(x0=0.8, x1=1.5, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Cardiac", annotation_position="bottom left", annotation_textangle=-90)
        
        fig_psd.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="Power (dB)", 
                              xaxis_range=[0, 2.0], template="plotly_white", legend=dict(x=0.80, y=0.99))
        st.plotly_chart(fig_psd, use_container_width=True)
