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
        
        # Isolate Oxyhemoglobin for visualization
        raw_hbo = raw_haemo.copy().pick(picks='hbo')
        
        # NOTE: We DO NOT delete the tmp_path here.
        # MNE requires the file to stay on disk to perform background operations like .crop()
        
        return raw_hbo
        
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

raw_hbo = load_and_prep_data(uploaded_file)

if raw_hbo is None:
    st.info("Please drag and drop a .snirf file into the sidebar to begin.")
else:
    # --- DATA TRIMMING UI ---
    st.sidebar.header("3. Data Trimming")
    max_time = float(raw_hbo.times[-1])
    
    # Dual-handled slider for cropping out motion artifacts at the start/end
    trim_range = st.sidebar.slider("Select Time Range (s)", 
                                   min_value=0.0, 
                                   max_value=max_time, 
                                   value=(0.0, max_time), 
                                   step=1.0)

    # --- APPLY CROP TO A COPY OF THE DATA ---
    working_raw = raw_hbo.copy().crop(tmin=trim_range[0], tmax=trim_range[1])

    # --- VIEW OPTIONS ---
    st.sidebar.header("4. View Options")
    ch_names = working_raw.ch_names
    
    channel_option = st.sidebar.selectbox("Select Channel to View", ["Grand Average"] + ch_names)
    overlay_raw = st.sidebar.checkbox("Overlay Raw Data", value=True)
    show_markers = st.sidebar.checkbox("Show Event Markers", value=True)

    # --- DATA EXTRACTION ---
    times = working_raw.times
    fs = working_raw.info['sfreq']
    
    if channel_option == "Grand Average":
        data_raw = np.mean(working_raw.get_data(), axis=0)
    else:
        ch_idx = ch_names.index(channel_option)
        data_raw = working_raw.get_data()[ch_idx, :]

    # Compute raw PSD
    freqs_raw, psd_raw = signal.welch(data_raw, fs, nperseg=1024)
    psd_raw_db = 10 * np.log10(psd_raw)

    # --- FILTER LOGIC ---
    if apply_filter:
        if highpass >= lowpass:
            st.error("⚠️ **Filter Error:** The High-pass cutoff must be strictly lower than the Low-pass cutoff. Please adjust the sliders.")
            st.stop() 

        b, a = signal.butter(4, [highpass, lowpass], btype='bandpass', fs=fs)
        data_to_plot = signal.filtfilt(b, a, data_raw)
        
        freqs_filt, psd_filt = signal.welch(data_to_plot, fs, nperseg=1024)
        psd_to_plot_db = 10 * np.log10(psd_filt)
    else:
        data_to_plot = data_raw
        psd_to_plot_db = psd_raw_db

    # --- BUILD THE PLOTS ---
    col1, col2 = st.columns(2) 

    # 1. Time Domain Plot
    with col1:
        st.subheader(f"Time Domain: {channel_option}")
        fig_time = go.Figure()
        
        if apply_filter and overlay_raw:
            fig_time.add_trace(go.Scatter(x=times, y=data_raw, mode='lines', 
                                          name='Raw Signal', line=dict(color='lightgrey', width=1)))
            
        line_color = '#1f77b4' if apply_filter else 'lightgrey'
        line_name = 'Filtered Signal' if apply_filter else 'Raw Signal'
        
        fig_time.add_trace(go.Scatter(x=times, y=data_to_plot, mode='lines', 
                                      name=line_name, line=dict(color=line_color, width=2)))

        # --- ADD EVENT MARKERS ---
        if show_markers and len(working_raw.annotations) > 0:
            unique_desc = list(set(working_raw.annotations.description))
            # Color palette for distinct events
            colors = ['#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#ff7f0e'] 
            color_map = {desc: colors[i % len(colors)] for i, desc in enumerate(unique_desc)}
            
            added_to_legend = set()

            for ann in working_raw.annotations:
                onset = ann['onset']
                duration = ann['duration']
                desc = ann['description']
                c = color_map[desc]
                
                show_leg = desc not in added_to_legend
                added_to_legend.add(desc)

                if duration > 0:
                    # Block design events
                    fig_time.add_vrect(x0=onset, x1=onset+duration, fillcolor=c, opacity=0.15, line_width=0, layer="below")
                    if show_leg:
                        fig_time.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color=c, symbol='square', size=12), name=f"Event: {desc}"))
                else:
                    # Point events
                    fig_time.add_vline(x=onset, line_color=c, line_dash="dash", line_width=1.5, layer="below")
                    if show_leg:
                        fig_time.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color=c, dash='dash', width=2), name=f"Marker: {desc}"))

        fig_time.update_layout(xaxis_title="Time (s)", yaxis_title="Amplitude (µM)", 
                               template="plotly_white", legend=dict(x=0.01, y=0.99))
        st.plotly_chart(fig_time, use_container_width=True)

    # 2. Frequency Domain (PSD) Plot
    with col2:
        st.subheader(f"Frequency Domain (PSD): {channel_option}")
        fig_psd = go.Figure()
        
        if apply_filter and overlay_raw:
            fig_psd.add_trace(go.Scatter(x=freqs_raw, y=psd_raw_db, mode='lines', 
                                         name='Raw PSD', line=dict(color='lightgrey', width=1)))
            
        line_color_psd = 'red' if apply_filter else 'lightgrey'
        line_name_psd = 'Filtered PSD' if apply_filter else 'Raw PSD'

        fig_psd.add_trace(go.Scatter(x=freqs_raw, y=psd_to_plot_db, mode='lines', 
                                     name=line_name_psd, line=dict(color=line_color_psd, width=2)))
        
        fig_psd.add_vrect(x0=0.01, x1=0.08, fillcolor="blue", opacity=0.1, line_width=0, annotation_text="Neural Hemodynamics")
        # Add the Physiological Artifact Bands
        fig_psd.add_vrect(x0=0.05, x1=0.15, fillcolor="orange", opacity=0.15, line_width=0, annotation_text="Mayer Waves")
        fig_psd.add_vrect(x0=0.2, x1=0.4, fillcolor="green", opacity=0.15, line_width=0, annotation_text="Respiration")
        fig_psd.add_vrect(x0=0.8, x1=1.5, fillcolor="red", opacity=0.15, line_width=0, annotation_text="Cardiac")
        
        fig_psd.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="Power (dB)", 
                              xaxis_range=[0, 2.0], template="plotly_white", legend=dict(x=0.80, y=0.99))
        st.plotly_chart(fig_psd, use_container_width=True)
