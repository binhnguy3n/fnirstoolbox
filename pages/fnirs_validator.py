import streamlit as st
import mne
import numpy as np
from scipy import signal
import plotly.graph_objects as go
import pandas as pd
import tempfile
import os

# --- PAGE SETUP ---
st.set_page_config(layout="wide", page_title="fNIRS Filter Validator")
st.title("fNIRS Filter Validation Dashboard")
st.markdown("Interactive tool to visually validate bandpass filters, edit markers, and generate Event-Related Averages (ERA).")

# --- SIDEBAR 1: UPLOAD ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a .snirf file", type=["snirf"])

# --- SIDEBAR 2: FILTER ---
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
        with tempfile.NamedTemporaryFile(delete=False, suffix='.snirf') as tmp_file:
            tmp_file.write(file_data.getvalue())
            tmp_path = tmp_file.name
            
        raw = mne.io.read_raw_snirf(tmp_path, preload=True)
        raw_od = mne.preprocessing.nirs.optical_density(raw)
        # We stop at beer_lambert_law so we keep both HbO and HbR in memory
        raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)
        return raw_haemo
        
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

raw_haemo = load_and_prep_data(uploaded_file)

if raw_haemo is None:
    st.info("Please drag and drop a .snirf file into the sidebar to begin.")
else:
    # --- SIDEBAR 3: MARKER EDITOR ---
    st.sidebar.header("3. Marker Editor (TMS)")
    if len(raw_haemo.annotations) > 0:
        ann_df = pd.DataFrame({
            'onset': raw_haemo.annotations.onset,
            'duration': raw_haemo.annotations.duration,
            'description': raw_haemo.annotations.description
        })
        
        edited_df = st.sidebar.data_editor(ann_df, num_rows="dynamic", hide_index=True, use_container_width=True)
        
        # Apply edits back to the raw data
        edited_df = edited_df.dropna(subset=['onset', 'description'])
        new_annotations = mne.Annotations(
            onset=edited_df['onset'].values,
            duration=edited_df['duration'].values,
            description=edited_df['description'].values
        )
        raw_haemo.set_annotations(new_annotations)
    else:
        st.sidebar.info("No markers found.")

    # --- SIDEBAR 4: HEMOGLOBIN SELECTION ---
    st.sidebar.header("4. Hemoglobin Selection")
    hemo_type = st.sidebar.radio("View Signal Type", ["HbO (Oxygenated)", "HbR (Deoxygenated)", "HbTot (Total)"])
    
    # Set dynamic theme colors based on the chromophore
    if "HbO" in hemo_type:
        theme_color = "#D62728" # Red
        target_raw = raw_haemo.copy().pick(picks='hbo')
    elif "HbR" in hemo_type:
        theme_color = "#1F77B4" # Blue
        target_raw = raw_haemo.copy().pick(picks='hbr')
    else:
        theme_color = "#2CA02C" # Green
        # Manually calculate HbTot = HbO + HbR
        raw_hbo = raw_haemo.copy().pick(picks='hbo')
        raw_hbr = raw_haemo.copy().pick(picks='hbr')
        hbt_data = raw_hbo.get_data() + raw_hbr.get_data()
        
        info = raw_hbo.info.copy()
        mne.rename_channels(info, {ch: ch.replace('hbo', 'hbt') for ch in info.ch_names})
        target_raw = mne.io.RawArray(hbt_data, info, verbose=False)
        target_raw.set_annotations(raw_haemo.annotations)

    # --- SIDEBAR 5: TRIMMING ---
    st.sidebar.header("5. Data Trimming")
    max_time = float(target_raw.times[-1])
    trim_range = st.sidebar.slider("Select Time Range (s)", min_value=0.0, max_value=max_time, value=(0.0, max_time), step=1.0)
    working_raw = target_raw.copy().crop(tmin=trim_range[0], tmax=trim_range[1])

    # --- SIDEBAR 6: VIEW OPTIONS ---
    st.sidebar.header("6. View Options")
    ch_names = working_raw.ch_names
    channel_option = st.sidebar.selectbox("Select Channel to View", ["Grand Average"] + ch_names)
    overlay_raw = st.sidebar.checkbox("Overlay Raw Data", value=True)
    show_markers = st.sidebar.checkbox("Show Event Markers", value=True)

    # --- DATA EXTRACTION & FILTERING ---
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

    # Apply stable SOS Filter
    if apply_filter:
        if highpass >= lowpass:
            st.error("⚠️ **Filter Error:** The High-pass cutoff must be strictly lower than the Low-pass cutoff.")
            st.stop() 
        sos = signal.butter(4, [highpass, lowpass], btype='bandpass', fs=fs, output='sos')
        data_to_plot = signal.sosfiltfilt(sos, data_raw)
        
        freqs_filt, psd_filt = signal.welch(data_to_plot, fs, nperseg=1024)
        psd_to_
