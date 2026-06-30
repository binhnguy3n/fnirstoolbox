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

# --- SIDEBAR CONTAINERS (Defines Visual Order) ---
# By creating these containers first, we can control exactly what order the UI 
# renders in, regardless of what order the background math executes.
cont_upload = st.sidebar.container()
cont_filter = st.sidebar.container()
cont_bulk = st.sidebar.container()
cont_hemo = st.sidebar.container()
cont_trim = st.sidebar.container()
cont_view = st.sidebar.container()
cont_manual = st.sidebar.container()

# --- 1. UPLOAD ---
cont_upload.header("1. Upload Data")
uploaded_file = cont_upload.file_uploader("Upload a .snirf file", type=["snirf"])

# --- 2. FILTER ---
cont_filter.header("2. Filter Parameters")
apply_filter = cont_filter.checkbox("Apply Bandpass Filter", value=True)
highpass = cont_filter.slider("High-pass Cutoff (Hz)", min_value=0.00, max_value=0.10, value=0.01, step=0.01)
lowpass = cont_filter.slider("Low-pass Cutoff (Hz)", min_value=0.05, max_value=2.00, value=0.08, step=0.01)

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
        raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)
        return raw_haemo
        
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

raw_haemo = load_and_prep_data(uploaded_file)

if raw_haemo is None:
    st.info("Please drag and drop a .snirf file into the sidebar to begin.")
else:
    # --- 3. BULK MARKER UPDATE ---
    cont_bulk.header("3. Bulk Marker Update")
    if len(raw_haemo.annotations) > 0:
        col_bulk1, col_bulk2 = cont_bulk.columns(2)
        bulk_desc = col_bulk1.text_input("Name", value="TMS")
        bulk_dur = col_bulk2.number_input("Duration (s)", value=0.2, step=0.1)
        
        if cont_bulk.button("Apply Bulk Update"):
            new_bulk_annots = mne.Annotations(
                onset=raw_haemo.annotations.onset,
                duration=[bulk_dur] * len(raw_haemo.annotations),
                description=[bulk_desc] * len(raw_haemo.annotations),
                orig_time=raw_haemo.annotations.orig_time
            )
            raw_haemo.set_annotations(new_bulk_annots)
            st.rerun() # Instantly refresh the UI
    else:
        cont_bulk.info("No markers found.")

    # --- 7. MANUAL EDITOR (Logic runs here, UI renders at the bottom) ---
    # We must process the manual edits here so the changes cascade down 
    # to the Hemoglobin selection and Trimming algorithms below.
    with cont_manual.expander("⚙️ Advanced: Manual Marker Editor", expanded=False):
        if len(raw_haemo.annotations) > 0:
            ann_df = pd.DataFrame({
                'onset': raw_haemo.annotations.onset,
                'duration': raw_haemo.annotations.duration,
                'description': raw_haemo.annotations.description
            })
            
            edited_df = st.data_editor(ann_df, num_rows="dynamic", hide_index=True, use_container_width=True)
            
            edited_df = edited_df.dropna(subset=['onset', 'description'])
            new_annotations = mne.Annotations(
                onset=edited_df['onset'].values,
                duration=edited_df['duration'].values,
                description=edited_df['description'].values,
                orig_time=raw_haemo.annotations.orig_time # Maintain original anchoring
            )
            raw_haemo.set_annotations(new_annotations)
        else:
            st.info("No markers available to edit.")

    # --- 4. HEMOGLOBIN SELECTION ---
    cont_hemo.header("4. Hemoglobin Selection")
    hemo_type = cont_hemo.radio("View Signal Type", ["HbO (Oxygenated)", "HbR (Deoxygenated)", "HbTot (Total)"])
    
    if "HbO" in hemo_type:
        theme_color = "#D62728" 
        target_raw = raw_haemo.copy().pick(picks='hbo')
    elif "HbR" in hemo_type:
        theme_color = "#1F77B4" 
        target_raw = raw_haemo.copy().pick(picks='hbr')
    else:
        theme_color = "#2CA02C" 
        raw_hbo = raw_haemo.copy().pick(picks='hbo')
        raw_hbr = raw_haemo.copy().pick(picks='hbr')
        hbt_data = raw_hbo.get_data() + raw_hbr.get_data()
        
        info = raw_hbo.info.copy()
        mne.rename_channels(info, {ch: ch.replace('hbo', 'hbt') for ch in info.ch_names})
        target_raw = mne.io.RawArray(hbt_data, info, verbose=False)
        target_raw.set_annotations(raw_haemo.annotations)

    # --- 5. TRIMMING ---
    cont_trim.header("5. Data Trimming")
    max_time = float(target_raw.times[-1])
    trim_range = cont_trim.slider("Select Time Range (s)", min_value=0.0, max_value=max_time, value=(0.0, max_time), step=1.0)
    working_raw = target_raw.copy().crop(tmin=trim_range[0], tmax=trim_range[1])

    # --- 6. VIEW OPTIONS ---
    cont_view.header("6. View Options")
    ch_names = working_raw.ch_names
    channel_option = cont_view.selectbox("Select Channel to View", ["Grand Average"] + ch_names)
    overlay_raw = cont_view.checkbox("Overlay Raw Data", value=True)
    show_markers = cont_view.checkbox("Show Event Markers", value=True)

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
        psd_to_plot_db = 10 * np.log10(psd_filt)
    else:
        data_to_plot = data_raw
        psd_to_plot_db = psd_raw_db

    # --- BUILD THE PLOTS ---
    col1, col2 = st.columns(2) 

    with col1:
        st.subheader(f"Time Domain: {channel_option}")
        fig_time = go.Figure()
        
        if apply_filter and overlay_raw:
            fig_time.add_trace(go.Scatter(x=times, y=data_raw, mode='lines', name='Raw Signal', line=dict(color='lightgrey', width=1)))
            
        line_color = theme_color if apply_filter else 'lightgrey'
        line_name = f'Filtered Signal ({hemo_type.split(" ")[0]})' if apply_filter else 'Raw Signal'
        
        fig_time.add_trace(go.Scatter(x=times, y=data_to_plot, mode='lines', name=line_name, line=dict(color=line_color, width=2)))

        if show_markers and len(working_raw.annotations) > 0:
            unique_desc = list(set(working_raw.annotations.description))
            colors = ['#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#ff7f0e'] 
            color_map = {desc: colors[i % len(colors)] for i, desc in enumerate(unique_desc)}
            added_to_legend = set()

            for ann in raw_haemo.annotations:
                orig_onset = ann['onset']
                duration = ann['duration']
                desc = ann['description']
                
                if trim_range[0] <= orig_onset <= trim_range[1]:
                    aligned_onset = orig_onset - trim_range[0]
                    c = color_map[desc]
                    show_leg = desc not in added_to_legend
                    added_to_legend.add(desc)

                    if duration > 0:
                        fig_time.add_vrect(x0=aligned_onset, x1=aligned_onset+duration, fillcolor=c, opacity=0.15, line_width=0, layer="below")
                        if show_leg:
                            fig_time.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color=c, symbol='square', size=12), name=f"Event: {desc}"))
                    else:
                        fig_time.add_vline(x=aligned_onset, line_color=c, line_dash="dash", line_width=1.5, layer="below")
                        if show_leg:
                            fig_time.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color=c, dash='dash', width=2), name=f"Marker: {desc}"))

        fig_time.update_layout(xaxis_title="Time (s)", yaxis_title="Amplitude (µM)", template="plotly_white", legend=dict(x=0.01, y=0.99))
        st.plotly_chart(fig_time, use_container_width=True)

    with col2:
        st.subheader(f"Frequency Domain (PSD): {channel_option}")
        fig_psd = go.Figure()
        
        if apply_filter and overlay_raw:
            fig_psd.add_trace(go.Scatter(x=freqs_raw, y=psd_raw_db, mode='lines', name='Raw PSD', line=dict(color='lightgrey', width=1)))
            
        line_color_psd = theme_color if apply_filter else 'lightgrey'
        line_name_psd = f'Filtered PSD ({hemo_type.split(" ")[0]})' if apply_filter else 'Raw PSD'

        fig_psd.add_trace(go.Scatter(x=freqs_raw, y=psd_to_plot_db, mode='lines', name=line_name_psd, line=dict(color=line_color_psd, width=2)))
        
        fig_psd.add_vrect(x0=0.01, x1=0.08, fillcolor="blue", opacity=0.1, line_width=0, annotation_text="Neural Hemodynamics")
        fig_psd.add_vrect(x0=0.05, x1=0.15, fillcolor="orange", opacity=0.15, line_width=0, annotation_text="Mayer Waves")
        fig_psd.add_vrect(x0=0.2, x1=0.4, fillcolor="green", opacity=0.15, line_width=0, annotation_text="Respiration")
        fig_psd.add_vrect(x0=0.8, x1=1.5, fillcolor="red", opacity=0.15, line_width=0, annotation_text="Cardiac")
        
        fig_psd.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="Power (dB)", xaxis_range=[0, 2.0], template="plotly_white", legend=dict(x=0.80, y=0.99))
        st.plotly_chart(fig_psd, use_container_width=True)

    # --- EVENT-RELATED AVERAGE (EPOCHS) PLOT ---
    st.divider()
    st.subheader(f"Event-Related Average (ERA): {channel_option}")
    
    col_epoch_ui, col_epoch_plot = st.columns([1, 3])
    
    with col_epoch_ui:
        st.markdown("**Epoch Window Parameters**")
        tmin = st.number_input("Start Time (s) relative to pulse", value=-2.0, step=0.5)
        tmax = st.number_input("End Time (s) relative to pulse", value=15.0, step=1.0)
        
        if len(working_raw.annotations) > 0:
            unique_events = list(set(working_raw.annotations.description))
            selected_event = st.selectbox("Select Event to Average", unique_events)
        else:
            selected_event = None
            st.warning("No events available to epoch.")

    with col_epoch_plot:
        if selected_event is not None:
            # Create a 1-channel Raw object and explicitly name the channel
            safe_ch_name = channel_option.replace(" ", "_")
            epoch_info = mne.create_info(ch_names=[safe_ch_name], sfreq=fs, ch_types=['misc'])
            
            meas_date = working_raw.info.get('meas_date', None)
            if meas_date is not None:
                epoch_info.set_meas_date(meas_date)
            
            filtered_raw = mne.io.RawArray(np.atleast_2d(data_to_plot), epoch_info, verbose=False)
            
            # Create a new Annotations object without orig_time
            safe_annots = mne.Annotations(
                onset=working_raw.annotations.onset,
                duration=working_raw.annotations.duration,
                description=working_raw.annotations.description,
                orig_time=None
            )
            filtered_raw.set_annotations(safe_annots)
            
            events, event_dict = mne.events_from_annotations(filtered_raw, verbose=False)
            event_id = event_dict[selected_event]
            
            # Epoching: explicitly pick the channel by name to avoid ambiguity
            epochs = mne.Epochs(filtered_raw, events, event_id=event_id, 
                                tmin=tmin, tmax=tmax, baseline=(tmin, 0), 
                                picks=[safe_ch_name], preload=True, verbose=False)
