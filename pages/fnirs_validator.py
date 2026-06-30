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
    st.sidebar.header("3. Marker Editor")
    if len(raw_haemo.annotations) > 0:
        
        # --- NEW: BULK UPDATE TOOL ---
        st.sidebar.markdown("**Bulk Update All Markers**")
        col_bulk1, col_bulk2 = st.sidebar.columns(2)
        bulk_desc = col_bulk1.text_input("Name", value="TMS")
        bulk_dur = col_bulk2.number_input("Duration (s)", value=0.2, step=0.1)
        
        if st.sidebar.button("Apply Bulk Update"):
            new_bulk_annots = mne.Annotations(
                onset=raw_haemo.annotations.onset,
                duration=[bulk_dur] * len(raw_haemo.annotations),
                description=[bulk_desc] * len(raw_haemo.annotations),
                orig_time=raw_haemo.annotations.orig_time
            )
            raw_haemo.set_annotations(new_bulk_annots)
            st.rerun() # Instantly refresh the UI
            
        st.sidebar.divider()
            
        # --- MANUAL ROW EDITOR ---
        st.sidebar.markdown("**Manual Row Editor**")
        ann_df = pd.DataFrame({
            'onset': raw_haemo.annotations.onset,
            'duration': raw_haemo.annotations.duration,
            'description': raw_haemo.annotations.description
        })
        
        edited_df = st.sidebar.data_editor(ann_df, num_rows="dynamic", hide_index=True, use_container_width=True)
        
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

    freqs_raw, psd_raw = signal.welch(data_raw, fs, nperseg=1024)
    psd_raw_db = 10 * np.log10(psd_raw)

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
            # FIX: Safely create a 1-channel Raw object and handle metadata timekeeping perfectly
            safe_ch_name = channel_option.replace(" ", "_")
            epoch_info = mne.create_info(ch_names=[safe_ch_name], sfreq=fs, ch_types=['misc'])
            
            # Sync the measurement date to prevent "Ambiguous operation" errors
            meas_date = working_raw.info.get('meas_date', None)
if meas_date is not None:
    epoch_info.set_meas_date(meas_date)
            
            filtered_raw = mne.io.RawArray(np.atleast_2d(data_to_plot), epoch_info, verbose=False)
            
            # Strip orig_time from the copied annotations to ensure perfect relative alignment
            safe_annots = working_raw.annotations.copy()
            safe_annots.orig_time = None
            filtered_raw.set_annotations(safe_annots)
            
            events, event_dict = mne.events_from_annotations(filtered_raw, verbose=False)
            event_id = event_dict[selected_event]
            
            try:
                epochs = mne.Epochs(filtered_raw, events, event_id=event_id, 
                                    tmin=tmin, tmax=tmax, baseline=(tmin, 0), preload=True, verbose=False)
                
                if len(epochs) > 0:
                    evoked = epochs.average()
                    fig_era = go.Figure()
                    
                    # Individual Trials
                    for i in range(len(epochs)):
                        fig_era.add_trace(go.Scatter(x=epochs.times, y=epochs.get_data()[i, 0, :], 
                                                     mode='lines', line=dict(color='lightgray', width=1), 
                                                     opacity=0.3, showlegend=False, hoverinfo='skip'))
                    
                    # Grand Average
                    fig_era.add_trace(go.Scatter(x=evoked.times, y=evoked.data[0], mode='lines', 
                                                 name=f'Average {hemo_type.split(" ")[0]} Response', 
                                                 line=dict(color=theme_color, width=3)))
                    
                    # Pulse Marker
                    fig_era.add_vline(x=0, line_color='black', line_dash='dash', annotation_text="TMS Pulse")
                    
                    fig_era.update_layout(xaxis_title="Time relative to pulse (s)", yaxis_title="Amplitude (µM)",
                                          template="plotly_white", title=f"Averaged Response to '{selected_event}' (n={len(epochs)} pulses)")
                    st.plotly_chart(fig_era, use_container_width=True)
                else:
                    st.info("No events found within the current trimmed time range.")
                    
            except Exception as e:
                st.error(f"Could not calculate Epochs. Check your time window. Error: {e}")
