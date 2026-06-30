import streamlit as st
import mne
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import tempfile
import zipfile
import io

# --- PAGE SETUP ---
st.set_page_config(layout="wide", page_title="fNIRS Image Exporter")
st.title("fNIRS Batch Image Exporter")
st.markdown("Preview your data, trim out artifacts, and automatically generate ultra-wide `.svg` and `.png` reports for every optode pair.")

# --- SIDEBAR UI: UPLOAD ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload .snirf", type=["snirf"])

# --- LOAD DATA FUNCTION ---
@st.cache_data
def load_snirf_data(file_data):
    if file_data is None:
        return None
    try:
        # Create a persistent temporary file for MNE to read and crop safely
        with tempfile.NamedTemporaryFile(delete=False, suffix='.snirf') as tmp_file:
            tmp_file.write(file_data.getvalue())
            tmp_path = tmp_file.name
            
        raw = mne.io.read_raw_snirf(tmp_path, preload=True)
        return raw, tmp_path
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None, None

# --- MAIN APP LOGIC ---
if uploaded_file is None:
    st.info("Please drag and drop a .snirf file into the sidebar to begin.")
else:
    raw, tmp_path = load_snirf_data(uploaded_file)
    
    if raw is not None:
        # --- 2. INTERACTIVE PREVIEW & TRIMMING ---
        st.header("Step 1: Preview & Trim Data")
        
        max_time = float(raw.times[-1])
        ch_names = raw.ch_names
        
        col_controls, col_empty = st.columns([1, 2])
        with col_controls:
            preview_channel = st.selectbox("Select Channel to Preview", ch_names)
            
            st.markdown("**Select exact start and end times to crop:**")
            trim_range = st.slider("Time Range (s)", 
                                   min_value=0.0, 
                                   max_value=max_time, 
                                   value=(0.0, max_time), 
                                   step=1.0)

        # Plotly Preview of the specific cropped region
        working_raw = raw.copy().crop(tmin=trim_range[0], tmax=trim_range[1])
        ch_idx = ch_names.index(preview_channel)
        preview_data = working_raw.get_data()[ch_idx, :]
        preview_times = working_raw.times
        
        fig_preview = go.Figure()
        fig_preview.add_trace(go.Scatter(x=preview_times, y=preview_data, mode='lines', 
                                         name=preview_channel, line=dict(color='#1f77b4', width=1)))
        
        # --- FIX: MANUAL MARKER ALIGNMENT FOR PREVIEW ---
        if len(raw.annotations) > 0:
            for ann in raw.annotations:
                orig_onset = ann['onset']
                # Only draw the marker if it falls inside our new trimmed window
                if trim_range[0] <= orig_onset <= trim_range[1]:
                    # Shift the marker's position to match the new 0-based time axis
                    aligned_x = orig_onset - trim_range[0]
                    fig_preview.add_vline(x=aligned_x, line_color='gray', line_dash="dash", line_width=1.5, layer="below")
                
        fig_preview.update_layout(title=f"Previewing: {preview_channel} (Trimmed to {trim_range[1] - trim_range[0]:.1f}s)",
                                  xaxis_title="Time (s)", yaxis_title="Amplitude (V)", 
                                  height=400, template="plotly_white", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_preview, use_container_width=True)


        # --- 3. EXPORT SETTINGS & GENERATION ---
        st.divider()
        st.header("Step 2: Generate Bulk Reports")
        
        st.sidebar.header("2. Export Settings")
        dpi_setting = st.sidebar.slider("PNG Resolution (DPI)", min_value=100, max_value=1000, value=300, step=100, 
                                        help="Warning: 1000 DPI creates massive files. Lower this if the cloud app crashes.")
        export_svg = st.sidebar.checkbox("Export SVGs (Vector)", value=True)
        export_png = st.sidebar.checkbox("Export PNGs (Raster)", value=True)

        if st.button("Generate Graphs", type="primary"):
            if not export_svg and not export_png:
                st.error("Please select at least one export format in the sidebar (SVG or PNG).")
            else:
                with st.spinner(f"Applying trims and generating {len(set([n.split(' ')[0] for n in ch_names if ' ' in n])) * 2} graphs..."):
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Apply the chosen trim limits to the main dataset
                        export_raw = raw.copy().crop(tmin=trim_range[0], tmax=trim_range[1])
                        
                        # Conversions
                        distances = mne.preprocessing.nirs.source_detector_distances(export_raw.info)
                        sd_distance_map = {ch.split(' ')[0]: dist for ch, dist in zip(export_raw.ch_names, distances)}
                        
                        raw_od = mne.preprocessing.nirs.optical_density(export_raw)
                        raw_cc = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)
                        
                        sd_pairs = sorted(list(set([name.split(' ')[0] for name in export_raw.ch_names if ' ' in name])))
                        
                        image_dir = os.path.join(temp_dir, "graphs")
                        os.makedirs(image_dir, exist_ok=True)
                        
                        # --- Generate Plots ---
                        progress_bar = st.progress(0)
                        for index, sd in enumerate(sd_pairs):
                            fig, axes = plt.subplots(2, 1, figsize=(30, 8), sharex=True)
                            
                            # Top Panel: Raw Intensity
                            raw_picks = [i for i, ch in enumerate(export_raw.ch_names) if ch.startswith(f"{sd} ")]
                            if raw_picks:
                                times = export_raw.times
                                data_raw, _ = export_raw[raw_picks, :]
                                for i, pick in enumerate(raw_picks):
                                    ch_name = export_raw.ch_names[pick]
                                    color = '#D62728' if '785' in ch_name else '#1F77B4' if '830' in ch_name else None 
                                    axes[0].plot(times, data_raw[i].T, label=ch_name, color=color)
                                axes[0].set_title(f"Raw Intensity - {sd}")
                                axes[0].set_ylabel("Intensity (V)")
                                axes[0].legend(loc="upper right")
                                axes[0].grid(True, alpha=0.3)

                            # Bottom Panel: Concentration Changes
                            cc_picks = [i for i, ch in enumerate(raw_cc.ch_names) if ch.startswith(f"{sd} ")]
                            if cc_picks:
                                times_cc = raw_cc.times
                                data_cc, _ = raw_cc[cc_picks, :]
                                for i, pick in enumerate(cc_picks):
                                    ch_name = raw_cc.ch_names[pick]
                                    color = '#D62728' if 'hbo' in ch_name.lower() else '#1F77B4' 
                                    axes[1].plot(times_cc, data_cc[i].T * 1e6, label=ch_name, color=color)
                                axes[1].set_title(f"Concentration Changes (HbO / HbR) - {sd}")
                                axes[1].set_ylabel("Δ Concentration (µM)")
                                axes[1].set_xlabel("Time (s)")
                                axes[1].legend(loc="upper right")
                                axes[1].grid(True, alpha=0.3)

                            # --- FIX: MANUAL MARKER ALIGNMENT FOR EXPORT ---
                            if len(raw.annotations) > 0:
                                for annot in raw.annotations:
                                    orig_onset = annot['onset']
                                    if trim_range[0] <= orig_onset <= trim_range[1]:
                                        aligned_x = orig_onset - trim_range[0]
                                        axes[0].axvline(x=aligned_x, color='gray', linestyle='--', alpha=0.5)
                                        axes[1].axvline(x=aligned_x, color='gray', linestyle='--', alpha=0.5)

                            plt.tight_layout()
                            
                            # Save formats
                            is_short = sd_distance_map.get(sd, 1.0) < 0.015
                            suffix = "_SHORT" if is_short else ""
                            
                            if export_svg:
                                plt.savefig(os.path.join(image_dir, f"{sd}{suffix}_raw_cc.svg"), format='svg', bbox_inches='tight')
                            if export_png:
                                plt.savefig(os.path.join(image_dir, f"{sd}{suffix}_raw_cc.png"), dpi=dpi_setting, bbox_inches='tight')
                            
                            plt.close(fig) # Free memory
                            progress_bar.progress((index + 1) / len(sd_pairs))

                        # --- Zip the images into memory ---
                        memory_zip = io.BytesIO()
                        with zipfile.ZipFile(memory_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                            for root, _, files in os.walk(image_dir):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    zf.write(file_path, arcname=file)
                        
                        memory_zip.seek(0)
                        
                        st.session_state['export_zip'] = memory_zip
                        st.session_state['export_count'] = len(os.listdir(image_dir))
                        st.success("Graphs generated successfully! Click below to download.")

        # --- DOWNLOAD BUTTON ---
        if 'export_zip' in st.session_state:
            st.download_button(
                label=f"Download {st.session_state['export_count']} Exported Files (ZIP)",
                data=st.session_state['export_zip'],
                file_name="fnirs_graphs.zip",
                mime="application/zip",
                type="primary"
            )
