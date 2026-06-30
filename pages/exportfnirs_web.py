import streamlit as st
import mne
import matplotlib.pyplot as plt
import os
import tempfile
import zipfile
import io

# --- PAGE SETUP ---
st.set_page_config(layout="wide", page_title="fNIRS Image Exporter")
st.title("fNIRS Batch Image Exporter")
st.markdown("Upload a `.snirf` file to automatically generate and download ultra-wide Raw Intensity and Concentration Change graphs for every optode pair.")

# --- SIDEBAR UI ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload .snirf", type=["snirf"])

st.sidebar.header("2. Export Settings")
trim_end = st.sidebar.number_input("Trim end of recording (seconds)", min_value=0.0, value=0.0, step=1.0)
dpi_setting = st.sidebar.slider("PNG Resolution (DPI)", min_value=100, max_value=1000, value=300, step=100, 
                                help="Warning: High DPI on Ultra-Wide graphs consumes massive RAM. Lower this if the cloud app crashes.")
export_svg = st.sidebar.checkbox("Export SVGs (Vector)", value=True)
export_png = st.sidebar.checkbox("Export PNGs (Raster)", value=True)

# --- PROCESSING LOGIC ---
if st.sidebar.button("Generate Graphs"):
    if uploaded_file is None:
        st.sidebar.error("Please upload a .snirf file first.")
    elif not export_svg and not export_png:
        st.sidebar.error("Please select at least one export format (SVG or PNG).")
    else:
        with st.spinner("Processing data and generating graphs... This may take a minute."):
            
            # 1. Create a master temporary directory to hold everything safely
            with tempfile.TemporaryDirectory() as temp_dir:
                
                # Save the uploaded file to the temp directory so MNE can read it
                snirf_path = os.path.join(temp_dir, "uploaded_data.snirf")
                with open(snirf_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Load the raw data
                raw = mne.io.read_raw_snirf(snirf_path, preload=True)
                
                # --- Marker Standardization ---
                if len(raw.annotations) > 0:
                    new_annotations = mne.Annotations(
                        onset=raw.annotations.onset,
                        duration=0.2,
                        description=raw.annotations.description
                    )
                    raw.set_annotations(new_annotations)
                
                # --- Trimming ---
                if trim_end > 0:
                    total_duration = raw.times[-1]
                    new_tmax = total_duration - trim_end
                    if new_tmax <= 0:
                        st.error(f"Error: You are trying to trim {trim_end}s, but the file is only {total_duration:.1f}s long!")
                        st.stop()
                    raw.crop(tmin=0, tmax=new_tmax)
                
                # --- Map Distances for Short Channel Naming ---
                distances = mne.preprocessing.nirs.source_detector_distances(raw.info)
                sd_distance_map = {ch_name.split(' ')[0]: dist for ch_name, dist in zip(raw.ch_names, distances)}
                
                # Conversions
                raw_od = mne.preprocessing.nirs.optical_density(raw)
                raw_cc = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)
                
                # Extract unique SD pairs
                ch_names = raw.ch_names
                sd_pairs = sorted(list(set([name.split(' ')[0] for name in ch_names if ' ' in name])))
                
                # Create a subfolder for just the images
                image_dir = os.path.join(temp_dir, "graphs")
                os.makedirs(image_dir, exist_ok=True)
                
                # --- Generate Plots ---
                progress_bar = st.progress(0)
                for index, sd in enumerate(sd_pairs):
                    fig, axes = plt.subplots(2, 1, figsize=(30, 8), sharex=True)
                    
                    # Top Panel: Raw Intensity
                    raw_picks = [i for i, ch in enumerate(raw.ch_names) if ch.startswith(f"{sd} ")]
                    if raw_picks:
                        times = raw.times
                        data_raw, _ = raw[raw_picks, :]
                        for i, pick in enumerate(raw_picks):
                            ch_name = raw.ch_names[pick]
                            color = '#D62728' if '785' in ch_name else '#1F77B4' if '830' in ch_name else None 
                            axes[0].plot(times, data_raw[i].T, label=ch_name, color=color)
                        axes[0].set_title(f"Raw Intensity - {sd}")
                        axes[0].set_ylabel("Intensity (V)")
                        axes[0].legend(loc="upper right")
                        axes[0].grid(True, alpha=0.3)

                    # Bottom Panel: CC
                    cc_picks = [i for i, ch in enumerate(raw_cc.ch_names) if ch.startswith(f"{sd} ")]
                    if cc_picks:
                        times_cc = raw_cc.times
                        data_cc, _ = raw_cc[cc_picks, :]
                        for i, pick in enumerate(cc_picks):
                            ch_name = raw_cc.ch_names[pick]
                            color = '#D62728' if 'hbo' in ch_name.lower() else '#1F77B4' 
                            axes[1].plot(times_cc, data_cc[i].T * 1e6, label=ch_name, color=color)
                        axes[1].set_title(f"Concentration Changes (HbO / HbR) - {sd}")
                        axes[1].set_ylabel("Δ Concentration (μM)")
                        axes[1].set_xlabel("Time (s)")
                        axes[1].legend(loc="upper right")
                        axes[1].grid(True, alpha=0.3)

                    # Markers
                    if len(raw.annotations) > 0:
                        for annot in raw.annotations:
                            axes[0].axvline(x=annot['onset'], color='gray', linestyle='--', alpha=0.5)
                            axes[1].axvline(x=annot['onset'], color='gray', linestyle='--', alpha=0.5)

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

                # --- 2. Zip the images into memory ---
                memory_zip = io.BytesIO()
                with zipfile.ZipFile(memory_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for root, _, files in os.walk(image_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            zf.write(file_path, arcname=file)
                
                memory_zip.seek(0)
                
                # Store the ZIP file in Streamlit's session state so the download button persists
                st.session_state['zip_data'] = memory_zip
                st.session_state['file_count'] = len(os.listdir(image_dir))
                st.success("Graphs generated successfully!")

# --- DOWNLOAD BUTTON ---
if 'zip_data' in st.session_state:
    st.markdown("### Export Complete")
    st.write(f"Generated {st.session_state['file_count']} image files.")
    
    st.download_button(
        label="Download ZIP Archive",
        data=st.session_state['zip_data'],
        file_name="fnirs_graphs.zip",
        mime="application/zip",
        type="primary"
    )