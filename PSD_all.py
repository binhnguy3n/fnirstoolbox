import streamlit as st
import mne
import numpy as np
from scipy import signal
import plotly.graph_objects as go
import tempfile
import os

# --- PAGE SETUP ---
st.set_page_config(layout="wide", page_title="fNIRS PSD Diagnostic")
st.title("fNIRS Signal Quality Diagnostic")
st.markdown("Upload a `.snirf` file to evaluate channel quality across the frequency spectrum before processing.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Drop a .snirf file here", type=["snirf"])

# --- LOAD DATA (Cached for speed) ---
@st.cache_data
def load_and_prep_data(file_data):
    if file_data is None:
        return None
        
    try:
        # Create a temporary file on the hard drive for MNE to read
        with tempfile.NamedTemporaryFile(delete=False, suffix='.snirf') as tmp_file:
            tmp_file.write(file_data.getvalue())
            tmp_path = tmp_file.name
            
        # Load the data and compute Hemoglobin concentration
        raw = mne.io.read_raw_snirf(tmp_path, preload=True)
        raw_od = mne.preprocessing.nirs.optical_density(raw)
        raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)
        
        # Isolate Oxyhemoglobin for visualization
        raw_hbo = raw_haemo.copy().pick(picks='hbo')
        
        # Delete the temporary file
        os.remove(tmp_path)
        
        return raw_hbo
        
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

raw_hbo = load_and_prep_data(uploaded_file)

if raw_hbo is None:
    st.info("Please drag and drop a .snirf file into the sidebar to begin.")
else:
    # ==========================================
    # EXTRACT & COMPUTE PSD
    # ==========================================
    data = raw_hbo.get_data() 
    fs = raw_hbo.info['sfreq']
    ch_names = raw_hbo.ch_names 

    # Compute PSD for all channels simultaneously
    frequencies, psd = signal.welch(data, fs, nperseg=1024, axis=1)
    psd_db = 10 * np.log10(psd)

    # Calculate mean and standard deviation for the "Average" view
    mean_psd = np.mean(psd_db, axis=0)
    std_psd = np.std(psd_db, axis=0)

    # ==========================================
    # BUILD INTERACTIVE PLOTLY FIGURE
    # ==========================================
    fig = go.Figure()
    n_channels = len(ch_names)

    # --- Traces 0 & 1: Variance Shadow ---
    upper_bound = mean_psd + std_psd
    lower_bound = mean_psd - std_psd

    fig.add_trace(go.Scatter(x=frequencies, y=upper_bound, mode='lines', 
                             line=dict(width=0), showlegend=False, visible=True))
    fig.add_trace(go.Scatter(x=frequencies, y=lower_bound, mode='lines', 
                             line=dict(width=0), fill='tonexty', fillcolor='rgba(128, 128, 128, 0.4)', 
                             name='Variance', visible=True))

    # --- Trace 2: Mean PSD Line ---
    fig.add_trace(go.Scatter(x=frequencies, y=mean_psd, mode='lines', 
                             line=dict(color='black', width=3), name='Average HbO', visible=True))

    # --- Traces 3 to N: Individual Channels ---
    for i, ch_name in enumerate(ch_names):
        fig.add_trace(go.Scatter(x=frequencies, y=psd_db[i], mode='lines', 
                                 name=ch_name, line=dict(width=1), opacity=0.6, visible=False)) 

    # ==========================================
    # BUILD THE DROPDOWN MENU
    # ==========================================
    buttons = []

    # View 1: "Grand Average" 
    visible_avg = [True, True, True] + [False] * n_channels
    buttons.append(
        dict(label="1. Grand Average (Mean ± SD)",
             method="update",
             args=[{"visible": visible_avg},
                   {"title": "Average HbO Power Spectral Density"}])
    )

    # View 2: "All Channels (Spaghetti Plot)"
    visible_all_overlay = [False, False, False] + [True] * n_channels
    buttons.append(
        dict(label="2. All Channels (Spaghetti Plot)",
             method="update",
             args=[{"visible": visible_all_overlay},
                   {"title": "All Channels Power Spectral Density"}])
    )

    # Views 3+: Individual Channels
    for i, ch_name in enumerate(ch_names):
        visible_single = [False, False, False] + [False] * n_channels
        visible_single[3 + i] = True 
        
        buttons.append(
            dict(label=f"Channel: {ch_name}",
                 method="update",
                 args=[{"visible": visible_single},
                       {"title": f"Power Spectral Density: {ch_name}"}])
        )

    # ==========================================
    # FORMATTING & ARTIFACT BANDS
    # ==========================================
    fig.update_layout(
        updatemenus=[dict(active=0, buttons=buttons, x=1.0, y=1.15, xanchor='right')],
        xaxis_range=[0, 2.0],
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power (dB)",
        template="plotly_white",
        height=600 # Makes the chart nice and large in the browser
    )

    # Add physiological artifact bands
    fig.add_vrect(x0=0.05, x1=0.15, fillcolor="orange", opacity=0.15, layer="below", line_width=0, annotation_text="Mayer Waves")
    fig.add_vrect(x0=0.2, x1=0.4, fillcolor="green", opacity=0.15, layer="below", line_width=0, annotation_text="Respiration")
    fig.add_vrect(x0=0.8, x1=1.5, fillcolor="red", opacity=0.15, layer="below", line_width=0, annotation_text="Cardiac")

    # Render the interactive plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)