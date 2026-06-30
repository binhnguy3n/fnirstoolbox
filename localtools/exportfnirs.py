import os
import mne
import matplotlib.pyplot as plt

def export_snirf_graphs(snirf_path, output_dir="snirf_exports", trim_end_seconds=None):
    """
    Loads a .snirf file, standardizes markers, trims the end, calculates Concentration 
    Changes, and exports ultra-wide SVG and High-Res PNG graphs.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading {snirf_path}...")
    
    # 1. Load raw intensity data
    raw = mne.io.read_raw_snirf(snirf_path, preload=True)
    
    # --- Marker Standardization: Set all durations to 0.2s ---
    if len(raw.annotations) > 0:
        print(f"Found {len(raw.annotations)} markers. Setting all durations to 0.2 seconds...")
        new_annotations = mne.Annotations(
            onset=raw.annotations.onset,
            duration=0.2,
            description=raw.annotations.description
        )
        raw.set_annotations(new_annotations)
    
    # --- Trimming: Chop off the END of the file ---
    if trim_end_seconds:
        total_duration = raw.times[-1]
        new_tmax = total_duration - trim_end_seconds
        
        if new_tmax <= 0:
            print(f"Error: You are trying to trim {trim_end_seconds}s, but the file is only {total_duration}s long!")
            return
            
        print(f"Original duration: {total_duration:.1f}s. Trimming the last {trim_end_seconds}s...")
        raw.crop(tmin=0, tmax=new_tmax)
        
    # --- Map Distances for Short Channel Naming ---
    distances = mne.preprocessing.nirs.source_detector_distances(raw.info)
    sd_distance_map = {}
    for ch_name, dist in zip(raw.ch_names, distances):
        sd = ch_name.split(' ')[0] 
        sd_distance_map[sd] = dist
    
    # 2. Convert to Optical Density (OD)
    raw_od = mne.preprocessing.nirs.optical_density(raw)
    
    # 3. Convert OD to Concentration Changes (CC)
    raw_cc = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)
    
    # Extract unique source-detector pairs
    ch_names = raw.ch_names
    sd_pairs = sorted(list(set([name.split(' ')[0] for name in ch_names if ' ' in name])))
    
    print(f"Found {len(sd_pairs)} optode pairs. Generating and saving SVG & PNG graphs...")

    # 4. Generate plots per SD pair
    for sd in sd_pairs:
        # --- NEW: Ultra-wide aspect ratio (30 inches wide by 8 inches tall) ---
        fig, axes = plt.subplots(2, 1, figsize=(30, 8), sharex=True)
        
        # --- Top Panel: Plot Raw Intensity ---
        raw_picks = [i for i, ch in enumerate(raw.ch_names) if ch.startswith(f"{sd} ")]
        if raw_picks:
            times = raw.times
            data_raw, _ = raw[raw_picks, :]
            for i, pick in enumerate(raw_picks):
                ch_name = raw.ch_names[pick]
                
                # Match Satori RAW colors (785nm = Red, 830nm = Blue)
                if '785' in ch_name:
                    color = '#D62728' 
                elif '830' in ch_name:
                    color = '#1F77B4'
                else:
                    color = None 
                    
                axes[0].plot(times, data_raw[i].T, label=ch_name, color=color)
            
            axes[0].set_title(f"Raw Intensity - {sd}")
            axes[0].set_ylabel("Intensity (V)")
            axes[0].legend(loc="upper right")
            axes[0].grid(True, alpha=0.3)

        # --- Bottom Panel: Plot Concentration Changes (CC) ---
        cc_picks = [i for i, ch in enumerate(raw_cc.ch_names) if ch.startswith(f"{sd} ")]
        if cc_picks:
            times_cc = raw_cc.times
            data_cc, _ = raw_cc[cc_picks, :]
            for i, pick in enumerate(cc_picks):
                ch_name = raw_cc.ch_names[pick]
                
                # Match Satori CC colors (HbO = Red, HbR = Blue)
                color = '#D62728' if 'hbo' in ch_name.lower() else '#1F77B4' 
                axes[1].plot(times_cc, data_cc[i].T * 1e6, label=ch_name, color=color)
                
            axes[1].set_title(f"Concentration Changes (HbO / HbR) - {sd}")
            axes[1].set_ylabel("Δ Concentration (μM)")
            axes[1].set_xlabel("Time (s)")
            axes[1].legend(loc="upper right")
            axes[1].grid(True, alpha=0.3)

        # --- Plot Markers ---
        if len(raw.annotations) > 0:
            for annot in raw.annotations:
                onset = annot['onset']
                axes[0].axvline(x=onset, color='gray', linestyle='--', alpha=0.5)
                axes[1].axvline(x=onset, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        
        # --- 5. Export as SVG AND High-Res PNG ---
        is_short = sd_distance_map.get(sd, 1.0) < 0.015
        suffix = "_SHORT" if is_short else ""
        
        # Save SVG
        save_path_svg = os.path.join(output_dir, f"{sd}{suffix}_raw_cc.svg")
        plt.savefig(save_path_svg, format='svg', bbox_inches='tight')
        
        # Save PNG (1000 DPI for extreme zoom clarity)
        save_path_png = os.path.join(output_dir, f"{sd}{suffix}_raw_cc.png")
        plt.savefig(save_path_png, dpi=1000, bbox_inches='tight')
        
        plt.close(fig) 
        
    print(f"Success! Exported {len(sd_pairs) * 2} files to the '{output_dir}' directory.")

# --- How to run ---
if __name__ == "__main__":
    # Point this to the original Borealis file
    snirf_file = r"C:/Users/bitnguyen/Desktop/2026-06-26/2026-06-26_002/2026-06-26_002.snirf" 
    
    # Trims the LAST 175 seconds off the file
    trim_time = 175 
    
    export_snirf_graphs(snirf_file, output_dir="exported_optode_graphs", trim_end_seconds=trim_time)
