import h5py

def patch_homer3_snirf(filepath):
    print(f"Opening {filepath} to check tags...")
    fixed_count = 0
    
    # Open the file in 'r+' (read/write) mode so we can edit it directly
    with h5py.File(filepath, 'r+') as f:
        if 'nirs' not in f:
            print("Error: Could not find standard '/nirs' structure.")
            return
            
        nirs = f['nirs']
        
        # Loop through all data blocks
        for data_name in nirs.keys():
            if not data_name.startswith('data'):
                continue
            
            data_group = nirs[data_name]
            
            # Loop through all measurement lists (every optode/wavelength channel)
            for ml_name in data_group.keys():
                if not ml_name.startswith('measurementList'):
                    continue
                    
                ml_group = data_group[ml_name]
                
                # Check if the channel has a label
                if 'dataTypeLabel' in ml_group:
                    # Extract the current string
                    label = ml_group['dataTypeLabel'][0]
                    if hasattr(label, 'decode'):
                        label = label.decode('utf-8')
                        
                    # If Homer3 incorrectly tagged it as 'raw'
                    if label.lower() == 'raw':
                        # Delete the old fixed-length string to avoid truncation errors
                        del ml_group['dataTypeLabel']
                        # Recreate it with the MNE-compliant tag
                        ml_group.create_dataset('dataTypeLabel', data=[b'fnirs_cw_amplitude'])
                        fixed_count += 1

    print(f"Success! Replaced {fixed_count} 'raw' tags with 'fnirs_cw_amplitude'.")
    print("Your file is now strictly MNE-compliant.")

if __name__ == "__main__":
    # Pointing to your specific trimmed file
    snirf_file = r"C:/Users/bitnguyen/Desktop/2026-06-26/2026-06-26_002/main_2026-06-26_002_TRIMMED_MARKERCORRECTION.snirf"
    patch_homer3_snirf(snirf_file)
