import streamlit as st

st.set_page_config(page_title="fNIRS Toolbox Home", layout="centered")

st.title("🧠 fNIRS Toolbox Development")
st.markdown("Welcome to your laboratory command center.")

# --- THE NOTEBOOK SECTION ---
st.divider()
st.subheader("📝 Developer Notebook")
st.markdown("""
Use this section to keep track of your progress and future ideas. 
You can edit this file directly in GitHub to update your roadmap.
""")

# You can use st.text_area to create a persistent text field, 
# or just write directly in Markdown here.
st.markdown("""
### Roadmap & Next Steps
- [ ] **Motion Correction:** Implement Spline Interpolation for motion artifacts.
- [ ] **Batch Processing:** Add a folder-upload feature for entire study cohorts.
- [ ] **Export:** Add PDF report generation for clinical collaborators.

### Log
- **2026-06-30:** Successfully deployed SOS filters and TMS marker editing.
- **2026-06-29:** Added ERA (Epoch) window and HbTot calculation.
""")

# --- NAVIGATION HELPER ---
st.divider()
st.info("Use the sidebar to jump to your active diagnostic and processing tools.")