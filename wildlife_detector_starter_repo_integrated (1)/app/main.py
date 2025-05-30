import sys
import os

# Add classifier directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tempfile
import streamlit as st
from classifier import predict as cp

# Streamlit UI
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Display audio playback
    st.audio(tmp_path, format='audio/wav')

    # Run Detection
    st.subheader("Running Detection...")
    result = cp.predict(tmp_path)
    st.subheader("Predicted Species")
    st.write(result)
