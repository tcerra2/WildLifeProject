import sys
import os
# 👇 This points to the base folder where "classifier" exists
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tempfile
import streamlit as st
from classifier import predict as cp

# ... other imports remain the same

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
if uploaded_file is not None:
    # Save to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Display audio
    st.audio(tmp_path, format='audio/wav')

    # Run Detection
    st.subheader("Running Detection...")
    result = cp.predict(tmp_path)
    st.subheader("Predicted Species")
    st.write(result)

