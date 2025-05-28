import sys
import os

# Add real full path
sys.path.append(r"C:\Users\tcerr\Documents\wildlifeproject\wildlife_detector_starter_repo_integrated (1)")

import streamlit as st
import soundfile as sf
from core import signal_model as sm
from classifier import predict as cp




st.title("Wildlife Signal Detection App")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
if uploaded_file is not None:
    audio_data, samplerate = sf.read(uploaded_file)
    st.audio(uploaded_file, format='audio/wav')

    st.subheader("Running Detection...")
    components = sm.run_model(audio_data, samplerate)
    st.write(f"Detected {len(components)} signal points.")

    result = cp.predict(components)
    st.subheader("Predicted Species")
    st.write(result)
