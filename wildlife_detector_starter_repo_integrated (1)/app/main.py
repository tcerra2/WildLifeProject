import sys
import os
import tempfile
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Add classifier path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from classifier import predict as cp

# Streamlit UI
st.title("Wildlife Audio Classifier")
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load audio for playback and plots
    y, sr = librosa.load(tmp_path, sr=None)

    # 🎧 Audio playback
    st.audio(tmp_path, format="audio/wav")

    # 📈 Waveform
    st.subheader("Waveform")
    fig_wave, ax_wave = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax_wave)
    ax_wave.set_title("Waveform")
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Amplitude")
    st.pyplot(fig_wave)

    # 🎛️ Spectrogram
    st.subheader("Spectrogram")
    fig_spec, ax_spec = plt.subplots()
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax_spec)
    fig_spec.colorbar(img, ax=ax_spec, format="%+2.0f dB")
    ax_spec.set_title("Mel-frequency spectrogram")
    st.pyplot(fig_spec)

    # 🧠 Run Prediction
    st.subheader("Running Detection...")
    result = cp.predict(tmp_path)
    st.subheader("Predicted Species")
    st.write(result)
