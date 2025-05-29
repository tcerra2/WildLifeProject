import streamlit as st

import librosa
import numpy as np

def predict(audio_path):
    try:
        # Load the audio
        y, sr = librosa.load(audio_path, sr=None)

        # Extract features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

        # Basic thresholds (these are just sample logic)
        if centroid < 2000 and zcr < 0.1:
            return "Species A"
        elif centroid < 4000:
            return "Species B"
        else:
            return "Species C"

    except Exception as e:
        return f"Error during prediction: {str(e)}"

