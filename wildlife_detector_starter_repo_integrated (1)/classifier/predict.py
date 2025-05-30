import os
import numpy as np
import librosa
import joblib

# Path to trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "audio_species_model.pkl")

# Load trained classifier once at module level
try:
    clf = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    clf = None

def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        features = np.hstack([
            np.mean(mfccs, axis=1),
            zcr,
            centroid
        ])
        return features
    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        return None

def predict(audio_path):
    if clf is None:
        return "Error: Model not loaded."

    try:
        features = extract_features(audio_path)
        if features is None:
            return "Error during feature extraction."

        prediction = clf.predict([features])[0]
        return f"Predicted: {prediction}"

    except Exception as e:
        return f"Error during prediction: {str(e)}"
