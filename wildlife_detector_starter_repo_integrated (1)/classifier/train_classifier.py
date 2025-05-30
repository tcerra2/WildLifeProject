# Train species classification model

import os
import librosa
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set correct path to your local audio folder
AUDIO_DIR = r"C:\Users\tcerr\Documents\wildlifeproject\wildlife_detector_starter_repo_integrated (1)\audio_samples\opensoundscape-master\tests\audio"

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        return np.hstack([mfccs, zcr, centroid])
    except Exception as e:
        print(f"❌ Failed to extract from {file_path}: {e}")
        return None

def assign_label(filename):
    name = os.path.splitext(os.path.basename(filename).lower())[0]
    if "great_plains_toad" in name:
        return "Great Plains Toad"
    elif name.startswith("aru"):
        return "ARU Recorder Test"
    elif name.startswith("loca"):
        return "Field Sample"
    elif "silence" in name or "empty" in name:
        return "Silence"
    elif "rugr_drum" in name:
        return "Red-winged Blackbird (Drum)"
    elif name.startswith("msd"):
        return "Cornell Bird Sample"
    elif "short" in name or "veryshort" in name:
        return "Short Clip"
    elif "1min" in name or "stereo" in name:
        return "General Test Sample"
    elif "audiomoth" in name or "metadata" in name:
        return "Metadata (Ignore)"
    else:
        return "Unknown"

def train():
    print("🔍 Scanning audio files...")
    features = []
    labels = []

    for file in os.listdir(AUDIO_DIR):
        if file.endswith(".wav"):
            file_path = os.path.join(AUDIO_DIR, file)
            feat = extract_features(file_path)
            if feat is not None:
                features.append(feat)
                labels.append(assign_label(file))
                print(f"✅ Using: {file}")

    if not features:
        print("❌ No valid training data found.")
        return

    # Train a basic classifier
    print("🧠 Training model...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print(f"✅ Model trained. Accuracy on test set: {accuracy:.2f}")

    # Save the model
    model_path = "classifier/audio_species_model.pkl"
    joblib.dump(clf, model_path)
    print(f"📦 Model saved to: {model_path}")

if __name__ == "__main__":
    train()
