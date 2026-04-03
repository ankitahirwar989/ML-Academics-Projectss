"""
🎙️ Indian Languages Audio Classifier - Local Prediction Script
================================================================
Usage:
    python predict_language.py <audio_file>

Example:
    python predict_language.py sample_hindi.mp3
    python predict_language.py "C:\path\to\audio.wav"

Requirements:
    - indian_language_classifier.keras (saved model)
    - label_mapping.json (class labels)
    - Both files should be in the same folder as this script
"""

import sys
import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import librosa
from tensorflow.keras.models import load_model


# ---- Configuration ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'indian_language_classifier.keras')
LABELS_PATH = os.path.join(SCRIPT_DIR, 'label_mapping.json')
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_LEN = 216


def extract_mfcc(file_path):
    """Extract MFCC features from an audio file."""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=5)
    except Exception:
        # Try with pydub for MP3 files
        from pydub import AudioSegment
        import io
        audio = AudioSegment.from_file(file_path)
        buffer = io.BytesIO()
        audio.export(buffer, format='wav')
        buffer.seek(0)
        y, sr = librosa.load(buffer, sr=SAMPLE_RATE, duration=5)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    # Pad or truncate to fixed length
    if mfccs.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :MAX_LEN]

    return mfccs


def predict(audio_path):
    """Load model and predict the language of an audio file."""

    # Check files exist
    if not os.path.exists(audio_path):
        print(f'❌ Audio file not found: {audio_path}')
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        # Also check for .h5 format
        h5_path = MODEL_PATH.replace('.keras', '.h5')
        if os.path.exists(h5_path):
            model_path = h5_path
        else:
            print(f'❌ Model file not found!')
            print(f'   Expected: {MODEL_PATH}')
            print(f'   Place the model file in: {SCRIPT_DIR}')
            sys.exit(1)
    else:
        model_path = MODEL_PATH

    if not os.path.exists(LABELS_PATH):
        print(f'❌ Label mapping not found: {LABELS_PATH}')
        print(f'   Place label_mapping.json in: {SCRIPT_DIR}')
        sys.exit(1)

    # Load model
    print('🔄 Loading model...')
    model = load_model(model_path)
    print('✅ Model loaded!')

    # Load labels
    with open(LABELS_PATH, 'r') as f:
        label_data = json.load(f)
    idx_to_label = label_data['index_to_label']

    # Extract features
    print(f'🎵 Processing: {audio_path}')
    mfcc_features = extract_mfcc(audio_path)

    # Reshape: (1, timesteps, n_mfcc)
    X_input = mfcc_features.T[np.newaxis, :, :]

    # Predict
    prediction = model.predict(X_input, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] * 100
    predicted_language = idx_to_label[str(predicted_class)]

    # Display results
    print()
    print('=' * 50)
    print('🎯 PREDICTION RESULT')
    print('=' * 50)
    print(f'   Audio File:   {os.path.basename(audio_path)}')
    print(f'   Language:     {predicted_language}')
    print(f'   Confidence:   {confidence:.2f}%')
    print('=' * 50)

    # Show all probabilities
    print('\n📊 All Language Probabilities:')
    probs = [(idx_to_label[str(i)], prediction[0][i] * 100)
             for i in range(len(idx_to_label))]
    probs.sort(key=lambda x: x[1], reverse=True)

    for lang, prob in probs:
        bar = '█' * int(prob / 2) + '░' * (50 - int(prob / 2))
        marker = ' ◄' if lang == predicted_language else ''
        print(f'   {lang:<12} {bar} {prob:6.2f}%{marker}')

    return predicted_language, confidence


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        print('❌ Please provide an audio file path!')
        print(f'\n   python {os.path.basename(__file__)} <audio_file.mp3>')
        sys.exit(1)

    audio_file = sys.argv[1]
    predict(audio_file)
