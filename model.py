import tensorflow as tf
import librosa
import numpy as np
from scipy.ndimage import zoom
N_MFCC = 40
MODEL_PATH = "deepfake_detector_final.h5"  # Adjust if located elsewhere

SAMPLE_RATE = 16000
DURATION = 5
N_MELS = 128
MAX_TIME_STEPS = 109

def load_model():
    return tf.keras.models.load_model(MODEL_PATH)
    

def predict_audio(file_path):
    model = load_model()
    
    # Load and process audio file
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

    # Generate MFCC features (same as in training)
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.vstack([mfcc, delta, delta2])

    # Pad or trim the spectrogram to match fixed length (120)
    if combined.shape[1] < 120:
        combined = np.pad(combined, ((0, 0), (0, 120 - combined.shape[1])), mode='constant')
    else:
        combined = combined[:, :120]

    # Normalize the features
    combined = (combined - np.mean(combined)) / (np.std(combined) + 1e-8)

    # Expand dims to match model input: (1, 120, 120, 1)
    input_data = np.expand_dims(combined, axis=[0, -1])

    # Get prediction
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)
    
    # Determine label
    label = "Bonafide!!" if predicted_class == 1 else "Spoof!!"

    return label
