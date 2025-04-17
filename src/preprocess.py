import os
import librosa
import numpy as np
from pydub import AudioSegment

def extract_features_with_librosa(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)

    # Extract Mel-frequency cepstral coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)  # Mean of MFCCs for the entire audio

    # Extract other features such as chroma, spectral contrast, etc.
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1)

    # Return all features as a flattened array
    return np.hstack([mfccs_mean, chroma_mean, spectral_contrast_mean])

def extract_features_with_pydub(audio_path):
    # Load audio using pydub
    audio = AudioSegment.from_wav(audio_path)
    samples = np.array(audio.get_array_of_samples())

    # Simple feature: Calculate mean and variance of the audio samples
    mean = np.mean(samples)
    variance = np.var(samples)

    return [mean, variance]
