from fastai.vision.all import PILImage
from PIL import Image
from fastai.vision.all import *
import librosa
import numpy as np
import sys
import pathlib

def load_and_preprocess_audio(file_path, sr=22050, duration=3):
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=duration)
        if len(y) < sr * duration:
            y = np.pad(y, (0, sr * duration - len(y)))
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
        return mel_spect_db
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def get_x(row):
    mel_spect = load_and_preprocess_audio(row.fname)
    if mel_spect is None or mel_spect.size == 0:
        print(f"Failed to process {row.fname}")
        return np.zeros((128, 128, 3), dtype=np.uint8)
    mel_spect = (mel_spect - mel_spect.min()) / (mel_spect.max() - mel_spect.min())
    mel_spect_3d = np.repeat(mel_spect[..., None], 3, axis=-1)
    mel_spect_uint8 = (mel_spect_3d * 255).astype(np.uint8)
    return mel_spect_uint8

def get_y(file_path):
    # Example: extract label from parent folder name
    return file_path.parent.name

def predict_audio(audio_path, model_path='audio_classifier.pkl'):
    # Apply StackOverflow fix for PosixPath/WindowsPath
    CLASS_NAMES = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music"
    }
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        learn = load_learner(model_path)
    finally:
        pathlib.PosixPath = temp
    # Preprocess audio
    mel_spect = load_and_preprocess_audio(audio_path)
    if mel_spect is None:
        return {"error": "Failed to preprocess audio"}
    # Normalize and convert to 3-channel uint8
    mel_spect = (mel_spect - mel_spect.min()) / (mel_spect.max() - mel_spect.min())
    mel_spect_3d = np.repeat(mel_spect[..., None], 3, axis=-1)
    mel_spect_uint8 = (mel_spect_3d * 255).astype(np.uint8)
    # Convert to PIL image (which is what fastai expects)
    pil_img = PILImage.create(Image.fromarray(mel_spect_uint8))
    # Predict
    pred_class, pred_idx, probs = learn.predict(pil_img)
    class_idx = int(pred_class)
    class_name = CLASS_NAMES.get(class_idx, str(pred_class))
    return {
        "predicted_class": class_name,
        "confidence": float(probs[pred_idx])
    }

# Example: set the path to your .wav file stored in Google Drive or local Colab path
"""audio_path = "C:\\Users\\moham\\Desktop\\testing audios\\7061-6-0-0.wav"

result = predict_audio(audio_path)
print("Prediction:", result)"""
