# 🔊 Suspicious Sound Detection using UrbanSound8K

This project focuses on detecting suspicious or abnormal sounds in urban environments using machine learning models trained on the [UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html). It includes pre-trained models for audio classification based on common urban sound categories such as gunshots, sirens, and screams.

## 📁 Dataset

We use the **UrbanSound8K** dataset removed unecessary classes to be left with these 4 only:

- car_horn
- dog_bark
- gun_shot
- siren


You can download the dataset from the official website:  
👉 [UrbanSound8K Dataset](https://urbansounddataset.weebly.com/urbansound8k.html)

## 🧠 Pre-trained Models

Trained models are available in `.pkl` format via Google Drive. These models can be directly loaded for inference without retraining.

📥 **Download Models:**  
[Google Drive Link](https://drive.google.com/file/d/1Kf5mGRMlxuPZfNmwNBswqJU_ekwD62dG/view?usp=sharing)

> 💡 Note: These models were trained using log-Mel spectrogram features with classifiers such as CNN, ResNet, and hybrid architectures.

