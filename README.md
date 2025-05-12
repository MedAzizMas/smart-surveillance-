# AI-Powered Security System Overview

This project implements an integrated intelligent surveillance system. The system combines six AI-powered modules to provide comprehensive security monitoring: facial recognition, license plate recognition, OCR for identity cards, suspicious sound detection, suspicious object detection, and silhouette/gait recognition. The solution addresses limitations of conventional security systems by providing automated, real-time threat detection and prevention capabilities.

## Features

- **Multi-modal Surveillance**: Combines visual, acoustic, and behavioral analysis
- **Real-time Processing**: All modules operate with low latency for immediate threat response
- **Comprehensive Threat Detection**:
  - Facial recognition with deepfake detection (89.9% accuracy)
  - Tunisian license plate recognition (99.4% accuracy)
  - ID card verification with OCR (97.2% accuracy)
  - Suspicious sound detection (99% accuracy)
  - Weapon and dangerous object detection (92.8% mAP)
  - Gait-based identification (97.33% accuracy)
- **Modular Architecture**: Components can be deployed independently or integrated
- **Explainable AI**: Provides interpretable detection results for operator review

## Tech Stack

### AI Components
- **Computer Vision**: 
  - YOLOv8, ArcFace, Xception
  - Human Body Segmentation: UNet
  - Gait Recognition: CSTL
- **Audio Processing**: ResNet34, Mel-spectrograms
- **OCR**: EasyOCR, PaddleOCR
- **Deep Learning Frameworks**: PyTorch, TensorFlow

### Backend
- **API Services**: FastAPI, Flask
- **Database**: PostgreSQL (for identity records), Redis (caching)
- **Message Broker**: RabbitMQ (for module communication)
- **Vector Database**: FAISS (for facial embeddings)

### Frontend (Monitoring Interface)
- Flask

## Acknowledgments

We acknowledge the contributions of:
- The open-source community for providing foundational models and frameworks
- Dataset providers (CASIA-B, WIDER FACE, UrbanSound8K)
- Research teams whose published work informed our approach
- Our academic advisors for their guidance
