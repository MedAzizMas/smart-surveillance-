# Gait Recognition Pipeline Components

This branch contains the core components for video-based gait recognition, including human segmentation and temporal gait analysis models.

## 📁 Datasets
1. **Human Body Segmentation**  
   - 2,667 high-quality segmented person images  
   - Professionally annotated masks  
   - Filtered to remove low-quality markup  
   https://www.kaggle.com/datasets/tapakah68/supervisely-filtered-segmentation-person-dataset

3. **Gait Recognition**  
   Modified CASIA-B dataset extended with custom subject (#125)  
   - Contains walking sequences from multiple angles  
   - Original dataset + custom subject data 

## Model Components

### 1. Human Body Segmentation (`Human_body_segmentation..ipynb`)
- U-Net architecture for frame-by-frame body segmentation
- Processes 128×128 RGB frames → 128×128 binary masks
- Features:  
  ✔ Data augmentation (flips, rotation)  
  ✔ Batch training with 16 samples  
  ✔ Model checkpointing each epoch

### 2. Gait Recognition (`CSTL.ipynb`)
- CNN-Spatial Temporal Learning (CSTL) model
- Processes sequences of 16 segmented frames (64×64)
- Key features:  
  ✔ Multi-Scale Temporal Encoding  
  ✔ Adaptive Temporal Attention  
  ✔ OneCycleLR scheduling  
  ✔ Gradient accumulation (3 steps)

## Model Availability
Pre-trained models available via Google Drive:
- Segmentation U-Net weights (.h5)
https://drive.google.com/file/d/1-s_qZMzLWDuYEXfqkKOkaR-0MP9iuOva/view?usp=drive_link
- CSTL model weights (.pth)
https://drive.google.com/file/d/11Hr298zZ4hAj_x9SEh5VeVwhICZpeyEw/view?usp=sharing

**Note:** These components integrate with the main project's video processing pipeline. See main branch for complete system implementation.
