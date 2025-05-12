import os
import cv2
import numpy as np
import torch
import tensorflow as tf
from PIL import Image
import torchvision.transforms as transforms
from gait_recognition.video_to_frames import extract_frames
from gait_recognition.segment_frames import process_image, preprocess_single_image, tta_predict, ensemble_predict
from gait_recognition.model_architecture import CSTL



def get_prediction_with_confidence(model, sequence,device):
    model.eval()
    with torch.no_grad():
        sequence = sequence.to(device)
        outputs = model(sequence)

        # Get probabilities using softmax
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Get prediction and confidence
        confidence, predicted = torch.max(probabilities, 1)

        # Get top 3 predictions
        top3_conf, top3_pred = torch.topk(probabilities, 3)

        return predicted.item(), confidence.item(), top3_pred[0].tolist(), top3_conf[0].tolist()

def load_gait_model(model_path):
    """Load the trained gait recognition model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model file
    model_data = torch.load(model_path, map_location=device)
    
    # Check if the loaded data is a dictionary (state dict) or a model
    if isinstance(model_data, dict):
        # If it's a dictionary, it's likely a state dict
        if 'model_state_dict' in model_data:
            # Create a model instance and load the state dict
            model = CSTL(num_classes=125)
            model.load_state_dict(model_data['model_state_dict'])
        else:
            # If it's just a state dict without the 'model_state_dict' key
            model = CSTL(num_classes=125)
            model.load_state_dict(model_data)
    else:
        # If it's already a model
        model = model_data
    
    model.eval()
    return model, device

def load_segmentation_model(model_path):
    """Load the trained segmentation model"""
    model = tf.keras.models.load_model(model_path, custom_objects={'iou_metric': lambda y_true, y_pred: tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))})
    return model

def prepare_sequence_for_gait(segmented_frames, transform=None):
    """Prepare a sequence of segmented frames for the gait recognition model"""
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    sequence = []
    for frame in segmented_frames:
        # Convert numpy array to PIL Image
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)
        
        # Apply transforms
        frame = transform(frame)
        sequence.append(frame)
    
    # Stack frames into a tensor
    sequence_tensor = torch.stack(sequence)
    return sequence_tensor

def process_video_pipeline(video_path, segmentation_model_path, gait_model_path, output_dir='gait_recognition/output'):
    """
    Complete pipeline for gait recognition from video
    
    Args:
        video_path: Path to the input video
        segmentation_model_path: Path to the segmentation model
        gait_model_path: Path to the gait recognition model
        output_dir: Directory to save intermediate results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    segmented_dir = os.path.join(output_dir, "segmented")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(segmented_dir, exist_ok=True)
    
    # Step 1: Extract frames from video
    print("Step 1: Extracting frames from video...")
    extract_frames(video_path, frames_dir)
    
    # Step 2: Load models
    print("Step 2: Loading models...")
    print("  - Loading gait recognition model...")
    gait_model, device = load_gait_model(gait_model_path)
    print("  - Gait recognition model loaded successfully")
    
    # Step 3: Segment frames
    print("Step 3: Segmenting frames...")
    segmented_frames = []
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    
    print(f"  - Processing {len(frame_files)} frames")
    
    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frames_dir, frame_file)
        print(f"  - Processing frame {i+1}/{len(frame_files)}: {frame_file}")
        
        # Process the frame with segmentation model
        segmented_frame = process_image(frame_path, segmentation_model_path, save_results=False)
        
        if segmented_frame is not None:
            # Convert to numpy array
            segmented_frame = tf.squeeze(segmented_frame).numpy() * 255
            segmented_frame = segmented_frame.astype(np.uint8)
            
            # Save segmented frame
            segmented_path = os.path.join(segmented_dir, f"segmented_{frame_file}")
            cv2.imwrite(segmented_path, segmented_frame)
            
            segmented_frames.append(segmented_frame)

            # Check if we have enough frames for prediction
            if len(segmented_frames) == 16:
                print("  - Found 16 segmented frames, stopping segmentation and starting prediction...")
                break  # Stop segmentation

        else:
            print(f"  - Warning: Failed to segment frame {frame_file}")

    # Proceed to prediction with the segmented frames
    if len(segmented_frames) < 16:
        print(f"  - Warning: Only found {len(segmented_frames)} segmented frames. Prediction will proceed with available frames.")

    # Prepare sequence for model
    sequence_tensor = prepare_sequence_for_gait(segmented_frames)
    sequence_tensor = sequence_tensor.unsqueeze(0).to(device)  # Add batch dimension
    
    # Get prediction and confidence
    predicted, confidence, top3_pred, top3_conf = get_prediction_with_confidence(gait_model, sequence_tensor, device)

    # Return or store the prediction result as needed
    return predicted, confidence  # Adjust as necessary for your use case

if __name__ == "__main__":
    import argparse
    print(torch.__version__)
    parser = argparse.ArgumentParser(description="Gait Recognition Pipeline")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--segmentation_model", required=True, help="Path to segmentation model")
    parser.add_argument("--gait_model", required=True, help="Path to gait recognition model")
    parser.add_argument("--output_dir", default="output", help="Directory to save intermediate results")
    
    args = parser.parse_args()
    
    person_id, confidence = process_video_pipeline(
        args.video, 
        args.segmentation_model, 
        args.gait_model, 
        args.output_dir
    ) 