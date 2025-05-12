import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from gait_recognition.metrics import iou_metric
def adaptive_enhance_image(img):
    # Analyze image characteristics
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # Initial denoising with adaptive strength
    if contrast > 50:
        img_denoised = cv2.fastNlMeansDenoisingColored(img, None, 15, 15, 7, 21)
    else:
        img_denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    # Adjust parameters based on image characteristics
    if brightness < 100:  # Dark image
        clahe_limit = 3.0
        alpha = 1.3
        beta = 15
    elif brightness > 200:  # Bright image
        clahe_limit = 2.0
        alpha = 0.9
        beta = -5
    else:  # Normal image
        clahe_limit = 2.5
        alpha = 1.1
        beta = 5
    
    # Convert to LAB and enhance
    lab = cv2.cvtColor(img_denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE with adaptive parameters
    clahe = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge channels
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Apply bilateral filter with adaptive parameters
    if contrast > 50:
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    else:
        enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
    
    # Adjust contrast and brightness
    enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
    
    # Convert to grayscale for additional processing
    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    
    # Second CLAHE pass with mild settings
    clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    enhanced_gray = clahe2.apply(gray)
    
    # Blend with color image
    enhanced_color = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
    enhanced = cv2.addWeighted(enhanced, 0.4, enhanced_color, 0.6, 0)
    
    # Final gentle bilateral filter
    enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
    
    return enhanced

def preprocess_single_image(image_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Store original for visualization
    original_img = img.copy()
    
    # Apply adaptive enhancement
    enhanced = adaptive_enhance_image(img)
    
    # Resize for model input (preserve aspect ratio)
    target_size = (128, 128)
    h, w = enhanced.shape[:2]
    aspect = w/h
    
    if aspect > 1:
        new_w = target_size[0]
        new_h = int(new_w/aspect)
    else:
        new_h = target_size[1]
        new_w = int(new_h*aspect)
    
    # Use LANCZOS interpolation for better quality downscaling
    enhanced_resized = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Create blank target size image
    model_input = np.zeros((target_size[0], target_size[1], 3))
    
    # Calculate center offset
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    
    # Place resized image in center
    model_input[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = enhanced_resized
    
    # Normalize to [0,1] with improved scaling
    model_input = (model_input - model_input.min()) / (model_input.max() - model_input.min() + 1e-8)
    
    return original_img, enhanced, model_input

def tta_predict(model, image_input):
    predictions = []
    
    # Original image
    pred = model.predict(np.expand_dims(image_input, axis=0))[0]
    predictions.append(pred)
    
    # Flipped image
    flipped = tf.image.flip_left_right(image_input)
    pred_flip = model.predict(np.expand_dims(flipped, axis=0))[0]
    pred_flip = tf.image.flip_left_right(pred_flip)
    predictions.append(pred_flip)
    
    # Slightly brighter
    brighter = tf.clip_by_value(image_input * 1.2, 0, 1)
    pred_bright = model.predict(np.expand_dims(brighter, axis=0))[0]
    predictions.append(pred_bright)
    
    # Slightly darker
    darker = tf.clip_by_value(image_input * 0.8, 0, 1)
    pred_dark = model.predict(np.expand_dims(darker, axis=0))[0]
    predictions.append(pred_dark)
    
    # Average all predictions
    avg_prediction = np.mean(predictions, axis=0)
    return avg_prediction

def ensemble_predict(model_dir, image_input, num_models=5):
    predictions = []
    # Use models from different epochs
    for epoch in range(17, 22):  # Using epochs 17-21
        model_path = f'{model_dir}/model_epoch_{epoch:02d}.h5'
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path, custom_objects={'iou_metric': iou_metric})
            pred = model.predict(np.expand_dims(image_input, axis=0))[0]
            predictions.append(pred)
    
    if not predictions:
        # If no checkpoints found, use the main model
        print("No model checkpoints found, using the main model instead")
        model_path = model_dir  # The model_dir is actually the model path in this case
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path, custom_objects={'iou_metric': iou_metric})
            pred = model.predict(np.expand_dims(image_input, axis=0))[0]
            predictions.append(pred)
        else:
            raise ValueError(f"Model file not found: {model_path}")
    
    # Average predictions
    avg_prediction = np.mean(predictions, axis=0)
    return avg_prediction

def visualize_prediction(original_img, preprocessed_img, predicted_mask, save_path=None):
    plt.figure(figsize=(15, 5))
    
    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis("off")
    
    # Preprocessed Image
    plt.subplot(1, 3, 2)
    plt.imshow(preprocessed_img)
    plt.title("Enhanced Image")
    plt.axis("off")
    
    # Predicted Mask
    plt.subplot(1, 3, 3)
    plt.imshow(tf.squeeze(predicted_mask), cmap='gray')
    plt.title("Predicted Mask")
    plt.axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def process_image(image_path, model_path, save_results=True):
    try:
        print(f"  - Loading model from {model_path}")
        # Load the model
        model = tf.keras.models.load_model(model_path, custom_objects={'iou_metric': iou_metric})
        
        print(f"  - Preprocessing image {image_path}")
        # Preprocess image
        original_img, enhanced_img, model_input = preprocess_single_image(image_path)
        
        # Get predictions using both TTA and ensemble
        print(f"  - Running TTA prediction")
        tta_pred = tta_predict(model, model_input)
        
        # For ensemble_predict, we need to check if model_path is a directory or a file
        print(f"  - Running ensemble prediction")
        if os.path.isdir(model_path):
            ensemble_pred = ensemble_predict(model_path, model_input)
        else:
            # If model_path is a file, use it directly
            ensemble_pred = ensemble_predict(model_path, model_input)
        
        # Combine predictions and apply threshold
        print(f"  - Combining predictions")
        final_pred = (tta_pred + ensemble_pred) / 2.0
        final_pred = tf.cast(final_pred > 0.45, tf.float32)
        
        # Visualize results
        if save_results:
            print(f"  - Visualizing results")
            visualize_prediction(original_img, enhanced_img, final_pred)
            
            # Create results directory if it doesn't exist
            save_dir = os.path.join(os.path.dirname(image_path), 'results')
            os.makedirs(save_dir, exist_ok=True)
            
            # Save enhanced image and prediction
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            cv2.imwrite(os.path.join(save_dir, f'{base_name}_enhanced.jpg'),
                       cv2.cvtColor((enhanced_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                       [cv2.IMWRITE_JPEG_QUALITY, 100])
            cv2.imwrite(os.path.join(save_dir, f'{base_name}_mask.png'),
                       (tf.squeeze(final_pred).numpy() * 255).astype(np.uint8),
                       [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
        print(f"  - Frame processing complete")
        return final_pred
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None
