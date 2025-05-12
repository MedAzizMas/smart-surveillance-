# Updated code with GPU handling
import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import torch

# Load both models
PLATE_DETECTOR_PATH = 'tunisian_lp_detector.pt'  # Update this
CHAR_DETECTOR_PATH = 'lp_recognition.pt'

plate_detector = YOLO(PLATE_DETECTOR_PATH)
char_detector = YOLO(CHAR_DETECTOR_PATH)

def sort_boxes_left_to_right(boxes, texts, scores):
    """Sort detected characters from left to right"""
    if len(boxes) == 0:
        return [], []

    char_positions = []
    for box, text, score in zip(boxes, texts, scores):
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        char_positions.append([x_center, y_center, text, score])

    char_positions.sort(key=lambda x: x[0])
    return [char[2] for char in char_positions], [char[3] for char in char_positions]

def process_image(image_path, plate_conf=0.5, char_conf=0.25):
    """Process a single image and return detected license plate numbers"""
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Could not read image"

    # Detect license plates
    plate_results = plate_detector(img, conf=plate_conf)[0]

    all_plates = []

    # Process each detected license plate
    for plate_box in plate_results.boxes.data:
        # Move tensor to CPU and convert to numpy
        if plate_box.is_cuda:
            plate_box = plate_box.cpu()
        plate_box = plate_box.numpy()

        x1, y1, x2, y2, conf, _ = plate_box
        if conf < plate_conf:
            continue

        # Crop license plate
        plate_img = img[int(y1):int(y2), int(x1):int(x2)]

        # Detect characters in the plate
        char_results = char_detector(plate_img, conf=char_conf)[0]

        # Extract character detections
        boxes = []
        texts = []
        scores = []

        for char_box in char_results.boxes.data:
            # Move tensor to CPU and convert to numpy
            if char_box.is_cuda:
                char_box = char_box.cpu()
            char_box = char_box.numpy()

            x1, y1, x2, y2, conf, cls = char_box
            if conf < char_conf:
                continue
            boxes.append([x1, y1, x2, y2])
            texts.append(char_detector.names[int(cls)])
            scores.append(conf)

        # Sort characters left to right
        sorted_texts, sorted_scores = sort_boxes_left_to_right(boxes, texts, scores)

        # Combine characters into plate number
        plate_number = ''.join(sorted_texts)
        avg_confidence = np.mean(sorted_scores) if sorted_scores else 0

        all_plates.append((plate_number, avg_confidence))

    return all_plates

def visualize_results(image_path, results):
    """Visualize the detection results"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 10))
    plt.imshow(img)

    # Add text with results
    y_pos = 50
    for plate_number, confidence in results:
        plt.text(10, y_pos, f"Plate: {plate_number} (Conf: {confidence:.2f})",
                color='white', backgroundcolor='black', fontsize=12)
        y_pos += 30

    plt.title("License Plate Detection Results")
    plt.axis('off')
    plt.show()

def process_directory(test_dir, max_images=5):
    """Process all images in a directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    image_files = [f for f in os.listdir(test_dir)
                  if os.path.splitext(f.lower())[1] in image_extensions]

    for img_file in image_files[:max_images]:
        img_path = os.path.join(test_dir, img_file)
        print(f"\nProcessing: {img_file}")

        try:
            results = process_image(img_path)

            print("Detected License Plates:")
            for plate_number, confidence in results:
                print(f"- {plate_number} (Confidence: {confidence:.2f})")

            visualize_results(img_path, results)

        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")



# Add video processing functions for license plate detection and recognition

def process_frame(frame, plate_conf=0.5, char_conf=0.25):
    """Process a single frame and return detected license plate numbers and their bounding boxes"""
    plate_results = plate_detector(frame, conf=plate_conf)[0]
    all_plates = []
    plate_boxes = []
    for plate_box in plate_results.boxes.data:
        if plate_box.is_cuda:
            plate_box = plate_box.cpu()
        plate_box = plate_box.numpy()
        x1, y1, x2, y2, conf, _ = plate_box
        if conf < plate_conf:
            continue
        plate_img = frame[int(y1):int(y2), int(x1):int(x2)]
        char_results = char_detector(plate_img, conf=char_conf)[0]
        boxes = []
        texts = []
        scores = []
        for char_box in char_results.boxes.data:
            if char_box.is_cuda:
                char_box = char_box.cpu()
            char_box = char_box.numpy()
            cx1, cy1, cx2, cy2, cconf, cls = char_box
            if cconf < char_conf:
                continue
            boxes.append([cx1, cy1, cx2, cy2])
            texts.append(char_detector.names[int(cls)])
            scores.append(cconf)
        sorted_texts, sorted_scores = sort_boxes_left_to_right(boxes, texts, scores)
        plate_number = ''.join(sorted_texts)
        avg_confidence = np.mean(sorted_scores) if sorted_scores else 0
        all_plates.append((plate_number, avg_confidence))
        plate_boxes.append((int(x1), int(y1), int(x2), int(y2), plate_number, avg_confidence))
    return all_plates, plate_boxes

def process_video(video_path, output_path=None, plate_conf=0.5, char_conf=0.25, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_count = 0
    freeze_plate = None
    freeze_counter = 0
    freeze_duration = int(2 * fps)  # Freeze for 2 seconds
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        plates, plate_boxes = process_frame(frame, plate_conf, char_conf)
        # Draw bounding boxes and plate numbers
        high_conf_plate = None
        for (x1, y1, x2, y2, plate_number, confidence) in plate_boxes:
            color = (0, 255, 0) if confidence > 0.85 else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{plate_number} ({confidence:.2f})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if confidence > 0.85:
                high_conf_plate = (plate_number, confidence)
        # Freeze logic: if high confidence, freeze for freeze_duration frames
        if high_conf_plate:
            freeze_plate = high_conf_plate
            freeze_counter = freeze_duration
        elif freeze_counter > 0:
            freeze_counter -= 1
        # Display frozen plate if needed
        if freeze_plate and freeze_counter > 0:
            cv2.putText(frame, f"FROZEN: {freeze_plate[0]} ({freeze_plate[1]:.2f})", (30, height-30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        # Show the frame
        # cv2.imshow('License Plate Detection', frame)  # Removed for web use
        if out:
            out.write(frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        frame_count += 1
        if max_frames and frame_count >= max_frames:
            break
    cap.release()
    if out:
        out.release()
    # cv2.destroyAllWindows()  # Removed for web use

# Example usage for video processing:
# Uncomment the following line to test on a video file:
#process_video("C:\\Users\\moham\\Downloads\\video_test_nour.mp4", output_path='output_video_nour.mp4', max_frames=300)