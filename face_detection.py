import cv2
from ultralytics import YOLO
import os
#import insightface
import face_recognition
import numpy as np

# Parameters
output_dir = 'static/facesDatabase'
compare_dir = 'static\compareDatabase'
model_path = 'facial_detection.pt'
N_FRAMES_RECOGNITION = 10  # Only recognize faces every 10 frames

# Create the directory for saving the faces if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load your trained YOLOv8 model
model = YOLO(model_path)

# Initialize the InsightFace model for face embeddings
#app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
#app.prepare(ctx_id=0, det_size=(640, 640))  # Use GPU or CPU based on your setup

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Load known face encodings and embeddings
known_face_encodings = []
known_face_names = []
known_face_embeddings = []

for filename in os.listdir(compare_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(compare_dir, filename)
        image = cv2.imread(image_path)
        
        # Face_recognition method (older method)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        # Save encodings for recognition
        for encoding in encodings:
            known_face_encodings.append(encoding)
            known_face_names.append(filename.split('.')[0])

        # InsightFace method (optimized method) - COMMENTED OUT
        # faces = app.get(image)
        # if faces:
        #     known_face_embeddings.append(faces[0].embedding)
        #     known_face_names.append(filename.split('.')[0])


# Counters
face_counter = len(os.listdir(output_dir))  # Prevent overwriting
frame_counter = 0

"""def recognize_using_insightface(frame, app, known_face_embeddings, known_face_names):
    
    Recognize faces using InsightFace embeddings.
  
    face_recognized = "Unknown"
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    faces = app.get(rgb_frame)
    
    if faces:
        detected_face_embedding = faces[0].embedding
        
        # Compare embeddings
        face_distances = [np.linalg.norm(embedding - detected_face_embedding) for embedding in known_face_embeddings]
        
        if face_distances:
            best_match_index = np.argmin(face_distances)
            if face_distances[best_match_index] < 0.6:  # Adjust threshold as needed
                face_recognized = known_face_names[best_match_index]
                
    return face_recognized"""

def recognize_using_face_recognition(frame, known_face_encodings, known_face_names):
    """
    Recognize faces using the old face_recognition method.
    """
    face_recognized = "Unknown"
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    
    encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    for encoding in encodings:
        matches = face_recognition.compare_faces(known_face_encodings, encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, encoding)

        if True in matches:
            best_match_index = np.argmin(face_distances)
            if face_distances[best_match_index] < 0.6:  # Adjust threshold as needed
                face_recognized = known_face_names[best_match_index]
                
    return face_recognized

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1

    # Perform YOLO face detection
    results = model(frame)
    detected_faces = results[0].boxes

    for bbox in detected_faces:
        x1, y1, x2, y2 = bbox.xyxy[0]
        conf = bbox.conf[0]
        cls = int(bbox.cls[0])
        label = model.names[cls]
        color = (0, 255, 0)

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Crop the face
        face_image = frame[y1:y2, x1:x2]

        # Save each cropped face
        if face_image.size > 0:
            face_filename = os.path.join(output_dir, f"face_{face_counter}.jpg")
            cv2.imwrite(face_filename, face_image)
            face_counter += 1

        name_to_display = "Unknown"

        # Only run recognition every N_FRAMES_RECOGNITION frames
        if frame_counter % N_FRAMES_RECOGNITION == 0 and face_image.size > 0:
            try:
                # Use InsightFace recognition
                #name_to_display = recognize_using_insightface(frame, app, known_face_embeddings, known_face_names)
                
                # Alternatively, use face_recognition method
                name_to_display = recognize_using_face_recognition(frame, known_face_encodings, known_face_names)
                
            except Exception as e:
                print(f"Recognition error: {e}")

        # Display the name + detection confidence
        text = f'{name_to_display} | {conf:.2f}'
        cv2.putText(frame, text, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if name_to_display != "Unknown" else (0, 0, 255), 2)

    cv2.imshow('Real-time Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
