from flask import Flask, render_template, url_for, request, redirect, Response, jsonify
from ultralytics import YOLO
import os
import uuid
import cv2
import datetime
import easyocr
import cv2
import numpy as np
import re

from PIL import Image
app = Flask(__name__, static_url_path='/static')


from predict import load_and_preprocess_audio, predict_audio, get_x, get_y
from lp import process_video
#ranim
model_path = 'ranim.pt'  
detection_model = YOLO(model_path)
reader_arabic = easyocr.Reader(['ar'])
reader_french = easyocr.Reader(['fr'])
##############

# Folder configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DETECTION_FOLDER'] = 'static/detections'

# Suspicious objects list
SUSPICIOUS_OBJECTS = {'Pistol', 'Grenade', 'Knife', 'RPG', 'Machine_Guns', 'Masked_Face', 'Bat'}

# Global model variable (start with suspicious model)
current_model = YOLO('static/best.pt')

# Global variable for real-time detection
detected_classes = set()

# --- Basic Pages ---
@app.route('/')
@app.route('/index.html')
def home():
    return render_template('index.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/service.html')
def service():
    return render_template('service.html')

@app.route('/project.html')
def project():
    return render_template('project.html')

@app.route('/feature.html')
def feature():
    return render_template('feature.html')

@app.route('/quote.html')
def quote():
    return render_template('quote.html')

@app.route('/team.html')
def team():
    return render_template('team.html')

@app.route('/testimonial.html')
def testimonial():
    return render_template('testimonial.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/face.html')
def face():
    return render_template('face.html')

@app.route('/ocr.html')
def ocr():
    return render_template('ocr.html')


##################################################### ala ################################################################################

############################################### New Part ################################################################

from flask import request
import subprocess
# Endpoint to start facial detection
@app.route('/start-detection', methods=['POST'])
def start_detection():
    try:
        # Run the Python script
        subprocess.Popen(["python", r"C:\Users\kills\Desktop\integration\integration\face_detection.py"])
        return '', 200  # success
    except Exception as e:
        print(e)
        return '', 500  # failure

@app.route('/switch_model/<model_type>')
def switch_model(model_type):
    global current_model
    if model_type == 'normal':
        current_model = YOLO('static/yolov8n.pt')
    elif model_type == 'suspicious':
        current_model = YOLO('static/best.pt')

    previous_page = request.referrer  # URL précédente
    if previous_page:
        return redirect(previous_page)
    else:
        return redirect(url_for('detection_type'))

# Detection Type Selection
@app.route('/detection_type')
def detection_type():
    return render_template('detection_type.html')

# Upload image page
@app.route('/upload_image')
def upload_image():
    return render_template('detectobj.html', detected_img=None, object_counts={}, alert=False)

# Detect objects in image
@app.route('/detectobj', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            file.save(filepath)

            results = current_model.predict(source=filepath, save=True, project='static/detections', name='results', exist_ok=True)
            result = results[0]

            result_image_path = result.save_dir + '/' + os.path.basename(filepath)
            relative_path = result_image_path.split('static\\')[-1].replace("\\", "/")

            labels = result.names
            boxes = result.boxes
            object_classes = [labels[int(cls)] for cls in boxes.cls]
            object_counts = {}

            for obj in object_classes:
                object_counts[obj] = object_counts.get(obj, 0) + 1

            alert = any(obj in SUSPICIOUS_OBJECTS for obj in object_classes)
            if alert:
                with open('static/alerts/alert_history.txt', 'a') as f:
                    for obj in object_classes:
                        if obj in SUSPICIOUS_OBJECTS:
                            f.write(f"[{datetime.datetime.now()}] | {obj} | ALERT: YES\n")
            else:
                with open('static/alerts/alert_history.txt', 'a') as f:
                    f.write(f"[{datetime.datetime.now()}] | No suspicious object | ALERT: NO\n")

            return render_template('detectobj.html', detected_img=relative_path, object_counts=object_counts, alert=alert)

    return render_template('detectobj.html', detected_img=None, object_counts={}, alert=False)

# Upload video page
@app.route('/upload_video')
def upload_video():
    return render_template('upload_video.html', detected_video=None, object_names=[], alert=False)

# Process uploaded video
@app.route('/upload_video_actual', methods=['POST'])
def upload_video_actual():
    if 'file' not in request.files:
        return redirect(url_for('upload_video'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('upload_video'))

    if file:
        filename = str(uuid.uuid4()) + ".mp4"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(os.path.dirname(upload_path), exist_ok=True)
        file.save(upload_path)

        cap = cv2.VideoCapture(upload_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None:
            fps = 25.0

        output_filename = 'detected_' + filename
        output_path = os.path.join(app.config['DETECTION_FOLDER'], output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        object_names = set()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = current_model.predict(frame, verbose=False)[0]

            if results.boxes:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    label = current_model.names[cls_id]

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    object_names.add(label)

            out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        detected_video_path = os.path.join('detections', output_filename).replace("\\", "/")

        alert = any(obj in SUSPICIOUS_OBJECTS for obj in object_names)
        if alert:
            with open('static/alerts/alert_history.txt', 'a') as f:
                for obj in object_names:
                    if obj in SUSPICIOUS_OBJECTS:
                        f.write(f"[{datetime.datetime.now()}] | {obj} | ALERT: YES\n")
        else:
            with open('static/alerts/alert_history.txt', 'a') as f:
                f.write(f"[{datetime.datetime.now()}] | No suspicious object | ALERT: NO\n")

        return render_template('upload_video.html', detected_video=detected_video_path, object_names=object_names, alert=alert)

# Start real-time detection
@app.route('/start_camera')
def start_camera():
    return render_template('real_time.html')

def gen_frames():
    global detected_classes
    detected_classes.clear()

    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 25.0

    output_path = os.path.join(app.config['DETECTION_FOLDER'], 'recorded_live.mp4')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = current_model.predict(frame, verbose=False)[0]

        if results.boxes:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = current_model.names[cls_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                detected_classes.add(label)

        out.write(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    out.release()
    cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_detected_classes')
def get_detected_classes():
    global detected_classes
    return jsonify(list(detected_classes))

@app.route('/alert_status')
def alert_status():
    global detected_classes
    alert = any(obj in SUSPICIOUS_OBJECTS for obj in detected_classes)

    if alert:
        with open('static/alerts/alert_history.txt', 'a') as f:
            for obj in detected_classes:
                if obj in SUSPICIOUS_OBJECTS:
                    f.write(f"[{datetime.datetime.now()}] | {obj} | ALERT: YES\n")
    else:
        with open('static/alerts/alert_history.txt', 'a') as f:
            f.write(f"[{datetime.datetime.now()}] | No suspicious object | ALERT: NO\n")

    return jsonify({'alert': alert})

################################################################# doua ######################################################################################
# Gait Analysis Routes
from flask import Flask, render_template, url_for, request, jsonify
from gait import GaitAnalyzer
from flask_cors import CORS
CORS(app)  # Enable CORS for all routes
gait_analyzer = GaitAnalyzer()

@app.route('/gait.html')
def gait():
    return render_template('gait.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    print("Received request to analyze video")
    if 'video' not in request.files:
        print("No video file provided")
        return jsonify({'error': 'No video file provided'}), 400

    success, result = gait_analyzer.analyze_video(request.files['video'])
    print(f"Analysis result: {result}")

    if success:
        return jsonify({
            'success': True,
            'person_id': result['person_id'],
            'confidence': result['confidence']
        })
    else:
        print(f"Error during analysis: {result['error']}")
        return jsonify({'error': result['error']}), 400
   
# """""""""""Ranim"""""""""
def extract_fields(results, image_np):
    """
    Extrait les champs en gérant les détections multiples et les cas spéciaux
    """
    extracted_fields = {}
    boxes = results[0].boxes

    # Dictionnaire pour stocker les scores de confiance par type de base
    confidence_by_type = {}
    # Dictionnaire pour stocker les parties d'adresse
    address_parts = []

    for box in boxes:
        cls_id = int(box.cls[0].item())
        cls_name = results[0].names[cls_id]
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        roi = image_np[y1:y2, x1:x2]

        # Extraire le type de base (avant le numéro)
        base_type = cls_name.split('_')[0]

        # Cas spécial pour l'adresse
        if base_type == 'address':
            address_parts.append({
                'roi': roi,
                'confidence': confidence,
                'bbox': (x1, y1, x2, y2)
            })
            continue

        # Pour les autres champs, garder celui avec la meilleure confiance
        field_key = base_type + '_name' if base_type in ['first', 'last', 'full', 'mother'] else base_type

        if field_key not in confidence_by_type or confidence > confidence_by_type[field_key]:
            extracted_fields[field_key] = {
                'roi': roi,
                'confidence': confidence,
                'bbox': (x1, y1, x2, y2)
            }
            confidence_by_type[field_key] = confidence

    # Traitement spécial pour l'adresse si des parties ont été détectées
    if address_parts:
        if len(address_parts) > 1:
            # Trier les parties d'adresse de gauche à droite
            address_parts.sort(key=lambda x: x['bbox'][0])

            # Calculer les dimensions de la boîte englobante
            min_x = min(part['bbox'][0] for part in address_parts)
            min_y = min(part['bbox'][1] for part in address_parts)
            max_x = max(part['bbox'][2] for part in address_parts)
            max_y = max(part['bbox'][3] for part in address_parts)

            # Extraire la ROI complète
            full_roi = image_np[min_y:max_y, min_x:max_x]

            # Calculer la confiance moyenne
            avg_confidence = sum(part['confidence'] for part in address_parts) / len(address_parts)

            # Stocker l'adresse complète
            extracted_fields['address'] = {
                'roi': full_roi,
                'confidence': avg_confidence,
                'bbox': (min_x, min_y, max_x, max_y)
            }
        else:
            # S'il n'y a qu'une partie, la stocker directement
            extracted_fields['address'] = address_parts[0]

    return extracted_fields
def preprocess_field(roi, field_type):
    """
    Prétraitement spécifique pour chaque type de champ

    Args:
        roi: Image de la région d'intérêt
        field_type: Type du champ (first_name, dob, address, etc.)

    Returns:
        np.array: Image prétraitée
    """
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

    # Initialisation de processed
    processed = gray.copy()

    if field_type.lower() in ['first_name', 'last_name', 'full_name', 'mother_name']:
        # Paramètres optimaux pour le texte arabe
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        processed = clahe.apply(gray)
        processed = cv2.fastNlMeansDenoising(processed)
        _, processed = cv2.threshold(processed, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Dilatation minimale
        kernel = np.ones((2,2), np.uint8)
        processed = cv2.dilate(processed, kernel, iterations=1)

    elif field_type.lower() == 'id':
        # Traitement simple pour les chiffres
        processed = cv2.GaussianBlur(gray, (3, 3), 0)
        _, processed = cv2.threshold(processed, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif field_type.lower() == 'dob':
        # Optimisé pour les dates
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        processed = clahe.apply(gray)
        processed = cv2.fastNlMeansDenoising(processed)
        _, processed = cv2.threshold(processed, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif field_type.lower() == 'pob':
        # Nouveau traitement optimisé pour le lieu de naissance
        # 1. Amélioration du contraste
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(gray)

        # 2. Débruitage plus agressif
        denoised = cv2.fastNlMeansDenoising(contrast_enhanced, h=15)

        # 3. Binarisation avec Otsu au lieu du seuillage adaptatif
        _, binary = cv2.threshold(denoised, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4. Nettoyage morphologique
        kernel = np.ones((2,2), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    elif field_type.lower() == 'address':
        # Traitement pour l'adresse
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        processed = clahe.apply(gray)
        processed = cv2.fastNlMeansDenoising(processed)
        _, processed = cv2.threshold(processed, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    else:
        # Traitement par défaut
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(gray)
        _, processed = cv2.threshold(processed, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Redimensionnement final modéré
    height, width = processed.shape
    scale_factor = 2
    processed = cv2.resize(processed, (width*scale_factor, height*scale_factor),
                         interpolation=cv2.INTER_CUBIC)

    # Ajout d'une bordure blanche
    processed = cv2.copyMakeBorder(processed, 10, 10, 10, 10,
                                 cv2.BORDER_CONSTANT,
                                 value=[255, 255, 255])

    return processed
def perform_ocr_with_easyocr(preprocessed_image, field_type):
    try:
        # Normalize field type to lowercase for consistent comparison
        field_type = field_type.lower()

        # Handle the 'issue' field type variant
        if field_type == 'issue':
            field_type = 'issue_date'

        # Initialize EasyOCR with appropriate language based on field type
        if field_type == 'id':
            reader = easyocr.Reader(['en'], gpu=True)
            allowlist = '0123456789'
        else:
            reader = easyocr.Reader(['ar', 'en'], gpu=True)
            allowlist = None

        # Configure parameters based on field type
        use_paragraph = field_type in ['full_name', 'dob', 'address', 'mother_name', 'issue_date']

        # Set OCR parameters
        results = reader.readtext(
            preprocessed_image,
            paragraph=use_paragraph,
            allowlist=allowlist,
            batch_size=1
        )

        print(f"Raw OCR results for {field_type}:")
        print(results)

        if not results:
            return None

        # Process based on field type
        if field_type == 'id':
            if len(results[0]) >= 2:
                text = results[0][1]
                print(f"Extracted ID text: {text}")
                print(f"Is digit check: {text.isdigit()}")
                print(f"Length check: {len(text)}")

                if text.isdigit() and len(text) == 8:
                    print(f"Valid ID found: {text}")
                    return text
                cleaned = ''.join(filter(str.isdigit, text))
                if len(cleaned) == 8:
                    print(f"Valid cleaned ID found: {cleaned}")
                    return cleaned
            return None

        elif field_type in ['dob', 'issue_date']:
            # Handle dates (both DOB and issue date)
            if len(results[0]) >= 2:
                text = results[0][1]
                # Keep both numbers and Arabic text for dates
                cleaned = re.sub(r'[^؀-ۿ0-9\s]', '', text)
                cleaned = cleaned.strip()
                print(f"Cleaned date text: {cleaned}")
                return cleaned if cleaned else None

        elif field_type == 'address':
            # Handle address with both numbers and Arabic text
            if len(results[0]) >= 2:
                text = results[0][1]
                # Keep both numbers and Arabic text for address
                cleaned = re.sub(r'[^؀-ۿ0-9\s]', '', text)
                # Split into parts and filter out invalid parts
                parts = [p.strip() for p in cleaned.split() if len(p.strip()) > 1]
                return ' '.join(parts) if parts else None

        elif field_type == 'profession':
            # Handle profession (Arabic text only)
            if len(results[0]) >= 2:
                text = results[0][1]
                cleaned = re.sub(r'[^؀-ۿ\s]', '', text)
                return cleaned.strip() if cleaned.strip() else None

        elif field_type == 'mother_name':
            # Handle mother's name
            if len(results[0]) >= 2:
                text = results[0][1]
                cleaned = re.sub(r'[^؀-ۿ\s]', '', text)
                return cleaned.strip() if cleaned.strip() else None

        elif field_type == 'pob':
            texts = [result[1] for result in results]
            cleaned_parts = []
            for text in texts:
                cleaned = re.sub(r'[^؀-ۿ\s]', '', text)
                parts = [p for p in cleaned.split() if len(p) > 1 and not p.isdigit()]
                cleaned_parts.extend(parts)
            return ' '.join(cleaned_parts) if cleaned_parts else None

        elif field_type == 'full_name':
            name_parts = []
            for result in results:
                if len(result) >= 2:
                    text = result[1]
                    cleaned = re.sub(r'[^؀-ۿ\s]', '', text)
                    if cleaned and len(cleaned.strip()) > 1:
                        name_parts.append(cleaned.strip())

            if len(results) > 1:
                sorted_results = sorted(zip(results, name_parts),
                                     key=lambda x: x[0][0][0][0])
                name_parts = [part for _, part in sorted_results]

            return ' '.join(name_parts) if name_parts else None

        elif field_type in ['first_name', 'last_name']:
            text_parts = []
            for result in results:
                if len(result) >= 3:
                    _, text, conf = result
                    if conf > 0.3:
                        cleaned = re.sub(r'[^؀-ۿ\s]', '', text)
                        if cleaned and len(cleaned.strip()) > 1:
                            text_parts.append(cleaned.strip())
                elif len(result) >= 2:
                    text = result[1]
                    cleaned = re.sub(r'[^؀-ۿ\s]', '', text)
                    if cleaned and len(cleaned.strip()) > 1:
                        text_parts.append(cleaned.strip())

            if text_parts:
                text_parts.sort(key=len, reverse=True)
                return text_parts[0]
            return None

        return None

    except Exception as e:
        print(f"Error in OCR processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None




def ton_module_de_detection(img_path):
    image = cv2.imread(img_path)

    results = detection_model(image)
    fields = extract_fields(results, image)

    extracted_data = {}
    for field_type, field_info in fields.items():
        roi = field_info['roi']
        preprocessed_roi = preprocess_field(roi, field_type)
        text = perform_ocr_with_easyocr(preprocessed_roi, field_type)

        if text:
            extracted_data[field_type] = text

    return extracted_data


# S'assurer que le dossier 'uploads' existe
if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route('/ocr', methods=['GET', 'POST'])
def ocr_detect():
    if request.method == 'POST':
        file = request.files['file']
        img_path = os.path.join('uploads', file.filename)
        file.save(img_path)

        detected_text = ton_module_de_detection(img_path)

        # Convertir en liste
        detected_text = [f"{key}: {value}" for key, value in detected_text.items()]

        return render_template('ocr.html', detected_text=detected_text)

    return render_template('ocr.html', detected_text=None)


#-----------------------sound detection
@app.route('/sound', methods=['GET', 'POST'])
def sound():
    prediction = None
    confidence = None
    if request.method == 'POST':
        if 'audiofile' not in request.files:
            return render_template('sound.html', error="No file part")
        file = request.files['audiofile']
        if file.filename == '':
            return render_template('sound.html', error="No selected file")
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            result = predict_audio(filepath)
            prediction = result.get('predicted_class')
            confidence = result.get('confidence')
    return render_template('sound.html', prediction=prediction, confidence=confidence)
@app.route('/licenseplate.html', methods=['GET', 'POST'])
def licenseplate():
    output_video = None
    if request.method == 'POST':
        if 'videofile' not in request.files:
            return render_template('licenseplate.html', error="No file part")
        file = request.files['videofile']
        if file.filename == '':
            return render_template('licenseplate.html', error="No selected file")
        if file:
            upload_folder = app.config['UPLOAD_FOLDER']
            os.makedirs(upload_folder, exist_ok=True)
            input_path = os.path.join(upload_folder, file.filename)
            file.save(input_path)
            print(f"[DEBUG] Input video path: {input_path}")
            # Output video path
            output_folder = app.config['DETECTION_FOLDER']
            os.makedirs(output_folder, exist_ok=True)
            output_filename = 'lp_' + file.filename.rsplit('.', 1)[0] + '.mp4'
            output_path = os.path.join(output_folder, output_filename)
            print(f"[DEBUG] Output video path: {output_path}")
            # Process video
            process_video(input_path, output_path=output_path)
            # Check if output file exists and its size
            if os.path.exists(output_path):
                print(f"[DEBUG] Output video exists. Size: {os.path.getsize(output_path)} bytes")
            else:
                print(f"[DEBUG] Output video does NOT exist!")
            # Pass relative path for HTML video tag
            output_video = output_path.replace('static/', '')
    return render_template('licenseplate.html', output_video=output_video)

# --- Run App ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')