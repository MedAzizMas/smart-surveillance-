import os
from werkzeug.utils import secure_filename
from gait_recognition.gait_recognition_pipeline import process_video_pipeline

class GaitAnalyzer:
    def __init__(self, upload_folder='gait_recognition/uploads', max_file_size=16 * 1024 * 1024):
        self.upload_folder = upload_folder
        self.max_file_size = max_file_size
        self.allowed_extensions = {'mp4', 'avi', 'mov'}
        
        # Create uploads directory if it doesn't exist
        os.makedirs(self.upload_folder, exist_ok=True)

    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.allowed_extensions

    def analyze_video(self, video_file):
        """
        Analyze a video file using the gait recognition pipeline
        
        Args:
            video_file: The uploaded video file object
            
        Returns:
            tuple: (success, result)
                success: boolean indicating if analysis was successful
                result: dictionary containing analysis results or error message
        """
        if not video_file:
            return False, {'error': 'No video file provided'}
            
        if video_file.filename == '':
            return False, {'error': 'No selected file'}
            
        if not self.allowed_file(video_file.filename):
            return False, {'error': 'Invalid file type'}
            
        try:
            # Save the uploaded file
            filename = secure_filename(video_file.filename)
            filepath = os.path.join(self.upload_folder, filename)
            video_file.save(filepath)
            
            # Process the video using the gait recognition pipeline with CSTL model
            person_id, confidence = process_video_pipeline(
                filepath,
                'gait_recognition/segmentation.h5',
                'gait_recognition/cstl_final_model.pth'  # Updated to use CSTL model
            )
            
            # Clean up the uploaded file
            os.remove(filepath)
            
            return True, {
                'person_id': person_id,
                'confidence': confidence
            }
            
        except Exception as e:
            # Clean up the uploaded file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            return False, {'error': str(e)} 