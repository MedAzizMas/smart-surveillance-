import cv2
import os

def extract_frames(video_path, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        print("Error: Could not open video file")
        return
    
    frame_count = 0
    
    while True:
        # Read a frame from the video
        success, frame = video.read()
        
        # If frame is not read successfully, break the loop
        if not success:
            break
        
        # Save the frame as an image
        frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        
        frame_count += 1
    
    # Release the video object
    video.release()
    print(f"Successfully extracted {frame_count} frames to {output_folder}")

if __name__ == "__main__":
    # Get video path from user
    video_path = input("Enter the path to your video file: ")
    
    # Create a folder for the frames
    output_folder = "frames"
    
    # Extract frames
    extract_frames(video_path, output_folder) 