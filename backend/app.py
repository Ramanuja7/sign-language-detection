from flask import Flask, render_template, Response
import cv2
import torch
import os

app = Flask(__name__)

# Load YOLOv5 model (force CPU usage)
device = 'cpu'  # Force CPU usage (no CUDA)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/Ramanuja/Sign-Language-Generation-From-Video-using-YOLOV5/yolov5/best.pt', device=device)

# Function to detect objects in a frame
def detect_frame(frame):
    # Resize the frame to 640x640 (YOLOv5 default input size)
    frame_resized = cv2.resize(frame, (640, 640))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Run the YOLOv5 model on the frame
    results = model(frame_rgb)
    frame_result = results.render()[0]  # Get frame with detection overlays
    return frame_result

# Video processing function (for continuous stream)
def process_video(video_source=0, vid_stride=1):
    # Open the video (can be webcam or video file)
    cap = cv2.VideoCapture(video_source)
    
    # Frame counter to implement vid_stride
    frame_counter = 0

    while True:
        success, frame = cap.read()
        if not success:
            break  # End of video or no frames

        # Skip frames based on vid_stride to reduce processing load
        if frame_counter % vid_stride == 0:
            # Perform detection on the frame
            frame_result = detect_frame(frame)

            # Encode the frame for streaming
            ret, buffer = cv2.imencode('.jpg', frame_result)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

        frame_counter += 1

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Change '0' to a video file path if you want to test with a video file
    # For example: 'path_to_video.mp4'
    video_source = 0  # 0 for webcam, or specify a path for a video file
    return Response(process_video(video_source, vid_stride=2), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Create the 'uploads' directory if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True)
