
from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import csv
import os
from roboflow import Roboflow
from datetime import datetime

app = Flask(__name__)

# Initialize Roboflow model
rf = Roboflow(api_key="G2GTN1K93dty31uz5sGb")
project = rf.workspace().project("license-plate-recognition-rxg4e")
model = project.version(4).model

# Video capture
cap = cv2.VideoCapture('video.mp4')

# CSV files for storing plate numbers
violation_file = 'violations.csv'
safe_file = 'safe_plate.csv'

# Ensure the CSV files exist
for file in [violation_file, safe_file]:
    if not os.path.exists(file):
        with open(file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Plate", "Timestamp"])

# Simulate traffic light state (this should be replaced by actual detection logic)
traffic_light_state = "red"  # Change to "green" as needed

def detect_license_plate(frame):
    height, width, _ = frame.shape
    result = model.predict(frame, confidence=40, overlap=30).json()
    plates = []
    
    for prediction in result['predictions']:
        x = int(prediction['x'] - prediction['width'] / 2)
        y = int(prediction['y'] - prediction['height'] / 2)
        w = int(prediction['width'])
        h = int(prediction['height'])
        label = prediction['class']
        confidence = prediction['confidence']
        
        # Draw rectangle around the license plate
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        plates.append(label)
    
    return frame, plates

def save_plate(plate, file):
    existing_plates = set()
    if os.path.exists(file):
        with open(file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                existing_plates.add(row[0])

    if plate not in existing_plates:
        with open(file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([plate, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame, plates = detect_license_plate(frame)
            
            for plate in plates:
                if traffic_light_state == "red":
                    save_plate(plate, violation_file)
                else:
                    save_plate(plate, safe_file)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
