from flask import Flask, render_template, Response
import cv2
import csv
import os
from roboflow import Roboflow
from datetime import datetime
import threading
import queue

app = Flask(__name__)

# Initialize Roboflow model
rf = Roboflow(api_key="G2GTN1K93dty31uz5sGb")
project = rf.workspace().project("license-plate-recognition-rxg4e")
model = project.version(4).model

# Video capture
cap = cv2.VideoCapture('IMG_0884cTrim.mov')

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

def detect_license_plate(frame, model):
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

        # Crop the license plate from the frame
        plate_image = frame[y:y+h, x:x+w]
        plates.append((label, plate_image))

    return frame, plates

def save_plate(plate, plate_image, file):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"static/{plate}_{timestamp}.jpg"
    cv2.imwrite(filename, plate_image)

    with open(file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([plate, timestamp])

def resize_frame(frame, scale_percent):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def process_frames(cap, frame_queue, model, scale_percent):
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = resize_frame(frame, scale_percent)
        frame, plates = detect_license_plate(frame, model)
        for plate, plate_image in plates:
            if traffic_light_state == "red":
                save_plate(plate, plate_image, violation_file)
            else:
                save_plate(plate, plate_image, safe_file)
        frame_queue.put(frame)

def encode_frames(frame_queue, encoded_queue):
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            encoded_queue.put(buffer.tobytes())

def generate_frames(encoded_queue):
    while True:
        frame = encoded_queue.get()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(encoded_queue), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    frame_queue = queue.Queue(maxsize=10)
    encoded_queue = queue.Queue(maxsize=10)

    # Percentage to scale down the video
    scale_percent = 50  # Scale down to 50% of the original size

    # Start the frame processing and encoding threads
    threading.Thread(target=process_frames, args=(cap, frame_queue, model, scale_percent)).start()
    threading.Thread(target=encode_frames, args=(frame_queue, encoded_queue)).start()

    app.run(debug=True, use_reloader=False)

    # Signal threads to stop and join them
    frame_queue.put(None)
    encoded_queue.put(None)
