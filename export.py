from flask import Flask, request, jsonify
import cv2
from ultralytics import YOLO
import supervision as sv
import pyttsx3
import threading

app = Flask(__name__)

def calculate_distance(xyxy):
    # Assuming you have calibrated your camera and know the focal length (f) in pixels
    f = 800  # Example focal length in pixels
    w = 0.38  # Example width of the object in meters (e.g., width of a human face)
    width_pixels = abs(xyxy[2] - xyxy[0])
    distance = (f * w) / width_pixels
    return distance

def speak(engine, text):
    engine.say(text)
    engine.runAndWait()

@app.route('/detect', methods=['POST'])
def detect_objects():
    image = request.files['image'].read()
    nparr = np.frombuffer(image, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    model = YOLO("yolov8n-oiv7.pt")
    results = model(frame, agnostic_nms=True)

    detected_objects = []
    for result in results:
        detections = sv.Detections.from_ultralytics(result)
        detected_names = [result.names[int(cls)] for cls in result.boxes.cls]
        labels = [
            f"{detected_names[i]} {result.boxes.conf[i]:.2f}"
            for i in range(len(result.boxes.cls))
        ]
        
        # Calculate distance for each detection
        for i, xyxy in enumerate(result.boxes.xyxy):
            distance = calculate_distance(xyxy)
            labels[i] += f" Distance: {distance:.2f}m"
            if 0.1 <= distance <= 3:
                threading.Thread(target=speak, args=(engine, f"{detected_names[i]} at {distance:.2f} meters")).start()
        
        detected_objects.append(labels)

    return jsonify(detected_objects)

if __name__ == '__main__':
    engine = pyttsx3.init()
    app.run(debug=True)
