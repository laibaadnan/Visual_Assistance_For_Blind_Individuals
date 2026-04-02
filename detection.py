import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import pyttsx3
import threading

def calculate_distance(xyxy):
    f = 800  # Example focal length in pixels
    w = 0.35  # Example width of the object in meters
    width_pixels = abs(xyxy[2] - xyxy[0])
    distance = (f * w) / width_pixels
    return distance

def meters_to_feet(meters):
    feet = meters * 1.5084  # 1 meter equals 3.28084 feet
    return feet

def speak(engine, text):
    engine.say(text)
    engine.runAndWait()
    engine.setProperty('rate', 120)

def process_frame(frame, model, bbox_annotator, label_annotator, engine):
    person_classes = ["Human face", "Clothing", "Hand", "Human arm", "Human beard", "Human body", "Human ear", "Human eye", "Human hair",
                      "Human leg", "Human mouth", "Human nose", "Man", "Glasses"]
    tree_classes = ["Houseplant", "Lavender (Plant)", "Plant", "Tree house", "Christmas tree", "Palm tree"]
    vehicle_classes = ['Vehicle registration plate', 'Land vehicle', 'Car']

    results = model(frame, agnostic_nms=True)
    labels = []
    voice_feedback = []

    for result in results:
        detected_objects = result.names
        if len(detected_objects) > 0:
            detections = sv.Detections.from_ultralytics(result)
            detected_names = [detected_objects[int(cls)] for cls in result.boxes.cls]

            result_labels = [
                f"{detected_names[i]} {result.boxes.conf[i]:.2f}"
                for i in range(len(result.boxes.cls))
            ]

            spoken_feedback = []

            for i, name in enumerate(detected_names):
                if name in person_classes:
                    detected_names[i] = "person"
                elif name in tree_classes:
                    detected_names[i] = "tree"
                elif name in vehicle_classes:
                    detected_names[i] = "Vehicle"

                distance = calculate_distance(result.boxes.xyxy[i])
                distance_feet = meters_to_feet(distance)
                result_labels[i] += f" Distance: {distance:.2f}m ({distance_feet:.2f}ft)"  # Added feet conversion

                if 0 <= distance_feet <= 16:  # Speaking feedback for distances between 3 to 12 feet
                    spoken_feedback.append(f"{detected_names[i]} at {distance_feet:.2f} feet")

            # Speak feedback for each detected object
            for feedback in spoken_feedback:
                threading.Thread(target=speak, args=(engine, feedback)).start()
                voice_feedback.append(feedback)

            labels.append(result_labels)

            frame = bbox_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(scene=frame, detections=detections)

    return labels, voice_feedback, frame

if __name__ == "__main__":
    camera = cv2.VideoCapture(0)  # Accessing the laptop's camera

    model = YOLO("yolov8n-oiv7.pt")
    bbox_annotator = sv.BoundingBoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1)
    engine = pyttsx3.init()
    engine.setProperty('rate', 120)  # Adjust speech rate as needed

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture frame from camera")
            break

        labels, voice_feedback, annotated_frame = process_frame(frame, model, bbox_annotator, label_annotator, engine)

        cv2.imshow("Object Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    camera.release()
    cv2.destroyAllWindows()
