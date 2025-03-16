# from flask import Flask, request, jsonify
# import cv2
# import torch
# import pygame
# import pyttsx3
# from ultralytics import YOLO
# import time
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)  # Allow frontend to call Flask API
# # Initialize YOLO model
# device = "mps" if torch.backends.mps.is_available() else "cpu"
# model = YOLO("yolov8x.pt").to(device)

# # Load alert sound
# pygame.mixer.init()
# pygame.mixer.music.load("alert-33762.mp3")

# def estimate_distance(bbox_width, frame_width, object_name):
#     """Estimate steps needed based on bounding box size relative to frame width and object type."""
#     if "car" in object_name.lower():
#         if bbox_width > frame_width * 0.6:
#             return 1
#         elif bbox_width > frame_width * 0.4:
#             return 2
#         elif bbox_width > frame_width * 0.2:
#             return 3
#         else:
#             return 5
#     else:
#         if bbox_width > frame_width * 0.6:
#             return 1
#         elif bbox_width > frame_width * 0.4:
#             return 3
#         elif bbox_width > frame_width * 0.2:
#             return 5
#         else:
#             return 7

# def get_navigation_steps(x_center, bbox_width, frame_width, object_name):
#     """Provide directional guidance based on the detected object's position."""
#     distance_steps = estimate_distance(bbox_width, frame_width, object_name)
    
#     if x_center < frame_width * 0.3:
#         return f"Walk {distance_steps} steps forward, then turn slightly left."
#     elif x_center > frame_width * 0.7:
#         return f"Walk {distance_steps} steps forward, then turn slightly right."
#     else:
#         return f"Walk {distance_steps} steps straight ahead."

# @app.route("/detect", methods=["POST"])
# def detect_object():
#     data = request.json
#     object_to_find = data.get("object_name", "").lower()

#     if not object_to_find:
#         return jsonify({"error": "No object specified"}), 400

#     cap = cv2.VideoCapture(0)
    
#     if not cap.isOpened():
#         return jsonify({"error": "Failed to open camera"}), 500

#     found = False
#     detected_object = None
#     steps = ""

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = cv2.resize(frame, (640, 480))
#         results = model(frame, verbose=False)

#         for detection in results[0].boxes.data:
#             class_id = int(detection[-1])
#             object_name = model.names[class_id]

#             if object_to_find in object_name.lower():
#                 found = True
#                 detected_object = object_name
#                 x1, y1, x2, y2 = map(int, detection[:4])
#                 x_center = (x1 + x2) // 2
#                 bbox_width = x2 - x1
#                 frame_width = frame.shape[1]

#                 # Navigation steps
#                 steps = get_navigation_steps(x_center, bbox_width, frame_width, object_name)

#                 # Initialize a new pyttsx3 instance
#                 tts_engine = pyttsx3.init()
#                 tts_engine.say(steps)
#                 tts_engine.runAndWait()

#                 # Play alert sound
#                 pygame.mixer.music.play()
#                 time.sleep(2)  # Small delay before stopping
#                 pygame.mixer.music.stop()

#                 cap.release()
#                 cv2.destroyAllWindows()
#                 return jsonify({"found": True, "object": object_name, "steps": steps})

#     cap.release()
#     cv2.destroyAllWindows()
#     return jsonify({"found": False, "message": "Object not found"})

# if __name__ == "__main__":
#     app.run(debug=False)


from flask import Flask, request, jsonify
import cv2
import torch
import pygame
import pyttsx3
from ultralytics import YOLO
from flask_cors import CORS
import time
import os
app = Flask(__name__)
CORS(app)

# Initialize YOLO model
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = YOLO("yolov8x.pt").to(device)

# Load alert sound
pygame.mixer.init()
pygame.mixer.music.load("alert-33762.mp3")

def estimate_distance(bbox_width, frame_width, object_name):
    """Estimate steps based on bounding box size relative to frame width."""
    if "car" in object_name.lower():
        return 1 if bbox_width > frame_width * 0.6 else 2 if bbox_width > frame_width * 0.4 else 3 if bbox_width > frame_width * 0.2 else 5
    return 1 if bbox_width > frame_width * 0.6 else 3 if bbox_width > frame_width * 0.4 else 5 if bbox_width > frame_width * 0.2 else 7

def get_navigation_steps(x_center, bbox_width, frame_width, object_name):
    """Provide navigation steps based on object position."""
    steps = estimate_distance(bbox_width, frame_width, object_name)
    if x_center < frame_width * 0.3:
        return f"Walk {steps} steps forward, then turn slightly left."
    elif x_center > frame_width * 0.7:
        return f"Walk {steps} steps forward, then turn slightly right."
    return f"Walk {steps} steps straight ahead."

@app.route("/detect", methods=["POST"])
def detect_object():
    data = request.json
    print("Received Data:", data)  # Debugging statement
    object_to_find = data.get("object_name", "").lower()

    if not object_to_find:
        return jsonify({"error": "No object specified"}), 400

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"error": "Failed to access camera"}), 500

    found = False
    detected_object = None
    steps = ""
    
    start_time = time.time()  # Start timer
    max_search_time = 40  # Set max search time in seconds
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        results = model(frame, verbose=False)

        for detection in results[0].boxes.data:
            class_id = int(detection[-1])
            object_name = model.names[class_id]

            if object_to_find in object_name.lower():
                found = True
                detected_object = object_name
                x1, y1, x2, y2 = map(int, detection[:4])
                x_center = (x1 + x2) // 2
                bbox_width = x2 - x1
                frame_width = frame.shape[1]

                steps = get_navigation_steps(x_center, bbox_width, frame_width, object_name)

                # Speak the navigation steps
                tts_engine = pyttsx3.init()
                tts_engine.say(steps)
                tts_engine.runAndWait()

                # Play alert sound
                pygame.mixer.music.play()

                # Reduce sleep time to avoid delay
                time.sleep(2)  
                pygame.mixer.music.stop()

                cap.release()
                cv2.destroyAllWindows()
                return jsonify({"found": True, "object": object_name, "steps": steps})

        # Stop search if max time is reached
        if time.time() - start_time > max_search_time:
            break  

    cap.release()
    cv2.destroyAllWindows()
    return jsonify({"found": False, "message": f"'{object_to_find}' not found in the current view."})

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=False)  # Set debug=False

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
