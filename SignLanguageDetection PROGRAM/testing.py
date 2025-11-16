# Robust testing script: handle missing model and ensure webcam opens
import pickle  # Used to load the trained model
import cv2  # OpenCV library for video capturing and image processing
import mediapipe as mp  # Mediapipe for hand landmark detection
import numpy as np  # NumPy for handling arrays and numerical operations
import os

# Note: model loading is deferred to run_camera so this module can be imported safely
model = None
model_path = os.path.join(os.path.dirname(__file__), 'model.p')

# Note: video capture is created inside run_camera; avoid opening camera on import
# Set up Mediapipe module reference (instances are created inside run_camera)
mp_hands = mp.solutions.hands
hands = None


# Dictionary for mapping model predictions to corresponding labels
# Default labels mapping (update if your folders map differently)
labels_dict = {0: 'A', 1: 'B', 2: 'C'}  # Update with actual labels corresponding to your dataset

# Try to load labels mapping from labels.json (project root) if present so server/frontend and this
# module share the same mapping. Expecting keys as stringified integers.
try:
    import json
    labels_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'labels.json'))
    if os.path.exists(labels_path):
        raw = json.load(open(labels_path, 'r', encoding='utf-8'))
        labels_dict = {int(k): v for k, v in raw.items()}
        print('Loaded labels mapping from', labels_path, labels_dict)
except Exception:
    pass

# Define zoom parameters for the video feed
zoom_factor = 1.5  # This factor controls the level of zoom in the video
# We'll obtain frame width/height after the first successful read to avoid zeros
width = 0
height = 0
center_x = center_y = None

# Initialize Mediapipe drawing utilities for rendering hand landmarks
mp_drawing = mp.solutions.drawing_utils
mp_hands_style = mp.solutions.drawing_styles


def run_camera(model=None, labels=None):
    """Run the camera preview and (optionally) predictions.

    Args:
        model: a trained model object with a predict method (optional). If None,
            the function will try to load `model.p` from the script directory.
        labels: dict mapping numeric labels to display characters. If None, uses
            the module-level `labels_dict`.
    """
    labels = labels or labels_dict

    # Load model if not provided
    if model is None:
        if os.path.exists(model_path):
            try:
                model_dict = pickle.load(open(model_path, 'rb'))
                model = model_dict.get('model', None)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Warning: failed to load model.p: {e}")
        else:
            print(f"model.p not found at {model_path}. Running in demo mode (no predictions).")

    # Initialize video capture from the webcam (device 0). Try DirectShow backend if default fails.
    cap_local = cv2.VideoCapture(0)
    if not cap_local.isOpened():
        cap_local = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Set up Mediapipe Hands for detecting hand landmarks (video mode)
    mp_hands_local = mp.solutions.hands
    hands_local = mp_hands_local.Hands(static_image_mode=False, max_num_hands=1,
                                       min_detection_confidence=0.3, min_tracking_confidence=0.3)

    # We'll obtain frame width/height after the first successful read to avoid zeros
    width = 0
    height = 0
    center_x = center_y = None

    # Main loop for capturing video and performing hand gesture prediction
    while True:
        data_aux = []  # Auxiliary list to store hand landmark data
        ret, frame = cap_local.read()  # Capture a frame from the webcam

        if not ret or frame is None:
            print("Warning: Could not read frame. Trying again...")
            if not cap_local.isOpened():
                print("Camera not available. Exiting.")
                break
            continue

        # If we haven't set width/height yet, derive from the first frame
        if width == 0 or height == 0:
            height, width = frame.shape[:2]
            center_x, center_y = int(width / 2), int(height / 2)

        # Calculate the zoomed region based on the zoom factor and center coordinates
        x1 = int(center_x - (width / (2 * zoom_factor)))
        x2 = int(center_x + (width / (2 * zoom_factor)))
        y1 = int(center_y - (height / (2 * zoom_factor)))
        y2 = int(center_y + (height / (2 * zoom_factor)))

        # Clamp coordinates to frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        # Crop the frame to apply the zoom effect
        frame_zoomed = frame[y1:y2, x1:x2]

        # If crop failed (empty), fallback to original frame
        if frame_zoomed.size == 0:
            frame_zoomed = frame.copy()

        # Resize the cropped (zoomed) frame back to the original frame dimensions if needed
        try:
            frame_zoomed = cv2.resize(frame_zoomed, (width, height))
        except Exception:
            frame_zoomed = frame.copy()

        # Convert the frame from BGR to RGB color space (Mediapipe requires RGB format)
        frame_rgb = cv2.cvtColor(frame_zoomed, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hand landmarks using Mediapipe
        results = hands_local.process(frame_rgb)

        # Check if any hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the detected hand landmarks and connections on the zoomed frame
                mp_drawing.draw_landmarks(
                    frame_zoomed,
                    hand_landmarks,
                    mp_hands_local.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # Green landmarks
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)  # Red connections
                )

                # Extract landmark coordinates for each detected landmark
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x  # x-coordinate of the landmark
                    y = hand_landmarks.landmark[i].y  # y-coordinate of the landmark
                    data_aux.append(x)  # Append x to data list
                    data_aux.append(y)  # Append y to data list

                # Pad the data to ensure it matches the expected input size of the model
                while len(data_aux) < 63:
                    data_aux.append(0)

                # Make a prediction using the trained model based on the hand landmarks (if model available)
                if model is not None:
                    try:
                        prediction = model.predict([np.asarray(data_aux)])
                        predicted_character = labels.get(int(prediction[0]), str(prediction[0]))
                        # Display the predicted character on the zoomed frame
                        cv2.putText(frame_zoomed, predicted_character, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                                    cv2.LINE_AA)
                    except Exception as e:
                        # If prediction fails, show debug text but continue
                        cv2.putText(frame_zoomed, 'Prediction error', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                    (0, 255, 255), 2, cv2.LINE_AA)
                        print(f"Prediction error: {e}")
                else:
                    # No model available — show notice on frame
                    cv2.putText(frame_zoomed, 'No model (demo)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0, 255, 255), 2, cv2.LINE_AA)
        else:
            # If no hands are detected, display a message on the screen
            cv2.putText(frame_zoomed, 'No hands detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3,
                        cv2.LINE_AA)

        # Draw a small on-screen hint at the bottom: Press Enter to leave
        try:
            if width and height:
                hint = 'Press Enter to leave (end program)'
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.6
                thickness = 1
                (text_w, text_h), _ = cv2.getTextSize(hint, font, scale, thickness)
                text_x = 10
                text_y = height - 10
                cv2.rectangle(frame_zoomed, (text_x - 5, text_y - text_h - 5), (text_x + text_w + 5, text_y + 5), (0, 0, 0), -1)
                cv2.putText(frame_zoomed, hint, (text_x, text_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        except Exception:
            pass

        # Show the zoomed frame with landmarks and predictions in a window
        cv2.imshow('Video Capture', frame_zoomed)

        # Exit the loop when the ESC key or Enter key is pressed
        key = cv2.waitKey(1) & 0xFF
        # 27 = ESC, 13 = Enter (CR), 10 = LF (some systems return 10)
        if key in (27, 13, 10):
            print('Exit key pressed, closing.')
            break

    # Release video capture resources and close all OpenCV windows
    cap_local.release()
    cv2.destroyAllWindows()
