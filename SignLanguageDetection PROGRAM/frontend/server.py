"""Simple Flask server that accepts a base64 image and returns a predicted label.

Usage:
  python server.py

It expects to find the trained model at ../model.p (one level up from frontend/). If your
model is elsewhere, modify MODEL_PATH below.

This is optional. The frontend will run without it (demo mode)."""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import io
import cv2
import numpy as np
import pickle

# mediapipe is optional at runtime. If it's not available the server will still run
# in a demo mode (useful for people who just want the frontend to be reachable).
try:
    import mediapipe as mp
except Exception as e:
    mp = None
    print('mediapipe unavailable:', e)

# Location of the trained model. By default we look one level up where training.py saves it.
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model.p'))

app = Flask(__name__)
CORS(app)

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = pickle.load(open(MODEL_PATH, 'rb')).get('model')
        print('Loaded model from', MODEL_PATH)
    except Exception as e:
        print('Failed to load model:', e)
else:
    print('model.p not found at', MODEL_PATH, " — server will return demo responses until a model exists.")

# Load numeric->letter mapping from project labels.json if present
LABELS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'labels.json'))


def load_labels():
    """Load labels.json from disk and return a numeric->string mapping.

    This is called per-request so edits to labels.json take effect immediately
    without restarting the server.
    """
    if not os.path.exists(LABELS_PATH):
        return {}
    try:
        import json
        with open(LABELS_PATH, 'r', encoding='utf-8') as fh:
            raw = json.load(fh)
            return {int(k): v for k, v in raw.items()}
    except Exception as e:
        print('Failed to load labels.json:', e)
        return {}

mp_hands = None
if mp is not None:
    try:
        mp_hands = mp.solutions.hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
    except Exception as e:
        print('Failed to initialize mediapipe hands:', e)


def decode_base64_image(data_url):
    # data_url: data:image/jpeg;base64,/9j/...
    if ',' in data_url:
        header, data = data_url.split(',', 1)
    else:
        data = data_url
    decoded = base64.b64decode(data)
    arr = np.frombuffer(decoded, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json()
    if not payload or 'image' not in payload:
        return jsonify({'status':'error', 'message':'no image provided'}), 400

    try:
        img = decode_base64_image(payload['image'])
        if img is None:
            return jsonify({'status':'error', 'message':'invalid image'}), 400

        # convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if mp_hands is None:
            # mediapipe not available — server is reachable but cannot perform hand detection
            return jsonify({'status': 'error', 'message': 'mediapipe not available on server'}), 501

        results = mp_hands.process(img_rgb)

        if not results or not results.multi_hand_landmarks:
            return jsonify({'status':'no_hand'})

        # take first hand
        hand_landmarks = results.multi_hand_landmarks[0]
        data_aux = []
        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x)
            data_aux.append(lm.y)
        while len(data_aux) < 63:
            data_aux.append(0)

        if model is None:
            return jsonify({'status':'ok', 'prediction': None, 'message':'no model loaded'})

        pred = model.predict([np.asarray(data_aux)])
        label = int(pred[0])
        # Load labels per-request so edits to labels.json take effect immediately
        labels_map = load_labels()
        # optional: if no mapping found, fall back to numeric string
        label_str = labels_map.get(label, str(label))
        # Log mapping and result for debugging
        print('Predict -> label:', label, 'label_str:', label_str, 'labels_map:', labels_map)
        return jsonify({'status':'ok', 'prediction': label, 'label': label_str})

    except Exception as e:
        print('Prediction error:', e)
        return jsonify({'status':'error', 'message': str(e)}), 500


if __name__ == '__main__':
    # Allow overriding host/port via environment for easy deployment (e.g. Docker/Render)
    HOST = os.environ.get('HOST', '127.0.0.1')
    PORT = int(os.environ.get('PORT', '5000'))
    print(f'Starting server on http://{HOST}:{PORT}')
    app.run(host=HOST, port=PORT)


@app.route('/labels', methods=['GET'])
def labels_endpoint():
    """Return the current labels mapping (useful to debug which mapping the server is using)."""
    try:
        labels_map = load_labels()
        return jsonify({'status': 'ok', 'labels': labels_map})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Simple health endpoint so load balancers or the frontend can check availability."""
    return jsonify({'status': 'ok', 'model_loaded': model is not None, 'mediapipe': mp is not None})
