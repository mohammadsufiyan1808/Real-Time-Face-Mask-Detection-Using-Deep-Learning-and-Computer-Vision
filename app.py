"""
VisionAI — Real-Time Face Mask Detection Server
Flask + OpenCV MJPEG Streaming + TensorFlow/Keras + OpenCV Haar Cascade
"""

import os

# ── Environment flags (must be set BEFORE importing TF/Keras) ────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"          # suppress TF info/warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"         # avoid oneDNN instability

import cv2
import time
import numpy as np
from datetime import datetime
from flask import Flask, Response, render_template

# ── App Initialization ───────────────────────────────────────────────────────

app = Flask(__name__)

# Lazy-loaded globals (initialized on first request)
model = None
face_cascade = None


def _init_model():
    """Load the Keras .h5 model (deferred to avoid import-time issues)."""
    global model
    if model is not None:
        return
    import tensorflow as tf

    # Keras 2 → 3 compatibility patch: strip unsupported 'groups' kwarg
    class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
        def __init__(self, *args, **kwargs):
            kwargs.pop("groups", None)
            super().__init__(*args, **kwargs)

    print("[INFO] Loading face mask detector model...")
    model = tf.keras.models.load_model(
        "mask_detector.h5",
        custom_objects={"DepthwiseConv2D": CustomDepthwiseConv2D},
        compile=False,
    )
    print("[INFO] Model loaded successfully.")


def _init_face_detector():
    """Initialize OpenCV Haar Cascade face detector."""
    global face_cascade
    if face_cascade is not None:
        return
    print("[INFO] Initializing OpenCV Face Detector...")
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"[ERROR] Failed to load cascade from {cascade_path}")
    print("[INFO] Face detector ready.")


# Colors: Green for Mask, Red for No Mask (BGR format)
COLORS = {
    "Mask":    (0, 200, 100),   # green
    "No Mask": (0, 0, 230),     # red
}


# ── Frame Generator ─────────────────────────────────────────────────────────

def generate_frames():
    """
    Capture webcam frames, detect faces via OpenCV Haar Cascade, classify
    each face as Mask / No Mask using the Keras model, annotate the frame,
    and yield it as an MJPEG byte stream.
    """
    # Lazy-init on first call
    _init_model()
    _init_face_detector()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("[ERROR] Cannot access webcam.")
        return

    prev_frame_time = 0
    last_snapshot_time = 0
    os.makedirs('security_logs', exist_ok=True)

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = current_time

        h, w, _ = frame.shape

        # Convert to grayscale for Haar cascade detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=15,
            minSize=(120, 120),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        for (startX, startY, fw, fh) in faces:
            endX = startX + fw
            endY = startY + fh

            # Clamp to frame bounds
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)

            # Extract the face ROI
            face_roi = frame[startY:endY, startX:endX]
            if face_roi.size == 0:
                continue

            # Preprocess for the model (224×224 RGB, normalized)
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (224, 224))
            face_array = np.expand_dims(face_resized, axis=0).astype("float32") / 255.0

            # Predict — model outputs [mask_prob, no_mask_prob]
            preds = model.predict(face_array, verbose=0)[0]
            mask_prob, no_mask_prob = preds[0], preds[1]

            label = "Mask" if mask_prob > no_mask_prob else "No Mask"
            confidence = max(mask_prob, no_mask_prob) * 100
            color = COLORS[label]
            text = f"{label}: {confidence:.1f}%"

            # Draw bounding box
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # Draw label background
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (startX, startY - th - 10),
                          (startX + tw + 6, startY), color, -1)
            cv2.putText(frame, text, (startX + 3, startY - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # ── Security Snapshot Logger ──────────────────────────────
            if label == "No Mask":
                if time.time() - last_snapshot_time > 5:
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filepath = f"security_logs/violator_{timestamp}.jpg"
                    cv2.imwrite(filepath, frame)
                    print(f"[SECURITY ALERT] Unmasked individual logged: {filepath}")
                    last_snapshot_time = time.time()

        # Draw FPS counter on top-left corner
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Encode the annotated frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        # Yield as multipart byte stream (MJPEG)
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        )

    camera.release()


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Render the main dashboard page."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """Stream the annotated webcam feed as MJPEG."""
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
