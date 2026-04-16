import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import mediapipe as mp
from imutils.video import VideoStream
import time

print("[INFO] Loading trained mask detector model...")
# Make sure "mask_detector.h5" is spelled exactly like your file
model = load_model("mask_detector.h5")

print("[INFO] Initializing MediaPipe Face Detection...")
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

print("[INFO] Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0) # Let the camera warm up

while True:
    frame = vs.read()
    if frame is None:
        break

    # MediaPipe needs RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    h, w = frame.shape[:2]

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            startX = int(bboxC.xmin * w)
            startY = int(bboxC.ymin * h)
            endX = int((bboxC.xmin + bboxC.width) * w)
            endY = int((bboxC.ymin + bboxC.height) * h)

            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)

            if startX < endX and startY < endY:
                face = frame[startY:endY, startX:endX]
                
                # Preprocess for the model
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)
                face = face / 255.0

                # Predict Mask vs No Mask
                (mask, withoutMask) = model.predict(face, verbose=0)[0]

                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Real-Time Face Mask Detection", frame)

    # Press 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

print("[INFO] Cleaning up...")
cv2.destroyAllWindows()
vs.stop()
face_detection.close()
