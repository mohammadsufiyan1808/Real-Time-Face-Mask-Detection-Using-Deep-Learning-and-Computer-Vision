# VisionAI — Real-Time Face Mask Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey.svg)

VisionAI is a real-time **Face Mask Detection & Access Control System**. Powered by deep learning and computer vision, it actively monitors webcam feeds to detect whether individuals are wearing face masks. It features a modern web dashboard and automated security logging for unmasked individuals.

## 🌟 Features

- **Real-Time Classification**: Live video inference using a custom-trained Keras model (`mask_detector.h5`) to classify faces as `Mask` or `No Mask`.
- **Automated Security Logging**: Automatically captures and saves snapshots of unmasked violators to a `security_logs/` directory.
- **Modern Web Dashboard**: A sleek, dark-themed dashboard built with Flask and Bootstrap, offering an MJPEG live view and a full-screen "Security Mode".
- **Multi-Engine Face Detection**: 
  - **Web App (`app.py`)**: Uses lightweight OpenCV Haar Cascades for high FPS streaming.
  - **Standalone Script (`detect_mask_video.py`)**: Uses Google's MediaPipe for highly robust and accurate face detection.

## 📂 Project Structure

```text
├── app.py                   # Main Flask backend & MJPEG streaming server
├── detect_mask_video.py     # Standalone OpenCV + MediaPipe local detection script
├── mask_detector.h5         # Pre-trained deep learning model for mask classification
├── plot.png                 # Training accuracy and loss plot
├── templates/
│   └── index.html           # Modern VisionAI web dashboard UI
├── security_logs/           # Auto-generated folder containing snapshots of violators
├── .gitignore               # Git ignore file
└── README.md                # Project documentation
```

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mohammadsufiyan1808/Real-Time-Face-Mask-Detection-Using-Deep-Learning-and-Computer-Vision.git
   cd Real-Time-Face-Mask-Detection-Using-Deep-Learning-and-Computer-Vision
   ```

2. **Install the required dependencies:**
   Make sure you have Python installed, then run:
   ```bash
   pip install opencv-python tensorflow numpy flask mediapipe imutils
   ```
   *(Note: Depending on your system and TensorFlow version, you may need `tensorflow-macos` for Apple Silicon).*

## 💻 Usage

### 1. Web Dashboard (VisionAI Server)
To run the web-based access control dashboard:
```bash
python app.py
```
- Open your browser and navigate to: `http://localhost:5001`
- The app will prompt for camera access and stream the annotated video. Snaphots of unmasked users will be saved in `security_logs/`.

### 2. Standalone Script (Local Window)
To run the detection purely via an OpenCV window (utilizing MediaPipe for face detection):
```bash
python detect_mask_video.py
```
- Press `q` in the OpenCV window to quit the video stream.

## 🧠 How It Works

1. **Face Detection**: The system first locates faces in the video frame using either Haar Cascades or MediaPipe.
2. **Preprocessing**: The bounded face Region of Interest (ROI) is extracted, converted to RGB, resized to 224x224 pixels, and scaled for the model.
3. **Classification**: The preprocessed ROI is fed into the `mask_detector.h5` model (MobileNetV2 base), which outputs probabilities for `Mask` and `No Mask`.
4. **Annotation & Logging**: A bounding box and confidence score are drawn around the face. If `No Mask` is detected for a sustained period, a timestamped image is logged to the local filesystem.
