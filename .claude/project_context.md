# Automated Attendance System: Project Context

## Overview
This project is an end-to-end face recognition-based automated attendance system. It uses modern deep learning architectures to process images or video feeds (including real-time webcam) to identify individuals and mark their attendance in a digital spreadsheet.

## Key Goals
- **Automation**: Replace manual attendance marking with an efficient biometric system.
- **Robustness**: Maintain high accuracy under varied lighting, facial orientations, and occlusions.
- **Efficiency**: Utilize pre-trained models (FaceNet) for fast and reliable face embedding extraction.

## Core Technologies
- **TensorFlow**: Underlying deep learning framework.
- **MTCNN**: Used for robust face detection and alignment.
- **FaceNet**: Used for generating high-quality face embeddings.
- **Support Vector Machine (SVM)**: Used as the final classifier to match embeddings to registered users.
- **Tkinter**: Provides the GUI for dataset creation, training, and running the system.
- **Excel/XLS**: Used for attendance recording.
- **OpenCV**: Handles image and video stream processing.
