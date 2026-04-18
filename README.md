# Attendance Management System

Automated attendance tracking using face recognition — built with FaceNet and MTCNN for real-time detection and a Tkinter-based UI for classroom/office use.

## Features

- Real-time face detection using MTCNN
- Face recognition via FaceNet embeddings
- Automated attendance logging to spreadsheet
- Desktop UI built with Tkinter
- Supports multi-face detection in a single frame

## Project Structure

```
├── main.py              # Entry point
├── src/
│   ├── face_detection/  # MTCNN detection & FaceNet recognition
│   ├── ui/              # Tkinter interface components
│   └── utils/           # Spreadsheet logging & helpers
├── models/              # Pre-trained MTCNN detector weights
├── scripts/             # Dataset alignment utilities
├── images/              # Sample/reference images
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/ajay062002/attendance-management-system.git
cd attendance-management-system
pip install -r requirements.txt
python main.py
```

## Tech Stack

Python · FaceNet · MTCNN · OpenCV · Tkinter · NumPy
