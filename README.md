# Attendance Management System

A desktop face recognition system that automatically identifies students from a webcam feed and records their attendance in an Excel file. Built on a full ML pipeline — from dataset collection through model training to real-time inference.

---

## What it does

- Admin registers students by capturing ~100 face images per person via webcam
- System trains a face recognition model on those images
- At attendance time, the webcam feed is analysed in real time — faces are detected, identified, and attendance is logged automatically to an Excel file
- Works under varied lighting conditions
- Tkinter GUI for all interactions (dataset creation, training, live recognition)

---

## ML Pipeline

```
DATASET CREATION
  Tkinter GUI → OpenCV webcam → MTCNN detects + crops face → saved per student

MODEL TRAINING
  Saved face images → FaceNet (pretrained CNN) → 128-d embedding vector per face
  All embeddings + labels → SVM classifier trained → model saved to disk

REAL-TIME RECOGNITION
  Webcam frame → MTCNN detects face → FaceNet embeds it
  → SVM predicts identity (+ confidence score)
  → If confidence > threshold: log name + timestamp to Excel
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3 |
| Face detection | MTCNN |
| Face recognition | FaceNet (TensorFlow / PyTorch) |
| Classifier | SVM (scikit-learn) |
| Video input | OpenCV |
| GUI | Tkinter |
| Attendance output | openpyxl (Excel) |

---

## Project Structure

```
attendance-management-system/
├── main.py                    # Entry point — launches Tkinter app
├── src/
│   ├── face_detection/
│   │   ├── detector.py        # MTCNN face detection + crop/align
│   │   └── embedder.py        # FaceNet embedding generation
│   ├── ui/
│   │   ├── app.py             # Main Tkinter window
│   │   ├── register.py        # Dataset capture screen
│   │   └── attendance.py      # Live recognition screen
│   └── utils/
│       ├── dataset.py         # Image I/O, dataset management
│       ├── trainer.py         # SVM training on embeddings
│       └── excel.py           # Attendance Excel read/write
├── models/                    # MTCNN weights (.npy files)
├── dataset/                   # Captured face images per student
├── attendance/                # Generated Excel files
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/ajay062002/attendance-management-system.git
cd attendance-management-system
pip install -r requirements.txt
python main.py
```

---

## Usage

1. **Register students** — enter name, click Capture, look at camera for ~5 seconds
2. **Train model** — click Train after all students are registered (takes ~30 seconds)
3. **Start attendance** — click Start Recognition, system logs faces it identifies automatically

---

## Requirements

```
opencv-python
mtcnn
tensorflow
scikit-learn
Pillow
openpyxl
numpy
```

---

## Source

- Web-based interface instead of Tkinter desktop GUI
- Upgrade to TensorFlow 2.x with Keras
- Support for multiple classrooms and subjects
- Email or SMS notification on attendance completion
- Dashboard with attendance analytics and visual reports

---

## 📄 Research Reference

Based on the paper:
[Automated Attendance System using CNN — IEEE](https://ieeexplore.ieee.org/document/9029001)

Related publication by the author:
[Sign Language Detection using CNN — IEEE ICSSCS 2023](https://ieeexplore.ieee.org/document/10169225)

---

## ⬇️ Download Pre-trained Model

[FaceNet Model — Google Drive](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-)

---

## 👤 Author

**Ajay Thota**
- GitHub: [@ajay062002](https://github.com/ajay062002)
- Portfolio: [ajaylive.com](https://ajaylive.com)
