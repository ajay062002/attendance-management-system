# 🎓 Automated Attendance Management System using CNN

An end-to-end face recognition system that automatically detects students from a webcam or video feed and marks their attendance in an Excel sheet — no manual roll call needed.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

---

## ✨ Features

- Real-time face detection and recognition via webcam
- Supports video file and image input modes
- GUI-based workflow — no command line needed
- Automatic Excel attendance sheet generation
- Robust against varied lighting, partial occlusion, and facial orientations
- Batch dataset creation and one-click training

---

## 🏗️ Architecture

### ML Pipeline

```
Webcam / Video / Images
          │
          ▼
    ┌─────────────┐
    │    MTCNN    │  ← Face Detection
    │  (3-stage)  │     Finds face in frame
    └──────┬──────┘     Returns bounding box + landmarks
           │
           ▼
    ┌─────────────┐
    │  FaceAligner│  ← Affine alignment using eye landmarks
    └──────┬──────┘     Produces clean 160×160 face crop
           │
           ▼
    ┌─────────────┐
    │   FaceNet   │  ← Embedding Extractor
    │ (.pb model) │     Converts face → 512-d vector fingerprint
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │     SVM     │  ← Classifier
    │ (sklearn)   │     Matches fingerprint → person name
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   sheet.py  │  ← Attendance Writer
    └─────────────┘     Marks student Present in Excel
```

### System Flow — Two Phases

```
━━━━━━━━━━━━ TRAINING PHASE ━━━━━━━━━━━━

  Collect face photos (via webcam/video)
             │
             ▼
    MTCNN detects + crops faces
             │
             ▼
    FaceNet generates embeddings
             │
             ▼
    SVM trained on labeled embeddings
             │
             ▼
    classifier.pkl saved to disk


━━━━━━━━━━━━ INFERENCE PHASE ━━━━━━━━━━━━

  Live webcam feed starts
             │
             ▼
    OpenCV reads frame by frame
             │
             ▼
    MTCNN detects face in frame
             │
             ▼
    FaceNet generates embedding
             │
             ▼
    SVM predicts identity
             │
             ▼
    Name + confidence displayed on frame
             │
             ▼
    Attendance marked in Excel sheet
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | Python 3 | Core logic |
| Deep Learning | TensorFlow 1.x | FaceNet + MTCNN backbone |
| Face Detection | MTCNN | Detect and align faces in frames |
| Face Embedding | FaceNet (20180402-114759) | Convert face → 512-d vector |
| Classifier | Scikit-learn SVM | Match embeddings to identities |
| Video Processing | OpenCV | Webcam/video frame handling |
| GUI | Tkinter | Desktop interface |
| Attendance Output | xlwt / xlrd / xlsxwriter | Excel sheet generation |
| Image Processing | Pillow (PIL) | Image resizing and handling |

---

## 📁 Project Structure

```
attendance-management-system/
├── main.py                        # Main GUI entry point (run this)
├── src/
│   ├── face_detection/
│   │   ├── detect_face.py         # MTCNN face detection (pnet/rnet/onet)
│   │   ├── face_aligner.py        # Affine face alignment using landmarks
│   │   ├── face_detect.py         # Face detection utilities
│   │   └── facenet.py             # FaceNet model utilities
│   ├── ui/
│   │   ├── user_interface.py      # Tkinter GUI components
│   │   └── tkinter_custom_button.py  # Custom button widget
│   └── utils/
│       ├── sheet.py               # Excel attendance writer
│       └── resizer.py             # Image resize utility
├── models/
│   ├── det1.npy                   # MTCNN PNet weights
│   ├── det2.npy                   # MTCNN RNet weights
│   └── det3.npy                   # MTCNN ONet weights
├── scripts/
│   └── align_dataset_mtcnn.py     # Bulk dataset alignment script
├── images/                        # Sample/reference images
├── requirements.txt
└── README.md
```

> ⚠️ The `20180402-114759/` pre-trained FaceNet model is not included due to its large size. Download it from the link below and place it in the root directory.

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.6
- Conda environment (strongly recommended — TF1 compatibility)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/ajay062002/attendance-management-system.git
cd attendance-management-system

# 2. Create conda environment with Python 3.6
conda create -n attendance python=3.6
conda activate attendance

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the pre-trained FaceNet model
# Link: https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-
# Extract and place the folder as: 20180402-114759/

# 5. Create required directories manually
mkdir output
mkdir attendance

# 6. Run the application
python main.py
```

---

## 🚀 How to Use

### Step 1 — Create Dataset
1. Run `python main.py`
2. Click **Create**
3. Enter a student name in the username field
4. Select **Webcam** mode
5. Click **Continue** — webcam will open
6. Press **S** to save face frames (aim for 80–100 images per student)
7. Press **Q** when done
8. Repeat for each student

### Step 2 — Train the Model
1. Click **Train** in the GUI
2. Training runs automatically — may take a few minutes
3. A `classifier.pkl` file is generated when complete

### Step 3 — Mark Attendance
1. Click **Run** in the GUI
2. Select **Webcam** from the input options
3. Click **Mark Attendance**
4. The system recognizes faces in real time and displays names with confidence scores
5. Excel attendance sheet is auto-saved in the `attendance/` folder with current date/time

---

## 📊 Model Details

**MTCNN** runs three cascaded neural networks — PNet (proposal), RNet (refine), ONet (output) — to progressively localize and align faces with high precision even under occlusion and varied orientations.

**FaceNet** uses a pre-trained Inception-ResNet model to produce a 512-dimensional embedding for each face. The embedding space is trained so that faces of the same person cluster together while different people are far apart (triplet loss training).

**SVM** (Support Vector Machine) is trained on these embeddings. It finds the optimal hyperplane between each student's embedding cluster and classifies new embeddings accordingly. Default confidence threshold is **0.50** — faces below this are marked Unknown.

---

## 🔮 Future Improvements

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
