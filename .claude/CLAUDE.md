# Automated Attendance System — Claude Context

## Project Overview

A facial recognition-based attendance management system. Users register faces via webcam or video, train a FaceNet+SVM classifier, then run live recognition to automatically mark student attendance in an Excel sheet.

**Entry point:** `python user_interface.py`

---

## Architecture

```
user_interface.py        Tkinter GUI — all 4 workflow screens
final_sotware.py         Core ML logic (dataset creation, training, testing, recognition)
sheet.py                 Excel attendance writer (xlwt/xlrd/xlsxwriter)
facenet.py               FaceNet model utilities (embeddings, dataset loading)
detect_face.py           MTCNN face detection (pnet/rnet/onet)
face_aligner.py          Affine face alignment using facial landmarks
```

### ML Pipeline

```
Webcam/Video/Images
       ↓
MTCNN (detect_face.py)      → bounding boxes + 5 facial landmarks
       ↓
FaceAligner (face_aligner.py) → aligned, cropped face (default 160×160)
       ↓
FaceNet (20180402-114759.pb)  → 512-d embedding vector
       ↓
SVM Classifier (classifier.pkl) → person name + confidence score
       ↓
sheet.py → mark_present()    → attendance/SAMPLE.xls
```

---

## Key Files

| File | Purpose |
|------|---------|
| `user_interface.py` | All Tkinter windows: welcome → main menu → CREATE/TRAIN/TEST/RUN screens |
| `final_sotware.py` | `dataset_creation()`, `train()`, `test()`, `recognize()`, `get_embeddings()` |
| `sheet.py` | `mark_present(st_name)` — reads `output/` dirs for student names, writes P/A to xls |
| `facenet.py` | `load_model()`, `get_dataset()`, `get_image_paths_and_labels()`, `load_data()` |
| `detect_face.py` | `create_mtcnn()`, `detect_face()` — wraps TF1 MTCNN three-stage pipeline |
| `face_aligner.py` | `FaceAligner.align()` — affine transform using eye landmarks |
| `20180402-114759/` | Pre-trained FaceNet `.pb` model directory (must be present at root) |
| `output/` | Dataset directory — one subfolder per person, containing aligned face images |
| `attendance/` | Output Excel files (e.g. `SAMPLE.xls`) |

---

## Four Workflows (UI Screens)

### 1. CREATE Dataset (`show_create`)
- Captures face images from webcam or video using MTCNN
- Press **S** to save a detected face frame, **Q** to finish current user
- Images saved to `output/<username>/` and auto-resized to 160×160
- Parameters: output path, webcam resolution, GPU fraction, face size, username, video path

### 2. TRAIN Classifier (`show_train`)
- Loads face images from `output/`, generates FaceNet embeddings in batches
- Trains a linear SVM (`sklearn.svm.SVC`) on embeddings
- Saves model as `<name>.pkl` (default: `classifier.pkl`)
- Optional train/test split (default 70%)
- Parameters: dataset path, FaceNet model path, GPU fraction, batch size, image size, SVM name, split %

### 3. TEST Classifier (`show_test`)
- Loads existing `classifier.pkl` and a test dataset folder
- Runs embeddings through SVM, prints accuracy
- Parameters: classifier path, FaceNet model path, dataset path, batch size, image size

### 4. RUN Recognition + Mark Attendance (`show_run`)
- Supports three input modes: **Webcam** (W), **Video** (V), **Images** (I)
- Runs MTCNN + FaceNet + SVM in real-time, draws bounding boxes and names on frame
- Calls `mark_present()` for each recognized face above the confidence threshold
- Parameters: classifier path, FaceNet model path, face size, GPU fraction, MTCNN threshold, classifier threshold, resolution

---

## Important Implementation Details

- Uses **TensorFlow 1.x** (`tensorflow.compat.v1` with `tf.disable_v2_behavior()`)
- Two separate `tf.Graph()` instances in `recognize()`: `g1` for FaceNet, `g2` for MTCNN — avoids graph conflicts
- Default GPU memory fraction: **0.8** (all entry fields default to 0.8 if left blank)
- Default MTCNN thresholds: `[0.6, 0.7, 0.8]` — only the third (onet) is user-configurable
- Default classifier confidence threshold: **0.50**
- `sheet.py` hardcodes subject as `'SAMPLE'` — the attendance file is always `attendance/SAMPLE.xls`
- `mark_present()` reads `output/` folder names as the class list — the `output/` directory must exist
- The `attendance/` directory must exist before running recognition (not auto-created)
- `sheet.py` has a stray `mark_present(st_name)` call at module level (line 44) — it runs on import

---

## Known Issues / Gotchas

- **Typo in filename:** Core logic file is `final_sotware.py` (not `software`) — do not rename without updating `user_interface.py` import
- **`sheet.py` auto-executes on import** — the `mark_present(st_name)` call at line 44 runs when the module is imported, which can fail if `output/` or `attendance/` don't exist
- **`bottom_frame` is `None`** in `show()` — `tkinter.Frame(...).pack()` returns `None`, so buttons placed on `bottom_frame` fall back to the root window
- **`final_sotware.py` calls `main()`** in `__main__` block but `main()` is never defined
- **Mixed MTCNN usage:** `dataset_creation` uses `mtcnn.mtcnn.MTCNN` (high-level), while `recognize` uses the lower-level `detect_face.detect_face()` with pnet/rnet/onet
- **`align_dataset_mtcnn.py`** — standalone script for bulk alignment, separate from the UI

---

## Dependencies

```
tensorflow          (TF1 compat mode via tf.compat.v1)
opencv-python       (cv2 — webcam, video, image I/O)
mtcnn               (high-level MTCNN for dataset creation)
scikit-learn        (SVM classifier)
pillow              (PIL — image resizing)
xlwt / xlrd / xlutils / xlsxwriter  (Excel attendance files)
scipy               (misc utilities)
```

Install: `pip install -r requirements.txt`

---

## Directory Structure

```
.
├── user_interface.py       GUI entry point
├── final_sotware.py        ML core
├── sheet.py                Attendance Excel writer
├── facenet.py              FaceNet utilities
├── detect_face.py          MTCNN detection
├── face_aligner.py         Affine alignment
├── align_dataset_mtcnn.py  Bulk dataset alignment script
├── resizer.py              Image resize utility
├── tkinter_custom_button.py Custom button widget
├── requirements.txt
├── 20180402-114759/        Pre-trained FaceNet model (required)
├── output/                 Face image dataset (one folder per person)
├── attendance/             Excel attendance files (must exist)
├── output_videos/          Saved output videos
└── images/                 Misc images
```
