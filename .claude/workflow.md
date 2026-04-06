# Operational Workflow

## 1. Dataset Creation
- Launch `user_interface.py`.
- Click **CREATE**.
- Enter the person's name.
- Select source (Webcam/Video).
- In the camera view, press **'s'** to save face samples (80-100 images per person recommended).
- Press **'q'** to finish.

## 2. Training
- Launch `user_interface.py`.
- Click **TRAIN**.
- Specify paths (default paths are usually pre-filled).
- Once completed, a `classifier.pkl` file is generated.

## 3. Running Recognition
- Launch `user_interface.py`.
- Click **RUN** or **Mark Attendance**.
- The system will start detecting and recognizing faces in real-time.
- Recognized individuals will have their status updated to "P" (Present) in the attendance sheet (`attendance/SAMPLE.xls`).

## 4. Attendance Sheets
- Located in the `attendance/` directory.
- Format: `.xls`.
- Includes a list of names with status (P/A) and timestamp.
