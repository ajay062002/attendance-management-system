# Common Workflows & Task Patterns

## Running the Application

```bash
python user_interface.py
```

Sequence: Welcome screen → Continue → Main menu → Choose workflow

---

## Adding a New Student

1. Run the app, click **CREATE**
2. Enter username (e.g. `john_doe`)
3. Point webcam at face — press **S** to capture, **Q** when done
4. Images saved to `output/john_doe/`
5. Re-train the classifier (**TRAIN**) to include the new person

---

## Full Setup from Scratch

```
1. CREATE  →  capture face images for each student
2. TRAIN   →  generate embeddings + fit SVM → classifier.pkl
3. RUN     →  load classifier.pkl, run webcam recognition, mark attendance
```

---

## Modifying Attendance Logic

All attendance writing is in `sheet.py:mark_present()`.
- Subject name is hardcoded as `'SAMPLE'` — change `sub` variable to parameterize it
- The function reads `output/` directory names as the student roster
- Writes `P` (present) or `A` (absent) based on whether the name appears in `st_name` string

## Changing the Confidence Threshold

In the RUN screen, the **"threshold to consider face is recognised"** field sets `classifier_threshold` in `final_sotware.py:recognize()` (default 0.50). Lower = more detections, higher = fewer false positives.

## Adding a New UI Screen

1. Define a new `show_X()` function in `user_interface.py` following existing pattern
2. Add a button in `show()` (the main menu)
3. Add HOME button calling `gotohome()` which calls `show()`

---

## Debugging Recognition Issues

- Check that `output/` directory exists and has one subfolder per enrolled person
- Check that `attendance/` directory exists (not auto-created)
- Check `classifier.pkl` is present in working directory or provide full path
- Check that `20180402-114759/20180402-114759.pb` model file is present
- Lower MTCNN threshold (field 1) if faces aren't being detected
- Lower classifier threshold (field 2) if recognized people aren't being marked
