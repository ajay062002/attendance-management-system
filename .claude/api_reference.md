# Internal API Reference

## final_sotware.py

### `dataset_creation(parameters) -> int`
Creates a face image dataset using webcam or video.

**Parameters tuple:** `(path1, webcam, face_dim, gpu, username, vid_path)`
| Param | Default | Description |
|-------|---------|-------------|
| `path1` | `'output'` | Output folder path |
| `webcam` | `(640, 480)` | Webcam resolution string e.g. `'640x480'` |
| `face_dim` | `(160, 160)` | Face crop size e.g. `'160x160'` |
| `gpu` | `0.8` | GPU memory fraction `0.0–1.0` |
| `username` | `'person1'` | Person's name (becomes folder name) |
| `vid_path` | `''` (webcam) | Path to video file or empty for webcam |

**Returns:** `1` on success

**Controls:** Press **S** to save face, **Q** to quit current user session

---

### `train(parameters) -> int`
Generates FaceNet embeddings and trains an SVM classifier.

**Parameters tuple:** `(path1, path2, batch, img_dim, gpu, svm_name, split_percent, split_data)`
| Param | Default | Description |
|-------|---------|-------------|
| `path1` | `'output'` | Dataset folder |
| `path2` | `'20180402-114759/20180402-114759.pb'` | FaceNet model folder |
| `batch` | `90` | Batch size for embedding generation |
| `img_dim` | `160` | Input image size (square) |
| `gpu` | `0.8` | GPU memory fraction |
| `svm_name` | `'classifier.pkl'` | Output classifier filename |
| `split_percent` | `70` | Training split percentage |
| `split_data` | `'y'` | `'y'` to split, `''` to use all for training |

**Returns:** `1` on success

---

### `test(parameters, classifier_filename, emb_array, labels_test, model, batch_size, image_size) -> int`
Tests a trained classifier against a dataset.

**Parameters tuple:** `(path1, path2, path3, batch_size, img_dim, gpu)`
- When called from `train()`, `classifier_filename`, `emb_array`, `labels_test`, `model`, `batch_size`, `image_size` are passed directly
- When called from UI, only `parameters` is used and the rest default to empty

**Returns:** `1` on success

---

### `recognize(mode, parameters) -> str`
Runs real-time face recognition and marks attendance.

**mode:** `'w'` (webcam) | `'v'` (video) | `'i'` (images)

**Parameters tuple:** `(path1, path2, face_dim, gpu, thresh1, thresh2, resolution, img_path, out_img_path, vid_path, vid_save, vid_see)`
| Param | Default | Description |
|-------|---------|-------------|
| `path1` | `'classifier.pkl'` | Path to SVM classifier |
| `path2` | `'20180402-114759/20180402-114759.pb'` | FaceNet model folder |
| `face_dim` | `(160, 160)` | Face alignment size |
| `gpu` | `0.8` | GPU memory fraction |
| `thresh1` | `0.8` | MTCNN detection threshold (onet stage) |
| `thresh2` | `0.50` | SVM confidence threshold for marking present |
| `resolution` | `'640x480'` | Webcam resolution |
| `img_path` | — | Input image folder (mode `'i'`) |
| `out_img_path` | — | Output image folder (mode `'i'`) |
| `vid_path` | — | Input video path (mode `'v'`) |
| `vid_save` | `''` | `'y'` to save output video |
| `vid_see` | `''` | `'y'` to display output (non-webcam) |

**Returns:** Comma-separated string of recognized student names

---

### `get_embeddings(model, paths, batch_size, image_size, gpu_fraction) -> np.ndarray`
Generates FaceNet 512-d embeddings for a list of image paths.

**Returns:** `np.ndarray` of shape `(n_images, 512)`

---

## sheet.py

### `mark_present(st_name: str)`
Marks attendance in `attendance/SAMPLE.xls`.

- Reads `output/` folder to get the full student roster
- Writes `'P'` for each name found in `st_name`, `'A'` otherwise
- Creates the xls file if it doesn't exist
- **Requires:** `output/` and `attendance/` directories to exist

---

## face_aligner.py

### `FaceAligner(desiredLeftEye, desiredFaceWidth, desiredFaceHeight)`
Affine alignment using MTCNN facial landmark points.

### `FaceAligner.align(image, points) -> np.ndarray`
- `image`: BGR image array
- `points`: MTCNN landmark array `(10,)` — first 5 are x-coords, next 5 are y-coords
- **Returns:** aligned face crop as numpy array
