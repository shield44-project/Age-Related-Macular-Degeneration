# AMD Detection System Project Notes

## Project Summary

This project is an Age-Related Macular Degeneration (AMD) screening application. It combines a Qt C++ desktop GUI, a Flask Python backend, a PyTorch/timm deep learning model, image preprocessing utilities, explainability heatmaps, and a SQLite database for patient scan history.

The user workflow is:

1. The user selects a retinal fundus image in the Qt GUI.
2. The GUI sends the image and patient details to the Python backend.
3. The backend validates that the image looks like a fundus photograph.
4. The image is preprocessed into the format expected by the model.
5. The model predicts either `AMD` or `Normal`.
6. A CAM/saliency image is generated to show visually important retinal regions.
7. The scan result is saved in SQLite and returned to the GUI.
8. The GUI displays the original fundus image, CAM image, diagnosis, confidence, and project metrics.

## Project Structure

```text
Age-Related-Macular-Degeneration/
├── CMakeLists.txt
├── README.md
├── PROJECT.md
├── TRAINING.md
├── requirements.txt
├── src/
│   └── main.cpp
├── backend/
│   ├── __init__.py
│   ├── __main__.py
│   ├── api.py
│   ├── database.py
│   ├── dl_model.py
│   ├── preprocessing.py
│   ├── server.py
│   ├── train_and_evaluate.py
│   └── models/
│       └── ViT_base/
├── docs/
│   └── pylibs-docs/
├── runtime/
│   ├── uploads/
│   ├── cams/
│   └── patient_records.db
├── build/
└── venv/ or .venv/
```

### Important Files And Folders

| Path | Purpose |
| --- | --- |
| `CMakeLists.txt` | Builds the Qt C++ desktop GUI with CMake. |
| `src/main.cpp` | Main Qt Widgets frontend, including upload UI, results UI, history table, theme handling, and backend HTTP calls. |
| `backend/api.py` | Flask API layer. Receives image uploads, calls preprocessing/model code, saves records, and returns JSON responses. |
| `backend/preprocessing.py` | Decodes images, validates fundus images, applies CLAHE, crops, resizes, normalizes, and prepares tensors. |
| `backend/dl_model.py` | Loads model checkpoints, performs prediction, manages backup inference, and creates explainability outputs. |
| `backend/database.py` | SQLite database layer for patient records. |
| `backend/server.py` | Backend server entry point. |
| `backend/train_and_evaluate.py` | Training/evaluation script for improving or retraining the AMD model. |
| `backend/models/` | Stores trained model checkpoint files such as `.pth` or `.pt`. |
| `requirements.txt` | Python package dependencies for backend, model inference, preprocessing, and training. |
| `runtime/uploads/` | Generated folder containing saved uploaded fundus images. |
| `runtime/cams/` | Generated folder containing CAM/saliency image outputs. |
| `runtime/patient_records.db` | Generated SQLite database file for patient scan history. |
| `build/` | Generated CMake/build output folder. |
| `venv/` or `.venv/` | Local Python virtual environment. |

### Main Runtime Flow By File

```text
src/main.cpp
  -> sends image and patient form data to POST /predict

backend/api.py
  -> validates request
  -> saves uploaded image
  -> calls preprocessing.py
  -> calls dl_model.py
  -> calls database.py
  -> returns JSON response to GUI

src/main.cpp
  -> reads JSON response
  -> displays diagnosis, metrics, original image, CAM image, and history
```

## Reported Classification Metrics

The GUI and API report these project metrics:

| Metric | Value |
| --- | ---: |
| Accuracy | 80.0% |
| Sensitivity | 79.0% |
| Specificity | 83.0% |
| Precision | 95.0% |
| F1 Score | 86.4% |

## Backend Tools

### Flask

Flask provides the HTTP API used by the GUI. The main backend file is `backend/api.py`.

Main endpoints:

- `GET /health`: returns backend status, active model information, and model metrics.
- `POST /predict`: accepts an uploaded image and patient details, runs prediction, saves the record, and returns the diagnosis.
- `GET /patients`: returns saved patient records.
- `GET /patients/<id>`: returns one saved record.
- `DELETE /patients/<id>`: deletes one record.

How it works with images:

- The GUI sends images as multipart form-data with the field name `image`.
- Flask reads the uploaded image bytes.
- The backend saves a PNG copy under `runtime/uploads`.
- The same image bytes are passed into preprocessing and prediction.
- The generated saliency/CAM image is saved under `runtime/cams`.

### Pillow

Pillow is used as a reliable image fallback and for saving decoded images.

In this project:

- It converts uploaded image bytes into RGB when OpenCV cannot decode an image variant.
- It saves validated uploads as PNG files in `runtime/uploads`.

### OpenCV

OpenCV is used for fundus validation and preprocessing.

It handles:

- Image decoding from raw bytes.
- BGR/RGB color conversion.
- Grayscale conversion.
- Brightness and color checks for fundus validation.
- CLAHE contrast enhancement.
- Center cropping, resizing, and border padding.

How fundus validation works:

- The backend checks whether image corners are dark, matching the black circular background common in fundus photography.
- It checks that the center is brighter than the corners.
- It checks that the bright retinal region has reasonable coverage.
- It checks that red/warm tones dominate the bright region, which is expected in retinal images.

This prevents clearly invalid images from being sent to the model.

### NumPy

NumPy is used for image array manipulation.

It handles:

- Converting image bytes into numeric arrays.
- Computing brightness statistics.
- Creating masks for bright retinal regions.
- Normalizing pixel values.
- Reordering image dimensions from image format `(H, W, C)` to model format `(C, H, W)`.

### PyTorch

PyTorch runs model inference.

It handles:

- Loading `.pth` or `.pt` checkpoint files.
- Moving model tensors to CPU or CUDA when available.
- Running the neural network in evaluation mode.
- Producing class probabilities for `Normal` and `AMD`.

The backend automatically uses CUDA if PyTorch detects a compatible GPU.

### timm

`timm` provides the Vision Transformer architecture used by the project.

The model loader can use:

- A legacy binary ViT classifier head.
- A two-class ViT classifier head.
- A timm architecture stored in checkpoint metadata.

This makes model loading more portable because different trained checkpoint formats can still be loaded.

### Flask-CORS

Flask-CORS allows browser or external clients to access the API if needed. The current Qt GUI talks directly to `127.0.0.1:5000`, but CORS support keeps the backend usable for future web frontends.

## Image Processing Pipeline

The preprocessing code is in `backend/preprocessing.py`.

### 1. Decode Image

Uploaded bytes are decoded into a BGR image using OpenCV. If OpenCV fails, Pillow attempts to decode and convert the image to RGB, then OpenCV converts it to BGR.

### 2. Validate Fundus Image

The validation step rejects images that do not match fundus-photo structure. It checks:

- Minimum size.
- Dark camera-background corners.
- Brighter center retina region.
- Reasonable bright-pixel coverage.
- Red channel dominance in the retinal region.

### 3. CLAHE Enhancement

CLAHE improves local contrast in the image. This helps retinal vessels, lesions, and texture patterns become more visible before model inference.

### 4. Center Crop

The image is cropped around the center because fundus photographs usually contain the retina near the center. Reflection padding is used when the image is smaller than the crop size.

### 5. Resize

The image is resized to `224 x 224`, matching the Vision Transformer input size.

### 6. Normalize

Pixel values are scaled to `0.0-1.0` and normalized using ImageNet mean and standard deviation values. This matches the preprocessing expected by common pretrained ViT backbones.

### 7. Tensor Shape

The final image tensor is shaped as:

```text
1 x 3 x 224 x 224
```

That means:

- `1`: batch size
- `3`: color channels
- `224 x 224`: image size

## Model Tools and Behavior

The model logic is in `backend/dl_model.py`.

### Model Loading

The backend searches for model checkpoints in this order:

1. `MODEL_PATH`, `MODEL_PATH_2`, or `MODEL_PATHS` environment variables.
2. `backend/models/ViT_base/best_vit_model_improved.pth`.
3. `backend/models/ViT_base/best_vit_model.pth`.
4. Any `.pth` or `.pt` file under `backend/models`.

This lets the project work on different PCs without hardcoded absolute paths.

### Model Type

The main model is a Vision Transformer classifier. Vision Transformers are strong for medical imaging because they can learn both local image features and wider spatial relationships across the retina.

The classifier predicts:

- `Normal`
- `AMD`

### Backup Mode

If PyTorch, timm, or a usable checkpoint is unavailable, the backend falls back to heuristic inference. Backup mode keeps the demo functional, but it is not a replacement for the trained deep learning model.

### CAM / Saliency Output

The backend generates a visual explanation image after prediction.

The CAM image helps users see which retinal areas influenced the prediction. This is useful for clinical decision-support interfaces because it makes the result more interpretable than a label alone.

## Why the Model Is Good

The project uses a Vision Transformer-based AMD classifier, which is a strong architecture choice for retinal fundus analysis.

Strengths:

- It uses a modern deep learning architecture suitable for image classification.
- It uses standardized `224 x 224` preprocessing and ImageNet normalization.
- It supports real trained checkpoints instead of only rule-based classification.
- It validates fundus images before prediction, reducing invalid input errors.
- It provides visual saliency maps so predictions are easier to interpret.
- It stores patient history, making the system useful beyond a single prediction.
- It reports clinically relevant metrics: accuracy, sensitivity, specificity, precision, and F1 score.

The current reported project metrics are:

- Accuracy: 80.0%
- Sensitivity: 79.0%
- Specificity: 83.0%
- Precision: 95.0%
- F1 Score: 86.4%

High precision is especially useful in this project because an AMD prediction should be reliable when the system flags a scan as AMD.

## How To Improve The Model

The model can be improved in several practical ways:

### Use More Data

Train on more retinal fundus datasets from different cameras, hospitals, lighting conditions, and patient groups. This improves generalization.

### Improve Class Balance

AMD datasets can be imbalanced. Use class weighting, balanced sampling, or targeted augmentation so the model learns both AMD and Normal cases well.

### Add Better Augmentation

Useful augmentations include:

- Small rotations.
- Brightness and contrast shifts.
- Mild blur.
- Crop and scale variation.
- Color jitter within medically reasonable limits.

These help the model handle real-world image variation.

### Use Stronger Architectures

Possible upgrades:

- Swin Transformer.
- ConvNeXt.
- EfficientNetV2.
- Ensemble of ViT and CNN models.

An ensemble can improve stability by averaging predictions from multiple models.

### Train With Cleaner Labels

Model quality depends heavily on label quality. Expert-reviewed labels and removal of noisy or ambiguous images can improve performance.

### Add Severity Classes

The current task is binary: `AMD` or `Normal`. A future model could classify:

- Normal
- Early AMD
- Intermediate AMD
- Late AMD

This would make the system more clinically useful.

### Add External Validation

Evaluate the model on a dataset that was not used during training. This is important because high training or validation performance does not always transfer to new hospitals or cameras.

### Calibrate Confidence

Use calibration methods such as temperature scaling so confidence scores better match real-world correctness.

### Improve Explainability

Use Grad-CAM, attention rollout, or lesion segmentation overlays to make the visual explanation more clinically meaningful.

## Database Tools

The database code is in `backend/database.py`.

### SQLite

SQLite stores patient scan records locally in:

```text
runtime/patient_records.db
```

SQLite is a good fit for this project because:

- It requires no separate database server.
- It is easy to ship with a desktop application.
- It stores records persistently between app runs.
- It works on Linux and Windows.

### Patient Table

The database stores:

- `id`: auto-generated record ID.
- `name`: patient name.
- `age`: patient age.
- `image_path`: saved uploaded fundus image path.
- `prediction`: `AMD` or `Normal`.
- `confidence`: model confidence score.
- `date`: scan date.

### Database Operations

The backend supports:

- Initializing the database table.
- Inserting a new scan record.
- Reading all records.
- Reading one record by ID.
- Deleting one record.
- Clearing all records.

How it works with images:

- The actual image files are stored on disk in `runtime/uploads`.
- The database stores the image file path.
- The GUI uses that path to reload historical fundus images.
- CAM images are stored separately in `runtime/cams`.

## Frontend Tools

The frontend is a Qt C++ desktop GUI in `src/main.cpp`.

### Qt Widgets

Qt Widgets provides the desktop interface.

The GUI includes:

- Patient name input.
- Patient age input.
- Upload button.
- Fundus image preview.
- CAM image preview.
- Diagnosis badge.
- Confidence bar.
- Metric display area.
- History table.
- Theme toggle.

### Qt Network

Qt Network sends HTTP requests from the GUI to the backend.

It handles:

- `GET /health` to check backend status.
- `POST /predict` to upload images.
- `GET /patients` to load history.
- `DELETE /patients/<id>` to remove records.

### Qt Image Tools

Qt uses `QPixmap` and `QLabel` to display images.

How images are displayed:

- The selected fundus image is shown immediately in the Fundus panel.
- The backend-generated CAM image path is returned in the prediction response.
- The GUI loads that CAM path and displays it in the CAM panel.
- Historical records can reload saved images from disk.

### Qt Process

`QProcess` starts the backend automatically if it is offline.

It detects Python from:

- `.venv/Scripts/python.exe` on Windows.
- `venv/Scripts/python.exe` on Windows.
- `.venv/bin/python` on Linux.
- `venv/bin/python` on Linux.
- `py -3` on Windows as a fallback.
- `python3` on Linux as a fallback.

### Qt Settings

`QSettings` stores user preferences such as dark mode or light mode.

## Build Tools

### CMake

CMake builds the Qt C++ GUI.

It now supports:

- Qt5.
- Qt6.
- Linux builds.
- Windows Visual Studio builds.
- Optional Qt DLL deployment with `windeployqt`.

### g++

The project can also be compiled directly with g++ on Linux:

```bash
g++ -fPIC src/main.cpp -o build/amd_gui `pkg-config --cflags --libs Qt5Widgets Qt5Network` -std=c++17
```

## Runtime Folders

The application uses these runtime folders:

```text
runtime/uploads   saved uploaded fundus images
runtime/cams      generated CAM/saliency images
runtime/patient_records.db   SQLite patient history database
```

These files are generated while using the app and are not part of the source code logic.

## Dependency Catalog

This section explains every major dependency that appears in `requirements.txt` or in the C++ build.

### Python Runtime and API Dependencies

#### Flask

- Used in `backend/api.py` and `backend/server.py`.
- Provides the local HTTP server that the Qt GUI talks to.
- Exposes the REST endpoints used for health checks, prediction, record listing, and record deletion.
- Handles multipart form-data uploads for retinal images.
- Serializes Python dictionaries into JSON responses for the GUI.

#### flask-cors

- Present in dependencies for future browser-based use.
- Not central to the current Qt-only desktop workflow.
- Useful if the backend is later reused by a web dashboard or React/Vue frontend.

#### requests

- Present in dependencies for HTTP client scripting and testing.
- Not required by the current Qt GUI because Qt uses `QNetworkAccessManager`.
- Useful for standalone API checks, automated smoke tests, or future integration scripts.

### Deep Learning and Numeric Dependencies

#### torch

- Core tensor library used for inference and training.
- Loads `.pth` / `.pt` model checkpoints.
- Runs the classifier on CPU or CUDA.
- Supports gradient-based saliency generation for CAM output.
- Supports AMP, gradient clipping, and optimizer steps in training.

#### torchvision

- Used mainly in `backend/train_and_evaluate.py`.
- Supplies `datasets.ImageFolder` for dataset loading.
- Supplies transform pipelines for resize, crop, normalization, blur, and augmentation.

#### timm

- Supplies the model backbones.
- Used both for inference-time architecture creation and for training-time model creation.
- Supports ConvNeXt, ViT, and Swin style architectures in a unified API.

#### numpy

- Used throughout preprocessing, inference fallback logic, saliency blending, and metric utilities.
- Handles image arrays, channel arithmetic, normalization, masking, and post-processing.

#### scikit-learn

- Used in `backend/train_and_evaluate.py`.
- Computes accuracy, precision, recall, F1, confusion matrix, and ROC AUC during validation/testing.

#### pandas

- Listed in dependencies but not central to the active inference path.
- Likely intended for future data analysis or dataset/report utilities.

#### tqdm

- Used in training loops.
- Shows progress bars for training, validation, and test passes.

### Image Processing Dependencies

#### opencv-python

- Main image processing library for decode, resize, crop support, grayscale conversion, border padding, CLAHE, and color operations.
- Used heavily in `backend/preprocessing.py`.
- Also used in fallback resize paths and BGR/RGB conversions in model-side saliency support.

#### opencv-python-headless

- Listed alongside regular OpenCV.
- Useful in server/headless environments where GUI bindings are unnecessary.
- In many environments you would keep only one OpenCV package to avoid duplication.

#### pillow

- Used as a decode fallback when OpenCV cannot parse an input image.
- Used to save uploaded images and CAM images.
- Used for cv2-free resize fallback in `backend/dl_model.py`.

#### pydicom

- Present in dependencies for medical-image workflows.
- The current GUI/API path is based on common image files such as PNG/JPG, not active DICOM ingestion.
- This package is a likely foundation for future ophthalmic workflow expansion.

#### matplotlib

- Present in dependencies.
- Not directly used in the active runtime path shown in the current code.
- Likely useful for training plots, exploratory analysis, or notebook work.

#### grad-cam

- Present in dependencies.
- The current runtime implementation uses an in-house gradient saliency path rather than importing this package directly.
- It remains relevant if the project later switches to Grad-CAM abstractions for model explanation.

#### albumentations

- Present in dependencies.
- The current training script uses `torchvision.transforms`, not Albumentations.
- Could be adopted later for stronger ophthalmic augmentation pipelines.

## End-to-End Image Lifecycle

This section describes what happens to a retinal image from selection in the GUI until the result appears on screen and gets written to disk.

### Stage 1: User Chooses an Image in Qt

- The user clicks `Upload and Analyse`.
- `src/main.cpp` opens a `QFileDialog`.
- The selected path is immediately previewed in the left analysis image panel using `QPixmap`.
- No backend inference happens yet at this exact sub-step.

### Stage 2: GUI Checks Backend Availability

- The GUI sends `GET /health` to `http://127.0.0.1:5000/health`.
- If the backend is already running, the GUI proceeds directly to prediction.
- If the backend is offline, the GUI attempts to start Python with `QProcess`.

### Stage 3: GUI Uploads the Image

- The GUI creates a `QHttpMultiPart` request.
- The selected file contents are streamed as form-data field `image`.
- Patient name and age are added as text form-data fields.
- The target route is `POST /predict`.

### Stage 4: Flask Receives the Request

- `backend/api.py` reads the multipart image or JSON `image_path`.
- Empty uploads are rejected.
- Invalid paths are rejected.
- The request is normalized so both Qt uploads and JSON clients can use the same inference path.

### Stage 5: Raw Image Decode

- The backend converts incoming bytes into an image array.
- OpenCV attempts `cv2.imdecode`.
- If OpenCV fails, Pillow attempts the decode and then converts RGB to OpenCV-style BGR.
- The output of decode is a `numpy.ndarray` in BGR channel order.

### Stage 6: Fundus Validation

- The decoded BGR image is passed to `is_valid_fundus_image`.
- The validator checks:
- Image size is at least `64 x 64`.
- Corners are dark enough to look like fundus camera background.
- The center is brighter than the corners.
- Bright retinal coverage is neither too tiny nor unrealistically full-frame.
- Red dominates blue in the bright region.
- If validation fails, the API returns HTTP 400 with `invalid_fundus: true`.

### Stage 7: Save Uploaded Image to Disk

- For multipart uploads, the backend creates a timestamped PNG path under `runtime/uploads`.
- Pillow saves the decoded fundus image at that path.
- The saved path becomes the canonical stored `image_path` for history reload and DB persistence.

### Stage 8: Preprocessing for Inference

- The saved image bytes are fed into `preprocess_for_inference`.
- The backend applies CLAHE to boost local contrast.
- The image is center-cropped to keep the retina dominant in frame.
- The crop is resized to `224 x 224`.
- The processed RGB image is preserved for later saliency overlay.
- The model tensor is normalized using ImageNet mean/std and reshaped to `1 x 3 x 224 x 224`.

### Stage 9: Prediction

- `predict_probabilities` is called with the normalized tensor.
- If a real checkpoint is loaded, one or more models produce class probabilities.
- If no checkpoint is available, backup heuristic inference estimates an AMD score using image statistics.
- The output is always a two-value array:
- Probability of `Normal`
- Probability of `AMD`

### Stage 10: Result Label and Confidence

- The backend picks the highest-probability class with `argmax`.
- The predicted class becomes either `Normal` or `AMD`.
- The confidence becomes the winning class probability.

### Stage 11: Explainability CAM / Saliency Image

- `generate_explainability_cam` builds a heatmap.
- In backup mode, the heatmap is based on intensity structure.
- In real-model mode, the backend computes input gradients with respect to the predicted class score.
- The heatmap is converted to a jet-style RGB overlay.
- The overlay is blended with the preprocessed RGB fundus image.
- The blended output is saved to `runtime/cams/<stem>_cam.png`.

### Stage 12: Database Record Creation

- `insert_patient_record` writes one row to SQLite.
- Stored fields include name, age, absolute image path, prediction, confidence, and date.
- This allows the GUI history tab to query and redisplay earlier scans.

### Stage 13: JSON Response to Qt

- The backend returns:
- Patient identity fields
- Absolute image path
- Absolute CAM image path
- Diagnosis / prediction text
- Confidence score
- Fixed reported project metrics
- Model status information
- Class probabilities

### Stage 14: GUI Result Rendering

- The GUI updates the diagnosis badge.
- The GUI updates the risk badge.
- The GUI sets the confidence progress bar and text.
- The GUI loads the CAM image path into `QPixmap`.
- The GUI refreshes patient history by calling `GET /patients`.

## Function Reference: `backend/api.py`

### Module Role

- File type: Flask API layer.
- Core responsibility: accept requests, validate image input, call preprocessing and inference, save records, and return JSON.
- Image ownership: this file decides when uploads are decoded, validated, saved, and routed to preprocessing/model code.

### Global: `app = Flask(__name__)`

- Purpose: create the Flask application object.
- Used by: `backend/server.py`, `backend/__main__.py`, and all route decorators.
- Side effect: registers routes on one process-local app instance.

### Global: `UPLOAD_DIR`

- Value: `runtime/uploads`.
- Purpose: destination directory for uploaded retinal images.
- Image impact: stores original/decoded uploads that are later referenced by the GUI and database.

### Global: `CAMS_DIR`

- Value: `runtime/cams`.
- Purpose: destination directory for generated saliency/CAM overlays.
- Image impact: stores the explanation image shown beside the original fundus scan.

### Global: `PROJECT_METRICS`

- Purpose: fixed reported metrics shown in the GUI and API.
- Current values:
- Accuracy `0.80`
- Sensitivity `0.79`
- Specificity `0.83`
- Precision `0.95`
- F1 score `0.864`

### `index()`

- Route: `GET /`
- Purpose: minimal root endpoint that confirms the backend is alive.
- Input: no payload required.
- Output: JSON with a human-readable message and route list.
- Image interaction: none.
- Database interaction: none.
- When useful: browser check, curl smoke test, or quick GUI troubleshooting.

### `health()`

- Route: `GET /health`
- Purpose: returns model/session status for the GUI status label.
- Input: no payload required.
- Output:
- `status`
- `model_type`
- `model_name`
- `backup_active`
- `model_path`
- `model_paths`
- `model_names`
- `models_loaded`
- `metrics`
- Image interaction: none.
- Database interaction: none.
- GUI usage: polled repeatedly to show whether the backend is online.

### `get_request_image()`

- Purpose: normalize image acquisition across multipart uploads and JSON path-based requests.
- Input mode 1: multipart file field `image`.
- Input mode 2: JSON key `image_path`.
- Output: tuple of `(image_bytes, image_path_string)`.
- Validation performed:
- Rejects empty filename.
- Rejects empty file content.
- Decodes bytes to verify the file is truly an image.
- Runs fundus validation before returning.
- Image interaction:
- Decodes image bytes using the preprocessing module.
- Saves multipart uploads as timestamped PNG files under `runtime/uploads`.
- Leaves JSON path requests pointing at the existing source file.
- Failure modes:
- Missing image field.
- Invalid filesystem path.
- Empty file.
- Invalid fundus image.

### `predict()`

- Route: `POST /predict`
- Purpose: main inference endpoint used by the GUI.
- Input fields:
- `patient_name`
- `patient_age`
- Uploaded `image` or JSON `image_path`
- Main internal steps:
- Parse patient metadata.
- Read and validate image via `get_request_image()`.
- Preprocess image into tensor and display RGB.
- Run `predict_probabilities()`.
- Compute winning class and confidence.
- Generate CAM overlay.
- Insert SQLite patient record.
- Gather model status.
- Return full JSON response.
- Output fields include:
- `patient_name`
- `patient_age`
- `patient_id`
- `image_path`
- `cam_image_path`
- `prediction`
- `confidence`
- project metrics
- `class_probabilities`
- Image interaction:
- This is the route that owns the full image lifecycle.
- It connects upload decode, preprocessing, prediction, CAM generation, and disk persistence.
- Database interaction:
- Inserts exactly one patient row on each successful prediction.
- Failure modes:
- Returns 400 with `invalid_fundus: true` for rejected non-fundus images.
- Returns generic 400 JSON on other processing failures.

### `patients()`

- Route: `GET /patients`
- Purpose: list all patient records for the History tab.
- Input: none.
- Output:
- `success`
- `count`
- `patients` list
- Image interaction: indirect only, because each patient row contains `image_path`.
- Database interaction: reads all rows ordered by date descending.

### `patient(patient_id)`

- Route: `GET /patients/<int:patient_id>`
- Purpose: fetch one patient row by numeric ID.
- Input: path parameter `patient_id`.
- Output: one patient record or 404 JSON.
- Image interaction: indirect via returned stored `image_path`.
- Database interaction: one-row read query.

### `patient_delete(patient_id)`

- Route: `DELETE /patients/<int:patient_id>`
- Purpose: remove one patient record from the DB.
- Input: path parameter `patient_id`.
- Output: success flag plus deleted count/id.
- Image interaction: does not delete stored image files.
- Database interaction: deletes one row if present.
- Important note:
- This removes metadata, not the corresponding files in `runtime/uploads` or `runtime/cams`.

### `patients_clear()`

- Route: `DELETE /patients`
- Purpose: clear all patient rows.
- Input: none.
- Output: deleted row count.
- Image interaction: does not wipe image files on disk.
- Database interaction: full-table delete for patient history.

## Function Reference: `backend/database.py`

### Module Role

- File type: SQLite persistence layer.
- Core responsibility: create, insert, read, update, and delete patient records.
- Image relationship: stores image paths, not image blobs.

### Global: `DB_PATH`

- Points to `runtime/patient_records.db`.
- Keeps the database beside the runtime image directories.
- Makes the application self-contained on one machine.

### `initialize_database()`

- Purpose: ensure the SQLite file and `patient` table exist.
- Input: none.
- SQL action:
- `CREATE TABLE IF NOT EXISTS patient (...)`
- Output: no structured return, only side effect plus console print.
- Image interaction: none directly.
- Database interaction: schema bootstrap.
- Typical call site: executed at API startup.

### `insert_patient_record(...)`

- Purpose: insert one patient record.
- Required inputs:
- `name`
- `age`
- Optional inputs:
- `image_path`
- `prediction`
- `confidence`
- `date`
- Validation:
- Ensures `name` is a non-empty string.
- Ensures `age` is integer between `0` and `120`.
- Ensures confidence is numeric and within `[0, 1]`.
- Ensures date uses `YYYY-MM-DD`.
- Default behavior:
- If no date is supplied, current date is used.
- SQL action:
- `INSERT INTO patient (...) VALUES (...)`
- Output:
- `success`
- `message`
- `record_id`
- echoed inserted `data`
- Image interaction:
- Stores the absolute or passed image path as text for later GUI reload.

### `get_all_patients()`

- Purpose: return all records for the history table.
- SQL action:
- `SELECT * FROM patient ORDER BY date DESC`
- Output:
- `success`
- `count`
- `patients`
- Image interaction:
- Returns `image_path` strings that the GUI can use to preview old scans.
- Design note:
- Results are converted from SQLite tuples into dictionaries keyed by column names.

### `get_patient_by_id(patient_id)`

- Purpose: fetch one row by primary key.
- SQL action:
- `SELECT * FROM patient WHERE id = ?`
- Output:
- `success`
- `patient` if found
- `message` if missing
- Image interaction:
- Returns the stored image path for one exact historical record.

### `delete_patient_record(patient_id)`

- Purpose: delete one patient row.
- SQL action:
- `DELETE FROM patient WHERE id = ?`
- Output:
- `success`
- `message`
- `deleted`
- Image interaction:
- Leaves files on disk untouched.
- Operational implication:
- History metadata disappears, but uploaded images and CAM images may remain in runtime folders.

### `delete_all_patient_records()`

- Purpose: wipe the patient table.
- SQL action:
- `DELETE FROM patient`
- Output:
- `success`
- `deleted`
- Image interaction:
- Like single delete, this does not remove files from `runtime/uploads` or `runtime/cams`.

### `update_patient_record(patient_id, **kwargs)`

- Purpose: update selected fields of one patient row.
- Allowed fields:
- `name`
- `age`
- `image_path`
- `prediction`
- `confidence`
- `date`
- Query construction:
- Builds the SQL `SET` clause dynamically from provided keyword arguments.
- Output:
- `success`
- `message`
- updated `patient` record
- Image interaction:
- Can rewrite the stored image path if needed.
- Current runtime usage:
- Not a central route in the visible Flask API, but available as a persistence helper.

## Function Reference: `backend/preprocessing.py`

### Module Role

- File type: image validation and tensor preparation layer.
- Core responsibility: accept raw retinal image bytes or BGR arrays and transform them into model-ready tensors plus display-friendly RGB images.

### Global Constants

- `IMAGE_SIZE = (224, 224)`
- `IMAGENET_MEAN`
- `IMAGENET_STD`

These values align preprocessing with standard pretrained vision backbones.

### `is_valid_fundus_image(bgr_img)`

- Purpose: reject non-fundus images before inference.
- Input: OpenCV-style BGR image array.
- Output: boolean.
- Checks performed:
- Minimum spatial resolution.
- Dark corner average.
- Brighter central retinal region.
- Bright pixel coverage in a plausible range.
- Red channel dominance over blue in bright retinal pixels.
- Why it matters:
- Stops random screenshots, documents, or unrelated photos from entering the model path.
- Gives the GUI a specific invalid-fundus error mode.

### `decode_image_bytes(image_bytes)`

- Purpose: convert raw bytes into a BGR image array.
- Primary decoder: OpenCV `cv2.imdecode`.
- Fallback decoder: Pillow with RGB conversion followed by RGB->BGR conversion.
- Output: `numpy.ndarray` image.
- Failure mode:
- Raises `ValueError` if both decode paths fail.
- Image interaction:
- This is the first true image parsing stage of the backend.

### `apply_clahe(bgr_img)`

- Purpose: apply adaptive histogram equalization to emphasize retinal structure.
- Input: BGR uint8 image.
- Steps:
- Convert to grayscale.
- Apply CLAHE with clip limit `2.0` and tile grid size `8x8`.
- Replicate the enhanced grayscale channel into three channels.
- Output: 3-channel uint8 image.
- Image effect:
- Improves vessel and lesion contrast while keeping tensor shape compatible with RGB-style models.

### `center_crop(img, crop_size)`

- Purpose: extract a square crop around the image center.
- Input:
- image array
- crop size integer
- Behavior:
- If the image is too small, reflection padding is added first.
- Then a centered square crop is taken.
- Output: cropped image array.
- Image effect:
- Keeps the retina centered and removes excess background or border regions.

### `preprocess_bgr_image(bgr_img)`

- Purpose: full preprocessing pipeline for an already decoded BGR image.
- Steps:
- CLAHE enhancement
- center crop
- resize to `224 x 224`
- convert BGR to display RGB
- scale to `[0,1]`
- normalize with ImageNet mean/std
- transpose to channel-first format
- add batch dimension
- Output:
- tensor shaped `1 x 3 x 224 x 224`
- display RGB image shaped `224 x 224 x 3`
- Image interaction:
- Returns both the model input and the RGB base image needed for CAM blending.

### `preprocess_for_inference(image_bytes)`

- Purpose: one-call helper for the common runtime path.
- Input: raw image bytes.
- Steps:
- decode bytes
- preprocess BGR image
- Output:
- model tensor
- display RGB image
- Callers:
- `backend/api.py` prediction route.

### `preprocess_image(image_bytes)`

- Purpose: return only the model tensor when the display RGB image is not needed.
- Input: raw bytes.
- Output: tensor only.
- Current usefulness:
- Lightweight helper for code paths that care only about inference input.

## Function Reference: `backend/dl_model.py`

### Module Role

- File type: model discovery, model loading, runtime prediction, backup heuristic inference, and saliency image generation.
- Core responsibility: convert model-ready tensors into probabilities and explanation overlays.
- Image relationship:
- Consumes normalized tensors produced by preprocessing.
- Produces class probabilities.
- Produces explanation images saved to `runtime/cams`.

### Global Capability Flags

#### `_HAS_CV2`

- Indicates whether OpenCV imported successfully.
- Influences whether resize operations can use `cv2.resize` or must fall back to Pillow.

#### `_HAS_TORCH`

- Indicates whether PyTorch imported successfully.
- If false, true neural inference cannot run.

#### `_HAS_TIMM`

- Indicates whether the `timm` model library is available.
- Needed for architecture construction during checkpoint loading.

### Global Constants

#### `CLASS_NAMES`

- Value: `["Normal", "AMD"]`
- Defines class order across inference and training.

#### `IMAGE_SIZE`

- Value: `(224, 224)`
- Matches preprocessing and CAM rendering dimensions.

#### `PACKAGE_DIR`, `PROJECT_ROOT`

- Provide path anchors for model discovery.

#### `DEFAULT_MODEL_PATH`, `DEFAULT_MODEL_PATH_IMPROVED`

- Conventional checkpoint locations.
- Used as preferred candidates during model search.

#### `DEVICE`

- Computed from PyTorch CUDA availability.
- Determines whether tensors and models execute on CPU or GPU.

#### `MAX_MODELS`

- Limits how many checkpoints the runtime ensemble will load.

#### `BACKUP_MODEL_TYPE`

- String constant describing heuristic mode.

#### `DEFAULT_MODEL_NAME`

- Human-readable fallback model name when checkpoint metadata does not provide one.

#### `DEFAULT_METRICS`

- Fixed reported project metrics shown by `/health` and `/predict`.
- Current values:
- Accuracy `0.80`
- Sensitivity `0.79`
- Specificity `0.83`
- Precision `0.95`
- F1 `0.864`

#### `IMAGENET_MEAN_T`, `IMAGENET_STD_T`

- Channel-wise tensors used in backup-mode reverse normalization.
- Shape is prepared for channel-first tensor broadcasting.

### Class: `ViTBinaryClassifier`

- Constructed only when both PyTorch and timm are available.
- Backbone:
- `vit_base_patch16_224`
- `num_classes=0` to expose features
- Head:
- dropout
- linear `768 -> 256`
- GELU
- dropout
- linear `256 -> 1`
- sigmoid
- Output meaning:
- single AMD probability-like scalar for each sample.

### Class: `ViTMultiClassClassifier`

- Constructed only when PyTorch and timm are available.
- Backbone:
- `vit_base_patch16_224`
- `num_classes=2`
- Output meaning:
- two-class logits for `Normal` and `AMD`.

### `_force_backup_mode()`

- Purpose: decide whether the runtime must skip real model loading.
- Returns true if:
- environment variable `FORCE_BACKUP_MODE=1`
- or PyTorch/timm imports are unavailable
- Effect:
- Forces the backend to operate with deterministic heuristic inference rather than checkpoints.

### `candidate_model_paths()`

- Purpose: produce an ordered list of checkpoint candidates.
- Search sources:
- `MODEL_PATH`
- `MODEL_PATH_2`
- `MODEL_PATHS`
- default improved checkpoint
- default standard checkpoint
- recursive search under `backend/models`
- recursive search under alternate likely roots
- Portability feature:
- attempts to remap stale absolute paths from another machine back into the local repo structure.
- Output:
- deduplicated list of `Path` objects sorted to prefer `best` and `improved` checkpoints.

### `_torch_load_checkpoint(path)`

- Purpose: load a checkpoint compatibly across PyTorch versions.
- Special handling:
- tries `weights_only=False` for PyTorch 2.6+ behavior.
- falls back to legacy call if that keyword is unsupported.
- Output:
- raw checkpoint object.

### `_extract_state_dict(checkpoint)`

- Purpose: normalize many checkpoint shapes into a clean state dict.
- Supports:
- direct `nn.Module`
- dictionaries containing `state_dict`
- dictionaries containing `model_state_dict`
- dictionaries containing `model`
- dictionaries containing `net`
- dictionaries containing `weights`
- Extra cleanup:
- strips `module.` prefix from keys for DataParallel checkpoints.
- Output:
- clean parameter dictionary ready for `load_state_dict`.

### `_extract_metrics(checkpoint)`

- Purpose: return the fixed project metrics dictionary.
- Current behavior:
- ignores checkpoint-embedded metrics.
- returns `dict(DEFAULT_METRICS)`.
- Effect:
- Keeps `/health` and `/predict` metric reporting stable and aligned with the project-level values chosen in this codebase.

### `_extract_model_name(model_path, checkpoint)`

- Purpose: derive a human-readable model name.
- Resolution order:
- metadata keys such as `model_name`, `name`, `arch`, `architecture`
- fallback derived from the filename stem
- fallback of `DEFAULT_MODEL_NAME`
- Output:
- string label shown in the GUI and health endpoint.

### `_extract_checkpoint_arch(checkpoint)`

- Purpose: inspect checkpoint metadata for an explicit architecture name.
- Looks for:
- `arch`
- `architecture`
- `model_arch`
- `backbone`
- Output:
- architecture string or `None`.

### `_build_timm_model(arch)`

- Purpose: instantiate a two-class timm model dynamically.
- Output:
- model moved to `DEVICE`
- set to eval mode

### `_build_legacy_model()`

- Purpose: instantiate `ViTBinaryClassifier`.
- Output:
- device-bound eval-mode model.

### `_build_multiclass_model()`

- Purpose: instantiate `ViTMultiClassClassifier`.
- Output:
- device-bound eval-mode model.

### `_looks_like_multiclass(state_dict)`

- Purpose: infer whether a checkpoint uses a two-logit head.
- Method:
- inspects keys ending in `head.weight` or `classifier.weight`
- checks whether the first dimension is `2`
- Output:
- boolean
- Why it matters:
- helps choose the correct runtime class when explicit architecture metadata is absent.

### `_try_load_model(model, state_dict, strict=True)`

- Purpose: perform state-dict load with controlled fallback.
- Behavior:
- first tries strict loading
- if strict loading fails and `strict=True`, retries with `strict=False`
- Output:
- loaded model object
- Why it matters:
- tolerates minor key mismatches while still preferring an exact load.

### `_load_real_model(model_path)`

- Purpose: load one concrete checkpoint into a usable model object.
- Steps:
- load raw checkpoint
- extract clean state dict
- extract display name
- extract metrics
- inspect architecture metadata
- try explicit architecture if metadata says so
- otherwise choose binary vs multiclass builder from head-shape inference
- return successfully loaded model
- Output:
- `(model, model_name, metrics)`
- Failure:
- raises if no compatible architecture can load the state dict.

### `load_models_with_fallback(max_models=MAX_MODELS)`

- Purpose: top-level model discovery and load orchestration.
- Steps:
- if torch/timm missing, declare backup mode immediately
- iterate candidate checkpoint paths
- skip missing files
- deduplicate resolved paths
- attempt load with `_load_real_model`
- accumulate loaded models and metadata
- stop once `max_models` models are ready
- Output:
- loaded model list
- info list
- runtime model type string
- attempted path string
- Important behavior:
- can operate as a small ensemble by loading multiple checkpoints.

### Runtime Globals After Model Load

#### `MODELS`

- Loaded real-model objects, possibly more than one.

#### `ACTIVE_MODEL_INFOS`

- Metadata dictionaries for loaded models.

#### `MODEL_TYPE`

- Either real or backup.

#### `ATTEMPTED_MODEL_PATHS`

- Human-readable record of attempted checkpoint search results.

#### `ACTIVE_MODEL_PATHS`, `ACTIVE_MODEL_PATH`, `ACTIVE_MODEL_NAME`

- Convenience globals for API reporting and GUI display.

### `_ensure_model_ready()`

- Purpose: lazy self-healing check before prediction/status calls.
- Behavior:
- if already in real mode, returns immediately
- otherwise re-runs `load_models_with_fallback`
- updates all active model globals
- Why it matters:
- allows the backend to start before a checkpoint is ready and later recover.

### `is_real_model_loaded()`

- Purpose: boolean helper for downstream logic.
- Output:
- true when the runtime is using actual model checkpoints.

### `is_backup_mode()`

- Purpose: boolean helper for downstream logic.
- Output:
- true when heuristic mode is active.

### `get_model_status()`

- Purpose: return a compact status dictionary for API consumers.
- Output fields:
- `model_type`
- `model_name`
- `backup_active`
- `model_path`
- `model_paths`
- `model_names`
- `models_loaded`
- `attempted_model_paths`
- `metrics`
- Image interaction: none directly.
- API usage:
- returned by `/health`
- also embedded into `/predict`

### `_as_input_tensor(input_tensor, requires_grad=False)`

- Purpose: convert a NumPy tensor into a PyTorch tensor on the correct device.
- Behavior:
- uses `torch.as_tensor`
- casts to float32
- optionally clones/detaches and enables gradients
- Output:
- device-bound `torch.Tensor`
- Why it matters:
- central bridge between NumPy preprocessing output and PyTorch inference/saliency code.

### `_backup_predict_prob_amd(input_tensor)`

- Purpose: provide deterministic heuristic inference when no real model is loaded.
- Inputs:
- normalized tensor expected to be `1 x C x H x W`
- Steps:
- reverse ImageNet normalization to recover approximate RGB `[0,1]`
- isolate the central retinal region
- compute luma statistics
- compute central brightness
- compute overall contrast
- compute red-minus-green dominance
- combine them into a bounded score with tanh-based calibration
- Output:
- scalar AMD probability clipped to `[0.05, 0.95]`
- Image logic:
- brighter healthy-looking centers reduce AMD score
- higher contrast and certain structural patterns increase AMD score
- Important note:
- this is a fallback heuristic, not a medically validated replacement for trained inference.

### `predict_probabilities(input_tensor)`

- Purpose: main runtime prediction entry point.
- Behavior in backup mode:
- compute heuristic AMD probability
- return `[1-AMD, AMD]`
- Behavior in real-model mode:
- convert tensor to PyTorch
- run all loaded models without gradients
- interpret outputs for either:
- two-logit softmax heads
- one-logit sigmoid heads
- median-aggregate AMD probabilities across models
- clamp to `[0,1]`
- return `[Normal, AMD]`
- Output:
- NumPy array of size 2.
- Why median:
- more robust to a single poorly behaving checkpoint than a plain mean in a small ensemble.

### `_pil_resize_rgb(arr, size)`

- Purpose: resize RGB images without OpenCV.
- Method:
- uses Pillow `Image.Resampling.LANCZOS`.
- Output:
- resized RGB NumPy array.

### `_jet_colormap(gray01)`

- Purpose: generate a heatmap without depending on Matplotlib.
- Input:
- normalized grayscale map in `[0,1]`
- Output:
- RGB heatmap approximating classic jet color mapping.
- Role in imaging:
- used for saliency overlays.

### `_resize_rgb(arr, size)`

- Purpose: choose the best available resize backend.
- If current size already matches target:
- returns input unchanged.
- If OpenCV is available:
- uses `cv2.resize`.
- Otherwise:
- uses `_pil_resize_rgb`.

### `generate_explainability_cam(input_tensor, base_rgb, predicted_idx, output_path)`

- Purpose: create and save a visual explanation image.
- Inputs:
- normalized model tensor
- display-friendly RGB retinal base image
- predicted class index
- output filesystem path
- Backup-mode path:
- compute grayscale intensity map from RGB image
- normalize grayscale
- colorize with jet palette
- blend `60%` base image + `40%` heatmap
- save with Pillow
- Real-model path:
- enable gradients on input tensor
- run primary model
- choose target score for predicted class
- backpropagate to input
- average absolute gradients across channels
- normalize gradient map
- colorize with jet palette
- blend with base RGB image
- save with Pillow
- Output:
- string path to the saved CAM image
- Image impact:
- this is the key step that turns the model from a classifier-only system into an interpretable diagnostic aid.

## Function Reference: `backend/server.py`

### Module Role

- File type: backend runtime entrypoint.
- Core responsibility: run the Flask app on port `5000`.

### `main()`

- Purpose: launch the Flask server.
- Behavior:
- reads `FLASK_DEBUG`
- binds `host=0.0.0.0`
- binds `port=5000`
- disables the reloader
- Why `use_reloader=False` matters:
- avoids duplicated process startup when the Qt GUI launches the backend via `QProcess`.

## Function Reference: `backend/__main__.py`

### Module Role

- File type: package execution shim.
- Core responsibility: allow `python -m backend` to work.

### `if __name__ == "__main__": main()`

- Purpose: delegate package execution to `backend.server.main`.
- Practical role:
- this is the command path used by the Qt GUI when it starts the backend automatically.

## Function Reference: `backend/train_and_evaluate.py`

### Module Role

- File type: model training and evaluation pipeline.
- Core responsibility:
- dataset loading
- augmentation
- class balancing
- model creation
- training loop
- validation/test evaluation
- checkpoint writing
- metrics report writing

### Global Training Constants

#### `CLASS_NAMES`, `NUM_CLASSES`

- Define label order for training and evaluation.
- `Normal` maps to `0`.
- `AMD` maps to `1`.

#### `DEFAULT_ARCH`

- Default value: `convnext_tiny.fb_in22k_ft_in1k`
- Rationale in code comments:
- strong retinal performance with practical GPU efficiency.

#### `DEFAULT_IMAGE_SIZE`

- Value: `224`
- Aligns training image size with inference pipeline size.

#### `IMAGENET_MEAN`, `IMAGENET_STD`

- Standard normalization constants used in training transforms.

### Dataclass: `TrainConfig`

- Purpose: bundle all train-time hyperparameters and paths in a typed object.
- Key fields:
- dataset roots
- output path
- architecture
- image size
- batch size
- epoch count
- learning rate
- weight decay
- label smoothing
- warmup epochs
- worker counts
- seed
- early stopping patience
- mixed precision flag
- weighted sampler flag
- gradient clipping
- drop path rate
- notes string

### `set_seed(seed)`

- Purpose: improve reproducibility.
- Seeds:
- Python `random`
- NumPy
- `torch.manual_seed`
- `torch.cuda.manual_seed_all`
- Impact:
- keeps data ordering and initialization behavior more stable across runs.

### `_imagefolder(path, transform)`

- Purpose: safely create one `ImageFolder` dataset if the directory exists and is non-empty.
- Output:
- dataset object or `None`.
- Why it matters:
- makes split discovery resilient when some roots omit a split.

### `_build_split_dataset(dataset_roots, split, transform)`

- Purpose: combine one or more dataset roots for a given split such as `train`, `val`, or `test`.
- Validation:
- requires class folders to be exactly `AMD` and `Normal`.
- forces consistent class ordering across datasets.
- Output:
- one `ImageFolder` or a `ConcatDataset`.
- Failure:
- raises if no valid datasets exist for the requested split.

### `build_transforms(image_size)`

- Purpose: create training and evaluation transform pipelines.
- Training pipeline includes:
- resize
- random resized crop
- horizontal flip
- rare vertical flip
- random rotation
- mild color jitter
- optional Gaussian blur
- tensor conversion
- normalization
- random erasing
- Evaluation pipeline includes:
- deterministic resize
- tensor conversion
- normalization
- Image philosophy:
- augment enough to generalize
- avoid unrealistic color changes for fundus imagery

### `_dataset_targets(dataset)`

- Purpose: extract label targets from either `ImageFolder` or `ConcatDataset`.
- Output:
- flat list of integer class labels.
- Importance:
- central helper for class weights and weighted sampling.

### `class_weights_from_dataset(dataset, device)`

- Purpose: compute inverse-frequency class weights.
- Output:
- tensor of per-class weights on the selected device.
- Use:
- passed into `CrossEntropyLoss`.
- Why it matters:
- helps the model avoid collapsing toward the majority class.

### `make_weighted_sampler(dataset)`

- Purpose: create a `WeightedRandomSampler`.
- Method:
- computes inverse-frequency sample weights from class counts.
- Output:
- sampler that oversamples minority classes in training.

### `create_model(arch, device, drop_path_rate)`

- Purpose: build a pretrained timm classification backbone for training.
- Special behavior:
- adds `drop_path_rate` for compatible architectures like ConvNeXt, Swin, and ViT.
- Output:
- model on the requested device with `num_classes=2`.

### `_metric_block(y_true, y_pred, y_prob_amd)`

- Purpose: compute evaluation metrics after a train/val/test pass.
- Metrics computed:
- accuracy
- precision
- recall
- F1
- ROC AUC when both classes are present
- confusion matrix
- Output:
- dictionary of metrics.
- Importance:
- central reusable evaluator across train/val/test stages.

### `run_epoch(model, loader, criterion, optimizer, device, scaler, grad_clip)`

- Purpose: perform one training epoch.
- Steps:
- set model to train mode
- iterate data loader
- move images and labels to device
- zero optimizer gradients
- forward pass
- compute loss
- backward pass with optional AMP scaler
- optional gradient clipping
- optimizer step
- collect probabilities and predictions
- compute aggregate metrics
- Output:
- metric dictionary including `loss`.

### `evaluate(model, loader, criterion, device, desc="Eval")`

- Purpose: perform validation or testing without gradient computation.
- Steps:
- set eval mode
- disable gradients
- iterate loader
- compute logits and loss
- compute softmax probabilities
- collect predictions
- aggregate metrics with `_metric_block`
- Output:
- metric dictionary including `loss`.

### `_build_scheduler(optimizer, cfg, steps_per_epoch)`

- Purpose: create the training LR schedule.
- Schedule shape:
- linear warmup
- then cosine decay
- Output:
- `LambdaLR` scheduler.
- Why it matters:
- smooth warmup and decay generally stabilize transfer learning on medical image tasks.

### `train_and_evaluate(cfg)`

- Purpose: full training orchestration entry point.
- High-level steps:
- seed everything
- select device
- build transforms
- load train/val/test datasets
- build loaders
- create model
- compute class weights
- create loss, optimizer, scheduler, and scaler
- iterate epochs
- run train epoch
- step scheduler
- run validation
- keep best checkpoint by validation F1
- apply early stopping
- optionally evaluate test split
- write `.metrics.json` report
- Saved checkpoint metadata includes:
- state dict
- model name
- architecture
- image size
- class names
- validation metrics
- ROC AUC
- confusion matrix
- notes
- Why this function is important:
- it is the only place in the repo that turns raw datasets into a new production checkpoint.

### `parse_args()`

- Purpose: parse CLI arguments for training.
- Key CLI knobs:
- dataset roots
- output path
- architecture
- image size
- batch size
- epochs
- learning rate
- weight decay
- label smoothing
- warmup epochs
- seed
- early stopping patience
- worker counts
- AMP toggle
- weighted sampler toggle
- notes

### `main()`

- Purpose: convert CLI args into `TrainConfig` and invoke `train_and_evaluate`.
- Behavior:
- resolves dataset and output paths
- flips no-amp / no-weighted-sampler flags into boolean config values
- hands off to the training pipeline.

## Function Reference: `src/main.cpp`

### File Role

- File type: Qt Widgets desktop frontend.
- Core responsibility:
- build the GUI
- collect user input
- start/check the backend
- upload retinal images
- render original/CAM images
- display prediction and metrics
- manage patient history
- export CSV records
- manage light/dark theme state

### Class: `AMD_GUI`

- Base class: `QMainWindow`
- Main state owned by the class:
- patient form widgets
- image display labels
- metrics labels
- patient history table
- `QNetworkAccessManager`
- `QProcess` for backend startup
- `QSettings` for theme persistence
- cached patient record list for client-side filtering

### Constructor: `AMD_GUI()`

- Purpose: initialize the main window and runtime services.
- Steps:
- set title and size
- create `QSettings`
- load dark mode preference
- create `QNetworkAccessManager`
- create backend `QProcess`
- build UI via `setupUI()`
- apply active theme
- start a repeating health-check timer
- call `refreshBackendStatus()` immediately
- Why it matters:
- this is the point where UI state, theme state, network state, and backend supervision all come together.

### Destructor: `~AMD_GUI()`

- Purpose: stop the backend process if this GUI instance launched it.
- Behavior:
- if the GUI started the backend and it is still running:
- call `terminate()`
- wait briefly
- call `kill()` if it does not exit
- Importance:
- prevents orphan backend processes when the GUI closes.

### `setupUI()`

- Purpose: construct the entire visible application window.
- Major layout zones created:
- left sidebar
- main analysis/history tab area
- status bar
- Sidebar widgets created:
- patient name input
- age spin box
- upload button
- backend status label
- total scan count label
- theme toggle
- Analysis tab widgets created:
- original fundus image group
- CAM image group
- prediction badge
- risk badge
- confidence progress bar
- model/metric labels
- History tab widgets created:
- statistics tiles
- search bar
- refresh button
- export CSV button
- delete selected button
- patient table
- empty state label
- Table behavior configured:
- row selection
- no direct editing
- sorting
- stretch/resize header rules
- alternating row colors
- context menu support
- Signals wired here:
- upload click
- theme toggle
- refresh history
- export CSV
- delete selected
- search filter
- cell click
- double click
- selection change
- custom context menu
- Key takeaway:
- `setupUI()` is both layout code and event wiring code.

### Local Lambda in `setupUI()`: `makeImgGroup`

- Purpose: build one labeled image container.
- Inputs:
- group title
- output `QLabel*&`
- Output:
- configured `QGroupBox` with a centered fixed-size `QLabel`
- Use:
- creates both the fundus image pane and CAM image pane.

### Local Lambda in `setupUI()`: `addMetricPair`

- Purpose: place one metric label/value pair into the results grid.
- Inputs:
- row index
- column index
- metric title
- output label reference
- Use:
- creates `Model`, `Accuracy`, `Sensitivity`, `Specificity`, `Precision`, and `F1 Score` cells.

### Local Lambda in `setupUI()`: `makeStatTile`

- Purpose: create one statistic tile for the History tab.
- Inputs:
- caption text
- output value label reference
- object name for styling
- Output:
- configured `QFrame` tile.

### `uploadImage()`

- Purpose: begin an analysis session from a file chosen by the user.
- Steps:
- open file dialog with image extensions filter
- return immediately if user cancels
- preview selected image in `fundusLabel`
- set CAM label to processing text
- reset prediction/risk/confidence UI
- update status bar
- call `ensureBackendAndPredict(fileName)`
- Image relationship:
- this is the first frontend function that touches a real user-selected fundus image.

### `ensureBackendAndPredict(const QString &fileName)`

- Purpose: check whether the backend is available before trying inference.
- Behavior:
- send `GET /health`
- if success, call `requestPrediction(fileName)`
- if failure, call `startBackendAndWait(...)`
- Why it matters:
- keeps the GUI simple for the user by auto-healing backend availability.

### `startBackendAndWait(const std::function<void()> &onReady)`

- Purpose: launch the backend if it is not already running, then wait for readiness.
- Steps:
- detect project root
- detect Python executable
- set backend command to `-m backend`
- set working directory to repo root
- start `QProcess`
- mark `backendStartedByGui = true`
- update status bar
- call `waitForBackend(...)`
- Platform logic:
- works with Windows `Scripts/python.exe`
- works with Linux `bin/python`
- has `py -3` and `python3` fallbacks

### `waitForBackend(const std::function<void()> &onReady, int attemptsLeft)`

- Purpose: poll `/health` until the backend becomes ready or times out.
- Timeout behavior:
- attempts are decremented on each failure
- polls every `500 ms`
- after exhaustion:
- updates status bar
- sets prediction badge to backend error
- shows a warning dialog
- Why it matters:
- smooths out backend startup delays when importing PyTorch or loading checkpoints.

### `refreshBackendStatus()`

- Purpose: update the sidebar status label periodically.
- Behavior:
- send `GET /health`
- on success:
- parse JSON
- show `Online | <model name>`
- style status green
- on failure:
- show `Offline`
- style status red
- Importance:
- gives the user a passive live heartbeat without manual refresh.

### `requestPrediction(const QString &fileName)`

- Purpose: upload a selected image and patient metadata to the backend.
- Steps:
- open file from disk
- construct multipart request
- attach image file as field `image`
- attach patient name and age
- POST to `/predict`
- on error:
- parse error JSON if possible
- handle invalid-fundus case specially
- show warning dialogs
- on success:
- parse JSON
- reject malformed responses
- call `updateAnalysisResults(obj)`
- call `refreshPatientRecords()`
- Image relationship:
- the frontend-to-backend image handoff happens here.

### `updateAnalysisResults(const QJsonObject &obj)`

- Purpose: render the backend prediction into the analysis UI.
- Updates:
- prediction badge text and style
- risk label text and style
- confidence progress bar
- confidence text
- model name
- accuracy
- sensitivity
- specificity
- precision
- F1 score
- CAM image preview
- status bar completion message
- Logic highlights:
- AMD gets red badge styling
- Normal gets green badge styling
- confidence controls bar color
- CAM image is loaded from `cam_image_path` if the file exists

### `refreshPatientRecords()`

- Purpose: reload full patient history from the backend.
- Behavior:
- send `GET /patients`
- parse JSON array
- cache all patient objects in `allPatients`
- call `populateTable(patients)`
- Importance:
- keeps the History tab synchronized after every successful scan or delete action.

### `filterRecords(const QString &query)`

- Purpose: client-side filter of the cached history list.
- Behavior:
- if query is empty, repopulate table with all cached rows
- otherwise keep only records whose `name` contains the query case-insensitively
- Note:
- current filtering is by patient name, even though the placeholder text suggests broader search.

### `populateTable(const QJsonArray &patients)`

- Purpose: render patient records into the table and related stats.
- Table work:
- disables sorting while rebuilding
- clears previous rows
- inserts one row per patient
- stores patient ID and image path in item data roles
- formats diagnosis cells
- formats confidence cells
- adds an open-image affordance in the final column
- Summary work:
- counts total records
- counts AMD rows
- counts Normal rows
- computes average confidence
- updates sidebar total scans label
- toggles empty-state label visibility
- Importance:
- this is the main UI projection of the database state.

### Local Lambda in `populateTable()`: `centeredItem`

- Purpose: create a table item with centered text.
- Used for IDs, age, diagnosis, confidence, date, and image-open affordance.

### Local Lambda in `populateTable()`: `leftItem`

- Purpose: create a table item with left-aligned text.
- Used primarily for patient names.

### `onRecordSelected(int row, int col)`

- Purpose: react when a user selects or clicks a row in the history table.
- Behavior:
- fetch stored image path from the row's ID item
- if the open-image column was clicked:
- open the full image dialog
- otherwise:
- load the historical fundus image into the main analysis panel
- update status bar
- Image relationship:
- lets the user revisit old scans without re-uploading them.

### `onRecordDoubleClicked(int row, int /*col*/)`

- Purpose: open the full-size image dialog directly on double click.
- Behavior:
- get row image path
- if missing or absent on disk, show status message
- otherwise call `openFullImageDialog`

### `openFullImageDialog(const QString &imgPath, const QString &title)`

- Purpose: show a large scrollable preview of a stored fundus image.
- UI elements created:
- `QDialog`
- `QScrollArea`
- `QLabel` containing scaled `QPixmap`
- close button
- Scaling:
- uses `KeepAspectRatio`
- smooth transformation
- Image relationship:
- this is the most detailed image viewer in the current GUI.

### `showRecordContextMenu(const QPoint &pos)`

- Purpose: provide right-click actions for a selected history row.
- Actions:
- load image into Analysis tab
- open full image
- delete record
- Behavior:
- selects the clicked row
- opens context menu
- dispatches chosen action

### `deleteSelectedRecord()`

- Purpose: delete the currently selected history row through the backend API.
- Steps:
- read selected row ID
- ask the user for confirmation
- send `DELETE /patients/<id>`
- on success:
- show status message
- refresh patient records
- on failure:
- show warning dialog
- Important note:
- this removes the DB row, not necessarily the on-disk image files.

### `exportRecordsCsv()`

- Purpose: save current cached records to a CSV file.
- Behavior:
- refuse export if no records are present
- suggest timestamped filename
- open save-file dialog
- write CSV header
- escape fields containing commas, quotes, or newlines
- write one line per patient record
- update status bar on completion
- Image relationship:
- exports `image_path` strings, not image binary data.

### Local Lambda in `exportRecordsCsv()`: `csvField`

- Purpose: CSV-escape individual string fields.
- Behavior:
- doubles embedded quotes
- wraps value in quotes if needed

### `toggleTheme()`

- Purpose: switch between dark and light UI styles.
- Steps:
- invert `isDarkMode`
- persist new value in `QSettings`
- call `applyDarkMode()` or `applyLightMode()`

### `applyLightMode()`

- Purpose: install the full light-theme stylesheet.
- Affects:
- root window colors
- sidebar
- buttons
- table
- stat tiles
- status bar
- image containers
- prediction/risk badges
- Notes:
- implemented entirely with one long Qt stylesheet string
- no separate `.qss` files are used

### `applyDarkMode()`

- Purpose: install the full dark-theme stylesheet.
- Affects:
- same widget families as light mode, with a black/dark-gray palette
- Highlights:
- white primary button
- dark history table
- color-coded stat values
- styled context menus

### `detectProjectRoot() const`

- Purpose: locate the repo root from several plausible runtime starting points.
- Search order:
- climb up from `applicationDirPath()`
- check current working directory
- check `AMD_PROJECT_SOURCE_DIR`
- fallback to application directory
- Why it matters:
- backend startup depends on finding the `backend` package and the local virtual environment.

### `detectPythonExecutable(const QString &projectRoot, QStringList &arguments) const`

- Purpose: choose the correct Python launcher for the current platform.
- Linux candidates:
- `.venv/bin/python`
- `venv/bin/python`
- `.venv/bin/python3`
- `venv/bin/python3`
- Windows candidates:
- `.venv/Scripts/python.exe`
- `venv/Scripts/python.exe`
- fallback launchers:
- `py -3` on Windows
- `python3` on Linux
- Output:
- executable string
- arguments list set to `-m backend` or Windows fallback equivalent

### `main(int argc, char *argv[])`

- Purpose: process entrypoint for the GUI executable.
- Steps:
- create `QApplication`
- create `AMD_GUI`
- show window
- enter event loop with `app.exec()`

## Inline Helpers and Closures in `src/main.cpp`

This file also contains several helper closures that are not class methods but still shape runtime behavior.

### `makeImgGroup`

- Creates a boxed image panel with centered label and fixed visual size.
- Used for the original fundus image and CAM image.

### `addMetricPair`

- Avoids repetitive label/value pair creation for the metric grid.
- Keeps all metrics consistent in style and layout.

### `makeStatTile`

- Creates repeated statistic summary cards in the History tab.
- Used for total scans, AMD count, Normal count, and average confidence.

### `centeredItem`

- Uniform helper for centered table cells.
- Also stores hidden role metadata such as image paths and numeric sort values.

### `leftItem`

- Simple left-aligned table helper for patient names.

### `csvField`

- Prevents malformed CSV output by quoting fields that need escaping.

### `fmt`

- Used inside `updateAnalysisResults`.
- Converts a JSON metric key into a display string like `80.0%` or `N/A`.

## C++/Qt Image Handling Summary

This section isolates only the image-handling behavior of the desktop GUI.

### Original Image Display Path

- User selects a file in `uploadImage()`.
- `fundusLabel` immediately previews the local file path.
- After backend response, historical reload can replace that preview with a saved absolute `image_path`.

### CAM Image Display Path

- `camsLabel` shows placeholder text while inference is running.
- On successful backend response, `cam_image_path` is read from JSON.
- If the file exists, `QPixmap(camPath)` is loaded and shown.
- If the file is missing, the GUI falls back to a text message.

### History Image Reload Path

- The patient table stores `image_path` in item role data.
- Selecting a row can reload the original image into the Analysis tab.
- Double-clicking or clicking the affordance column opens the full-size preview dialog.

### Full-Image Dialog Path

- `openFullImageDialog()` displays the original fundus image, not the CAM image.
- The image is scaled for dialog display but remains scrollable.

## Cross-File Runtime Trace

This section names the exact function chain for a successful prediction.

### Frontend Chain

- `main()`
- `AMD_GUI()` constructor
- `setupUI()`
- `uploadImage()`
- `ensureBackendAndPredict()`
- `requestPrediction()`
- `updateAnalysisResults()`
- `refreshPatientRecords()`

### Backend Chain

- `backend.server.main()`
- Flask route `predict()`
- `get_request_image()`
- `decode_image_bytes()`
- `is_valid_fundus_image()`
- `preprocess_for_inference()`
- `preprocess_bgr_image()`
- `predict_probabilities()`
- `generate_explainability_cam()`
- `insert_patient_record()`

### Disk Artifacts Created

- one uploaded PNG under `runtime/uploads`
- one CAM PNG under `runtime/cams`
- one patient row in `runtime/patient_records.db`

## Maintenance Notes

### Places Where Metrics Are Controlled

- `backend/api.py`:
- `PROJECT_METRICS`
- `backend/dl_model.py`:
- `DEFAULT_METRICS`
- `src/main.cpp`:
- metric labels and display wiring

If someone changes project metrics later, these are the first places they need to inspect.

### Places Where Image Validation Is Controlled

- `backend/preprocessing.py`
- function: `is_valid_fundus_image`

Threshold changes here directly affect whether uploads are accepted or rejected.

### Places Where Saved Files Are Written

- uploads:
- `backend/api.py` via `UPLOAD_DIR`
- CAM overlays:
- `backend/api.py` and `backend/dl_model.py` via `CAMS_DIR` and `generate_explainability_cam`
- patient DB:
- `backend/database.py` via `DB_PATH`

### Places Where Automatic Backend Startup Is Controlled

- `src/main.cpp`
- `ensureBackendAndPredict()`
- `startBackendAndWait()`
- `waitForBackend()`
- `detectProjectRoot()`
- `detectPythonExecutable()`
