# Project Bone Identify Age

Project Bone Identify Age provides a YOLO-based hand and wrist detection pipeline for pediatric hand/wrist X-ray images.

The goal of the project is to automatically locate the useful anatomical region of the hand and wrist in an X-ray image and crop it for later use in bone age estimation or other medical image processing tasks.

> **Important:** This repository is intended for research and educational purposes only. It is not a certified medical device and must not be used as a standalone clinical decision system.

---

## What this project does

The project uses a trained YOLO model to detect the hand/wrist region in an X-ray image.

The pipeline is:

```text
Input X-ray image
        ↓
YOLO hand detection
        ↓
Bounding box around the hand/wrist
        ↓
Automatic crop
        ↓
Cropped hand image saved to the output folder
```

The YOLO bounding box is used directly as the crop region. A small optional padding can be added to avoid cutting fingertips, the thumb, or the distal wrist.

---

## Repository structure

Expected project structure:

```text
Project-Bone_Identify_Age/
│
├── data/
│   └── yolo_dataset/
│       └── data.yaml
│
├── detection/
│   ├── check_dataset.py
│   ├── create_yolo_dataset.py
│   └── train_yolo11.py
│
├── models/
│   └── yolo_hand/
│       └── best.pt
│
├── src/
│   └── crop_with_yolo.py
│
├── tools/
│   └── auto_label_xray_region.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

### Main files

| File                               | Purpose                                                    |
| ---------------------------------- | ---------------------------------------------------------- |
| `models/yolo_hand/best.pt`         | Trained YOLO model for hand/wrist detection.               |
| `src/crop_with_yolo.py`            | Script used to detect and crop the hand/wrist region.      |
| `tools/auto_label_xray_region.py`  | Utility script used to generate YOLO labels automatically. |
| `detection/train_yolo11.py`        | Training script for the YOLO model.                        |
| `detection/create_yolo_dataset.py` | Creates the YOLO dataset structure.                        |
| `detection/check_dataset.py`       | Checks if images and labels are correctly paired.          |
| `data/yolo_dataset/data.yaml`      | YOLO dataset configuration file.                           |

---

## Requirements

Recommended environment:

```text
Python 3.11
Ultralytics YOLO
OpenCV
NumPy
Pillow
Matplotlib
```

The required Python packages are listed in `requirements.txt`.

Example `requirements.txt`:

```txt
ultralytics
opencv-python
numpy
pillow
matplotlib
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Franciscoafonseca/Project-Bone_Identify_Age.git
cd Project-Bone_Identify_Age
```

### 2. Create a Python environment

Using Conda:

```bash
conda create -n bone_identify_age python=3.11 -y
conda activate bone_identify_age
```

Or using Python virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check if Ultralytics is installed correctly:

```bash
yolo version
```

---

## Model file

The trained YOLO model should be located at:

```text
models/yolo_hand/best.pt
```

If this file is missing, the cropping script will not work.

Expected structure:

```text
models/
└── yolo_hand/
    └── best.pt
```

---

## How to use the model to crop X-ray images

### Option 1: Crop all images inside a folder

Place your X-ray images inside a local folder, for example:

```text
data/raw/
```

Then run:

```bash
python src/crop_with_yolo.py --input data/raw --output data/cropped_yolo --preview data/yolo_crop_preview
```

This will create:

```text
data/cropped_yolo/
```

with the cropped hand/wrist images.

It will also create:

```text
data/yolo_crop_preview/
```

with preview images showing the YOLO detection box.

### Option 2: Crop a single image

```bash
python src/crop_with_yolo.py --input path/to/image.jpeg --output output/cropped --preview output/preview
```

Example:

```bash
python src/crop_with_yolo.py --input data/raw/13.jpeg --output data/cropped_yolo --preview data/yolo_crop_preview
```

---

## Output folders

After running the cropping script, the output folders will contain:

```text
data/cropped_yolo/
```

Cropped images of the detected hand/wrist region.

```text
data/yolo_crop_preview/
```

Original images with the predicted YOLO bounding box drawn on top.

The preview folder is useful for checking whether the model is detecting the hand correctly.

---

## Cropping parameters

The cropping script supports optional parameters.

### Confidence threshold

Default:

```bash
--conf 0.30
```

Example:

```bash
python src/crop_with_yolo.py --input data/raw --output data/cropped_yolo --preview data/yolo_crop_preview --conf 0.30
```

Use a higher value if the model produces weak or incorrect detections:

```bash
--conf 0.45
```

Use a lower value if the model fails to detect some hands:

```bash
--conf 0.25
```

### Horizontal padding

Default:

```bash
--padding-x 0.04
```

Increase this if the crop is cutting the thumb or fingers on the sides:

```bash
--padding-x 0.08
```

Decrease this if the crop is too wide:

```bash
--padding-x 0.02
```

### Vertical padding

Default:

```bash
--padding-y 0.04
```

Increase this if the crop is cutting fingertips or the wrist:

```bash
--padding-y 0.08
```

Decrease this if the crop includes too much background:

```bash
--padding-y 0.02
```

### Disable preview generation

If you only want the cropped images and do not need preview images:

```bash
python src/crop_with_yolo.py --input data/raw --output data/cropped_yolo --no-preview
```

---

## Recommended command

For most cases, start with:

```bash
python src/crop_with_yolo.py --input data/raw --output data/cropped_yolo --preview data/yolo_crop_preview --conf 0.30 --padding-x 0.04 --padding-y 0.04
```

Then inspect the images in:

```text
data/yolo_crop_preview/
```

If the boxes look correct, use the cropped images from:

```text
data/cropped_yolo/
```

---

## Using the YOLO model directly

You can also run YOLO prediction directly from the command line:

```bash
yolo detect predict model=models/yolo_hand/best.pt source=data/raw save=True conf=0.30 iou=0.45 max_det=1 imgsz=640 project=runs/detect name=predict_raw_max1
```

Recommended YOLO parameters:

| Parameter |  Value | Purpose                                    |
| --------- | -----: | ------------------------------------------ |
| `conf`    | `0.30` | Minimum detection confidence.              |
| `iou`     | `0.45` | IoU threshold for non-maximum suppression. |
| `max_det` |    `1` | Keeps only one hand detection per image.   |
| `imgsz`   |  `640` | Inference image size.                      |

Since each X-ray image should contain one hand/wrist region, `max_det=1` is recommended.

---

## Using the model in Python

Example:

```python
from ultralytics import YOLO

model = YOLO("models/yolo_hand/best.pt")

results = model.predict(
    source="data/raw/13.jpeg",
    conf=0.30,
    iou=0.45,
    max_det=1,
    imgsz=640
)

result = results[0]
boxes = result.boxes

if boxes is not None and len(boxes) > 0:
    x1, y1, x2, y2 = boxes.xyxy[0].cpu().numpy()
    print("Detected box:", x1, y1, x2, y2)
else:
    print("No hand detected.")
```

---

## How the crop is generated

The crop is created from the YOLO predicted bounding box.

In simplified form, the logic is:

```python
box = result.boxes.xyxy[0].cpu().numpy()
x1, y1, x2, y2 = box
crop = image[y1:y2, x1:x2]
```

This means that the blue YOLO detection box defines the region that is cropped from the original X-ray image.

---

## Training the YOLO model again

Training is optional. If you only want to use the provided model, you do not need this section.

The expected YOLO dataset structure is:

```text
data/yolo_dataset/
│
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

The `data.yaml` file should contain:

```yaml
path: data/yolo_dataset

train: images/train
val: images/val
test: images/test

names:
  0: hand
```

To check the dataset:

```bash
python detection/check_dataset.py
```

To train the model:

```bash
python detection/train_yolo11.py
```

The trained model will be saved in:

```text
runs/detect/yolo_hand/weights/best.pt
```

After training, copy the best model to:

```text
models/yolo_hand/best.pt
```

Example on Windows:

```bash
copy runs\detect\yolo_hand\weights\best.pt models\yolo_hand\best.pt
```

Example on macOS/Linux:

```bash
cp runs/detect/yolo_hand/weights/best.pt models/yolo_hand/best.pt
```

---

## Important privacy note

Do not upload real medical images to a public GitHub repository unless you have explicit permission and the data is fully anonymized.

Recommended files and folders to keep out of GitHub:

```text
data/raw/
data/cropped/
data/cropped_yolo/
data/auto_preview/
data/auto_debug_masks/
data/yolo_crop_preview/
runs/
```

The trained model can be stored in:

```text
models/yolo_hand/best.pt
```

---

## Troubleshooting

### `yolo` command not found

The Python environment is probably not activated.

Activate the environment first:

```bash
conda activate bone_identify_age
```

Then check:

```bash
yolo version
```

If it still fails, run:

```bash
python -m pip install ultralytics
```

### `No module named ultralytics`

Install the requirements:

```bash
python -m pip install -r requirements.txt
```

### The model file is missing

Check if this file exists:

```text
models/yolo_hand/best.pt
```

If not, copy it from the training output:

```text
runs/detect/yolo_hand/weights/best.pt
```

### The crop cuts the thumb or fingers

Increase horizontal padding:

```bash
--padding-x 0.08
```

Example:

```bash
python src/crop_with_yolo.py --input data/raw --output data/cropped_yolo --preview data/yolo_crop_preview --padding-x 0.08
```

### The crop cuts the fingertips or wrist

Increase vertical padding:

```bash
--padding-y 0.08
```

Example:

```bash
python src/crop_with_yolo.py --input data/raw --output data/cropped_yolo --preview data/yolo_crop_preview --padding-y 0.08
```

### The model detects more than one box

Use `max_det=1`.

In the provided cropping script, this is already configured internally.

---

## Suggested workflow for a new user

Recommended workflow:

```text
1. Clone the repository.
2. Install dependencies.
3. Confirm that models/yolo_hand/best.pt exists.
4. Put X-ray images in data/raw/ or another local folder.
5. Run src/crop_with_yolo.py.
6. Check previews in data/yolo_crop_preview/.
7. Use cropped images from data/cropped_yolo/.
```

Recommended command:

```bash
python src/crop_with_yolo.py --input data/raw --output data/cropped_yolo --preview data/yolo_crop_preview --conf 0.30 --padding-x 0.04 --padding-y 0.04
```

---

## Disclaimer

This project is for research, academic and educational use only.

The model detects and crops the hand/wrist region in X-ray images, but it does not provide a medical diagnosis. Any clinical use requires validation by qualified professionals and compliance with applicable medical regulations.
