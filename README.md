# 🛣️ Smart Infrastructure — Crack Detection System

> **Project Batch 32** · Computer Vision-based structural crack detection for roads and bridges using aerial imagery.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Tests](https://img.shields.io/badge/Tests-pytest-brightgreen)

---

## 📌 Overview

This system automates the detection and severity classification of structural cracks in roads and bridges from aerial images. It uses a classical computer vision pipeline built on **OpenCV**, making it lightweight, interpretable, and deployable without GPU hardware.

**Key outputs for each image:**
- Annotated image with crack contours highlighted by severity color
- Crack area (px²), perimeter length (px), and region count
- Severity classification: `Low` · `Moderate` · `Severe`

---

## 👥 Authors

| Name | Student ID |
|---|---|
| Aditya Francis Masih | 2361002 |
| Amal Tom Ajith | 2361004 |
| Anshika Bansal | 2361005 |
| Aravindhan K | 2361006 |

---

## 🗂️ Project Structure

```
crack-detection/
├── src/
│   ├── crack_detector.py     # Core detection pipeline (CLI + importable class)
│   └── batch_process.py      # Batch-process a folder of images → CSV report
├── tests/
│   └── test_crack_detector.py
├── samples/                  # Place your test images here
├── outputs/                  # Annotated results are saved here
├── docs/
│   └── project_report.pdf
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔬 Detection Pipeline

```
Input Image
    │
    ▼
Grayscale Conversion       ← reduces 3-channel RGB to 1-channel
    │
    ▼
Gaussian Blur              ← suppresses high-frequency noise
    │
    ▼
Adaptive Thresholding      ← handles uneven aerial lighting
    │
    ▼
Canny Edge Detection       ← isolates crack boundary pixels
    │
    ▼
Morphological Dilation     ← closes gaps in fragmented crack lines
    │
    ▼
Contour Detection          ← finds and filters crack regions by area
    │
    ▼
Measurement & Classification
    │
    ▼
Annotated Output Image + Report
```

---

## ⚙️ Severity Classification

| Crack Area (px²) | Severity | Annotation Color |
|---|---|---|
| < 500 | 🟢 Low | Green |
| 500 – 1999 | 🟠 Moderate | Orange |
| ≥ 2000 | 🔴 Severe | Red |

> **Note:** Pixel thresholds are calibrated for images around 1024×768px. For real-world deployments, multiply by your image's ground sampling distance (GSD) to convert px² → cm².

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/crack-detection.git
cd crack-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run on a single image

```bash
python src/crack_detector.py samples/road_crack.jpg --output outputs/result.jpg --show
```

**CLI options:**

| Flag | Default | Description |
|---|---|---|
| `image` | *(required)* | Path to input image |
| `--output / -o` | None | Save annotated image to this path |
| `--show` | False | Display result in a window |
| `--canny-low` | 50 | Canny edge lower threshold |
| `--canny-high` | 150 | Canny edge upper threshold |
| `--min-area` | 50.0 | Minimum contour area to count as a crack |

### 4. Batch process a folder

```bash
python src/batch_process.py \
    --input  samples/ \
    --output outputs/ \
    --report outputs/report.csv
```

This generates an annotated image for every file in `samples/` and writes a `report.csv` with severity, area, length, and region count per image.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Tests cover:
- Preprocessing (grayscale conversion, shape integrity)
- Thresholding (binary output guarantee)
- Edge detection (synthetic crack produces edges)
- Severity classification (boundary conditions)
- Full pipeline (type safety, file-not-found error, blank image baseline)

---

## 🧩 Using as a Python Module

```python
from src.crack_detector import CrackDetector

detector = CrackDetector(
    canny_low=50,
    canny_high=150,
    min_contour_area=50.0,
)

output_image, report = detector.run("path/to/image.jpg")

print(report.severity)       # "Low" | "Moderate" | "Severe"
print(report.total_area)     # float (px²)
print(report.total_length)   # float (px)
print(report.contour_count)  # int
```

---

## 🛠️ Implementation Details

### Grayscale Conversion
Converting BGR to grayscale reduces the 3-channel color image to a single intensity channel, decreasing computational load while preserving the contrast between crack regions (dark) and road surface (light).

```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

### Adaptive Thresholding
Standard binary thresholding fails under non-uniform lighting common in aerial images. Adaptive thresholding computes a local threshold for each pixel neighborhood using a Gaussian-weighted mean.

```python
thresh = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    blockSize=11, C=2
)
```

### Canny Edge Detection
Canny uses a two-stage hysteresis threshold to identify strong and weak edges, retaining only edges that are connected to strong edges.

```python
edges = cv2.Canny(gray, 50, 150)
```

### Morphological Dilation
Dilation expands detected edges to close small gaps, producing more contiguous contours that better represent actual crack geometry.

```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(edges, kernel, iterations=1)
```

### Contour Analysis
Each detected contour is measured for area (`cv2.contourArea`) and perimeter (`cv2.arcLength`). Small contours below `min_contour_area` are discarded as noise.

---

## 📈 Potential Improvements

- [ ] Replace pixel thresholds with physical measurements using GSD metadata from drone telemetry
- [ ] Add crack width estimation via perpendicular cross-section sampling
- [ ] Integrate a lightweight CNN (e.g., MobileNetV2) for improved recall on complex textures
- [ ] Build a Flask/FastAPI web interface for field use
- [ ] Export results as GeoJSON for GIS integration

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [OpenCV Documentation](https://docs.opencv.org/)
- Aerial crack imagery used for testing is sourced from public infrastructure inspection datasets.
