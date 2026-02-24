# Normalization in Data Processing and Computer Vision

In data processing and computer vision, **Normalization** is the process of scaling numeric data into a specific, standard range—usually **0 to 1** or **-1 to 1**.
Think of it like *leveling the playing field*. If one feature ranges from **0–10** and another from **0–10,000**, a computer may incorrectly treat the larger values as more important. Normalization removes this bias by bringing everything to the same scale.

---

## 1. Why Do We Need Normalization in Computer Vision?

### ✅ Uniformity

Different cameras or medical scanners (e.g., DICOM images) store pixel intensities in different ranges.

* 8-bit images → 0 to 255
* 16-bit medical images → 0 to 65,535

Normalizing them to a common range like **0–1** makes images comparable across devices and datasets.

### ⚡ Faster Learning

Machine Learning models (especially Neural Networks) train faster and more stably when input values are small and similarly scaled.

### 🎨 Better Contrast

Normalization can “stretch” pixel intensities so that dark or low-contrast images become easier to visualize and process.

---

## 2. The Math Behind It

The most common technique is **Min–Max Normalization**.
It converts any value `x` into the range **0 to 1** using:

`x_norm = (x - min(x)) / (max(x) - min(x))`

* If `x` is the minimum → result is **0**
* If `x` is the maximum → result is **1**
* All other values fall between **0 and 1**

---

## 3. Normalization in Code

### A. Using OpenCV (`cv2.normalize`)

```python
import cv2
import numpy as np

img = cv2.imread('low_contrast.jpg', 0)

# Normalize to range 0–255 (standard 8-bit image)
normalized_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

cv2.imshow('Original', img)
cv2.imshow('Normalized', normalized_img)
cv2.waitKey(0)
```

---

### B. Using NumPy (Manual Method)

```python
import numpy as np

data = np.array([100, 200, 300, 400, 500], dtype=float)

data_min = np.min(data)
data_max = np.max(data)

normalized_data = (data - data_min) / (data_max - data_min)

print(normalized_data)
# Output: [0.   0.25  0.5  0.75  1. ]
```

---

## 4. Normalization vs Standardization

These terms are often confused but are different:

* **Normalization**

  * Scales data into a fixed range (e.g., 0–1)
  * Best when data boundaries are known

* **Standardization**

  * Transforms data to mean = 0, standard deviation = 1
  * Used when data follows a Gaussian (bell-curve) distribution

---

## Summary

| Feature   | Raw Data               | Normalized Data                 |
| --------- | ---------------------- | ------------------------------- |
| Range     | Varies (e.g., 0–65535) | Fixed (0–1)                     |
| Data Type | Usually Integer        | Usually Float                   |
| Purpose   | Storage / Capture      | ML / Processing / Visualization |

---

**Key Takeaway:**
Normalization ensures that images and numeric data from different sources become consistent, easier to learn from, and visually clearer—making it a fundamental step in computer vision pipelines.
