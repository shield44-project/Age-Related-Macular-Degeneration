# Working with DICOM Files using **pydicom**

A **DICOM file** is not just an image—it is a complete medical record:

* 📷 Pixel image data
* 🏷 Metadata tags
* 🏥 Patient & study information
* ⚙ Device / acquisition parameters

Conceptually:

```
DICOM = { header metadata + pixel array }
```

---

## 1. Loading a DICOM File (Basics)

```python
import pydicom

ds = pydicom.dcmread("sample.dcm")
```

### View all metadata

```python
print(ds)
```

### Access specific fields

```python
print(ds.PatientName)
print(ds.Modality)
print(ds.StudyDate)
```

---

## 2. Essential pydicom Operations

### A. Read a file

```python
ds = pydicom.dcmread(path)
```

### B. Get tags safely (avoid crashes)

```python
ds.get("PatientName", "Not Available")
```

### C. Check if a tag exists

```python
if "PatientID" in ds:
    print(ds.PatientID)
```

### D. Access pixel data

```python
img = ds.pixel_array
print(img.shape)
```

---

## 3. Displaying and Normalizing Images

### Display using matplotlib

```python
import matplotlib.pyplot as plt

plt.imshow(ds.pixel_array, cmap='gray')
plt.title("DICOM Image")
plt.axis("off")
plt.show()
```

### Normalize with OpenCV

```python
import cv2

img = ds.pixel_array
img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
```

---

## 4. Modifying DICOM Metadata

### Edit and save

```python
ds.PatientName = "TEST PATIENT"
ds.save_as("modified.dcm")
```

### Delete a tag

```python
del ds.PatientName
```

---

## 5. Important DICOM Concepts

### VR – Value Representation

| VR | Meaning           |
| -- | ----------------- |
| PN | Person Name       |
| DA | Date              |
| UI | Unique Identifier |
| LO | Long String       |
| US | Unsigned Short    |

### Core pydicom Utilities

| Function / Attribute | Purpose                  | Example                          |
| -------------------- | ------------------------ | -------------------------------- |
| dcmread()            | Load DICOM from disk     | ds = pydicom.dcmread('file.dcm') |
| pixel_array          | Convert to NumPy image   | img = ds.pixel_array             |
| save_as()            | Save modified file       | ds.save_as('new.dcm')            |
| dir()                | Search tags              | ds.dir("Patient")                |
| decompress()         | Decode compressed images | ds.decompress()                  |

### Visualization Helpers

| Function     | Purpose              |
| ------------ | -------------------- |
| imshow()     | Show 2D image        |
| colorbar()   | Add intensity legend |
| pcolormesh() | Large heatmaps       |
| matshow()    | Matrix-style display |

---

# 🔥 Additional Useful Functions & Real Examples

## 6. Rescale Using DICOM Windowing

Medical images often require **Window Center / Width**:

```python
import numpy as np

def apply_window(ds):
    img = ds.pixel_array.astype(float)

    center = ds.WindowCenter
    width = ds.WindowWidth

    img = (img - (center - width/2)) / width
    img = np.clip(img, 0, 1)
    return img

windowed = apply_window(ds)
plt.imshow(windowed, cmap='gray')
```

---

## 7. Handling 16-bit → 8-bit Conversion

```python
img = ds.pixel_array
img8 = (img / img.max() * 255).astype('uint8')
```

---

## 8. Reading Multi-Frame DICOM (CT/MRI)

```python
if hasattr(ds, "NumberOfFrames"):
    frames = ds.pixel_array
    print("Frames:", frames.shape)

    plt.imshow(frames[0], cmap='gray')
```

---

## 9. Extract Key Study Info

```python
info = {
    "Patient": ds.get("PatientName"),
    "ID": ds.get("PatientID"),
    "Modality": ds.get("Modality"),
    "Rows": ds.Rows,
    "Columns": ds.Columns
}
print(info)
```

---

## 10. Convert DICOM → PNG

```python
import cv2

img = ds.pixel_array
img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite("output.png", img)
```

---

## 11. Check Compression Type

```python
print(ds.file_meta.TransferSyntaxUID)
```

---

## 12. Anonymize DICOM (Privacy!)

```python
private_tags = [
    "PatientName",
    "PatientID",
    "PatientBirthDate"
]

for tag in private_tags:
    if tag in ds:
        delattr(ds, tag)

ds.save_as("anon.dcm")
```

---

## 13. Create DICOM from NumPy

```python
from pydicom.dataset import Dataset

new = Dataset()
new.Rows, new.Columns = img.shape
new.PixelData = img.tobytes()
new.save_as("created.dcm")
```

---

# Practical Workflow Summary

### Typical Pipeline

1. Load → `dcmread()`
2. Inspect → metadata
3. Extract → `pixel_array`
4. Window / Normalize
5. Process with OpenCV
6. Save result

---

## Key Takeaways

* DICOM = **image + rich medical metadata**
* `pydicom` handles structure; OpenCV handles processing
* Always check:

  * Window settings
  * Bit depth
  * Compression type
* Never modify patient data without anonymization

---

### Minimal Starter Template

```python
import pydicom, cv2, matplotlib.pyplot as plt

ds = pydicom.dcmread("file.dcm")
img = ds.pixel_array

img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

plt.imshow(img, cmap='gray')
plt.show()
```

---

This forms the core foundation for **medical image AI pipelines**, including pediatric eye disease detection, CT analysis, and radiology automation.
