# AMD Detection System - Qt C++ GUI

## 📌 Overview

This project implements a lightweight Graphical User Interface (GUI) for the detection of **Age-Related Macular Degeneration (AMD)** using retinal fundus images.

The GUI is developed using **Qt Framework in C++** and is designed to interact with a deep learning-based Python backend that performs AMD classification and generates Class Activation Mapping (CAMS) visualizations.

The system allows clinicians or users to upload fundus images captured from fundus cameras and visualize diagnostic outputs in an intuitive interface.

---

## 🎯 Features

- Upload retinal fundus images from local system
- Enter patient name for identification
- Display original fundus image
- Display CAMS heatmap image (affected retinal regions)
- View predicted diagnosis:
  - Treatable AMD
  - Non-Treatable AMD
  - Normal Eye
- Fundus Image History panel for tracking previous scans
- Lightweight and responsive Qt-based GUI

---

## 🧩 System Architecture

```text
Qt C++ GUI
   ↓
Upload Fundus Image
   ↓
Python Backend (DL Model)
   ↓
Diagnosis + CAMS Output
   ↓
Displayed on GUI
```

---

## 🖥️ GUI Components

| Component | Description |
|-----------|-------------|
| Patient Name | Input field for patient details |
| Upload Button | Upload fundus image |
| Fundus Panel | Displays original scan |
| CAMS Panel | Displays heatmap output |
| Diagnosis Label | Displays AMD classification |
| History Panel | Displays previous fundus scans |

---

## ⚙️ Installation (Ubuntu)

Install Qt:

```bash
sudo apt install qtbase5-dev
```

Compile the GUI:

```bash
g++ -fPIC src/main.cpp -o build/amd_gui `pkg-config --cflags --libs Qt5Widgets`
```

Run the application:

```bash
./build/amd_gui
```

---

## 🔗 Backend Integration

The GUI communicates with the Python backend through file-based inter-process communication (IPC):

1. C++ GUI sends fundus image path to Python backend
2. Python performs AMD classification
3. Python generates CAMS output
4. Diagnosis and CAMS image path are returned to GUI
5. Results displayed to the user

---

## 🧠 Technologies Used

- C++
- Qt Framework
- Python (Backend)
- Deep Learning Model
- SQLite Database (Backend)

---

## 📌 Use Case

This system can be used as a clinical decision-support tool for the screening and monitoring of Age-Related Macular Degeneration using retinal fundus imaging.

---

## 🧾 Problem Statement

Age-Related Macular Degeneration is one of the leading causes of vision loss in older adults, and early screening is critical for timely intervention. Manual interpretation of fundus images can be time-consuming and dependent on specialist availability. This project addresses the need for a lightweight diagnostic support interface that connects clinicians to AI-assisted AMD detection and interpretable CAMS visualizations.

---

## 🎯 Objectives

- Build a user-friendly Qt C++ GUI for AMD screening workflows
- Integrate the GUI with a Python deep learning backend
- Provide clear diagnostic output (Treatable AMD / Non-Treatable AMD / Normal Eye)
- Visualize lesion-relevant regions using CAMS heatmaps
- Maintain scan history for quick review and comparison
- Support efficient clinical decision-making with minimal UI complexity

---

## 🧪 Methodology

1. Acquire retinal fundus image from user input
2. Collect patient identifier (name) from GUI
3. Send image path from Qt GUI to backend via file-based IPC
4. Preprocess image and run deep learning inference in Python
5. Generate classification result and CAMS heatmap
6. Return output paths/results to GUI
7. Render original image, CAMS image, diagnosis, and history in the interface

---

## 📜 License

Academic Project for Analysis

