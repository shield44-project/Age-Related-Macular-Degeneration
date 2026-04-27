# AMD Detection System - Qt C++ GUI

## 📌 Overview

This project implements a lightweight Graphical User Interface (GUI) for the detection of **Age-Related Macular Degeneration (AMD)** using retinal fundus images.

The GUI is developed using **Qt Framework in C++** and is designed to interact with a deep learning-based Python backend that performs AMD classification and generates Class Activation Mapping (CAMS) visualizations.

The system allows clinicians or users to upload fundus images captured from fundus cameras and visualize diagnostic outputs in an intuitive interface.

---

## 🎯 Features

- Upload retinal fundus images from the local system
- Enter patient name for identification
- Display original fundus image
- Display CAMS heatmap image (affected retinal regions)
- View predicted diagnosis:
   - AMD
   - Normal
- Fundus Image History panel for tracking previous scans
- Live backend status indicator (Online/Offline)
- Automatic backend startup from GUI when backend is not already running
- **Light/Dark Mode Toggle** with persistent theme preference
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
| Theme Toggle Button | Switch between light and dark mode (🌙/☀️) |
| Fundus Panel | Displays original scan |
| CAMS Panel | Displays heatmap output |
| Diagnosis Label | Displays AMD classification |
| History Panel | Displays previous fundus scans |

---

## ⚙️ Building with CMake (Recommended)

### Prerequisites
- Qt 5.15+ or Qt 6 development libraries
- CMake (version 3.16 or higher)
- C++17/20 compatible compiler
- Python 3.12+ with virtual environment support

### Windows Prerequisites

Install these once:
- Visual Studio 2022 with **Desktop development with C++**
- CMake 3.16 or newer
- Python 3.12 or newer, with **Add python.exe to PATH** enabled
- Qt 5.15+ or Qt 6 from the Qt online installer

During Qt installation, select a desktop kit that matches your compiler, for example:
- `MSVC 2022 64-bit` for Visual Studio builds
- `MinGW 64-bit` only if you plan to use a MinGW compiler

### Windows Build Steps

Open **Developer PowerShell for VS 2022** in the project root.

Create and prepare the Python environment:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\pip install -r requirements.txt
```

Configure CMake. Replace the Qt path with the Qt version installed on your PC:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH="C:\Qt\6.7.3\msvc2022_64"
```

Build the GUI:

```powershell
cmake --build build --config Release
```

Run the GUI:

```powershell
.\build\bin\Release\AMD_GUI.exe
```

If your Qt version is Qt 5, use its MSVC folder instead, for example:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH="C:\Qt\5.15.2\msvc2019_64"
```

The CMake build automatically runs `windeployqt` when it is available, so the required Qt DLLs are copied beside the executable. The GUI also starts the backend automatically by using `.venv\Scripts\python.exe` when the virtual environment exists.

### Linux Prerequisites

Install system dependencies:
```bash
sudo apt update
sudo apt install qtbase5-dev cmake build-essential python3 python3-venv
```

Create and prepare Python environment (project root):
```bash
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
```

### Linux Build Steps

Check `CMakeLists.txt` and verify your CMake version. To check your installed version, run `cmake --version` in the terminal.

Build the GUI:
```bash
cmake -S . -B build
cmake --build build -j
```

Run the GUI:
```bash
./build/bin/AMD_GUI
```

The GUI checks backend health automatically and can start the backend process when needed.

### Packaging (ZIP)

Use the helper script to build, install, and zip the GUI + backend into `dist/`:

Windows example:
```powershell
py -3 scripts/package.py --generator "Visual Studio 17 2022" --arch x64 --qt-prefix "C:\Qt\6.7.3\msvc2022_64"
```

Linux example:
```bash
python3 scripts/package.py
```

Optional flags:
- `--config Release|Debug`
- `--clean` to wipe previous build/install output
- `--zip-name AMD_GUI-Windows.zip` to override the archive name

### Building a Windows installer (.exe) from Ubuntu

The recommended way to produce a Windows `.exe` installer from Ubuntu (or any OS) is the provided **GitHub Actions workflow**. It runs on a real `windows-latest` runner, so no Windows machine or cross-compiler is needed.

**Steps:**

1. Push your branch (or tag it with `v1.0.0`) — the workflow triggers automatically.  
   Or go to **GitHub → Actions → Build Windows Installer (.exe) → Run workflow**.
2. Wait for the job to finish (~10–15 min).
3. Download `AMD_GUI-Windows-Installer` from the **Artifacts** section of the run.

The workflow (``.github/workflows/build-installer.yml``) performs these steps on the Windows runner:

| Step | Tool |
|------|------|
| Build the Qt C++ GUI | CMake + Visual Studio 2022 + Qt 6 |
| Deploy Qt runtime DLLs | `windeployqt` (runs automatically via CMake) |
| Freeze the Flask backend | PyInstaller `--onefile` → `backend_server.exe` |
| Bundle into a Windows installer | Inno Setup 6 (`installer/AMD_GUI.iss`) |

The resulting `AMD_GUI-<version>-Windows-Installer.exe`:
- Installs `AMD_GUI.exe` + Qt DLLs + `backend_server.exe` under `Program Files\AMD Detection System`
- Creates a Start Menu shortcut (optional desktop icon)
- Launches the app on finish
- Includes an uninstaller

**Building the installer locally on a real Windows machine** (optional):

```powershell
# Requires: Visual Studio 2022, Qt 6, Python 3.12+, PyInstaller, Inno Setup 6
py -3 scripts/package.py --installer --generator "Visual Studio 17 2022" --arch x64 --qt-prefix "C:\Qt\6.7.3\msvc2022_64"
```

### CMake Notes

- The project supports both Qt5 and Qt6.
- If CMake cannot find Qt, pass Qt's install folder with `-DCMAKE_PREFIX_PATH=...`.
- On Windows, use a Qt kit that matches your compiler. MSVC Qt builds should be used with Visual Studio; MinGW Qt builds should be used with MinGW.
- The executable is written to `build/bin` on Linux and usually `build/bin/Release` or `build/bin/Debug` on Windows multi-config generators.

### Theme Persistence
The application automatically saves your theme preference using Qt's QSettings. Your chosen mode (light/dark) will be restored when you restart the application.

---

## ⚙️ Alternative: Direct Compilation

If you prefer not to use CMake, you can compile directly:

```bash
g++ -fPIC src/main.cpp -o build/amd_gui `pkg-config --cflags --libs Qt5Widgets Qt5Network` -std=c++17
./build/amd_gui
```

---

## 🔗 Backend Integration

The GUI communicates with the Python backend over HTTP:

1. C++ GUI sends image + patient name to `POST /predict`
2. Python performs AMD classification
3. Python generates CAMS output
4. Diagnosis, confidence, model type, and CAM image path are returned to GUI
5. Results are displayed to the user

### Run Backend Manually (Optional)

```bash
./.venv/bin/python -m backend
```

Alternative script mode:

```bash
./.venv/bin/python backend/server.py
```

Model health check:

```bash
curl -s http://127.0.0.1:5000/health
```

If `model_type` is `real`, the trained model is loaded.

---

## 🧠 Technologies Used

- C++
- Qt Framework
- Python (Backend)
- Deep Learning Model
- Flask REST API

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
- Provide clear diagnostic output (AMD / Normal)
- Visualize lesion-relevant regions using CAMS heatmaps
- Maintain scan history for quick review and comparison
- Support efficient clinical decision-making with minimal UI complexity

---

## 🧪 Methodology

1. Acquire retinal fundus image from user input
2. Collect patient identifier (name) from GUI
3. Send image to backend using HTTP multipart request
4. Preprocess image and run deep learning inference in Python
5. Generate classification result and CAMS heatmap
6. Return output paths/results to GUI
7. Render original image, CAMS image, diagnosis, and history in the interface

---

## 📜 License

Academic Project for Analysis
