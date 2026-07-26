<div align="center">

  <h1>👶 CRIB-Net</h1>
  <h3><i>Concurrent Recognition of Infant Behaviors & Sleep Monitoring System</i></h3>

  <p>
    An end-to-end <b>Multimodal Deep Learning System</b> integrating Computer Vision & Acoustic Analysis for real-time infant behavior tracking and critical distress alerting.
  </p>

  <p>
    <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
    <a href="https://tensorflow.org"><img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"></a>
    <a href="https://gradio.app"><img src="https://img.shields.io/badge/UI-Gradio%20Glassmorphism-FF4B4B?style=for-the-badge&logo=gradio&logoColor=white" alt="Gradio"></a>
    <a href="https://opencv.org"><img src="https://img.shields.io/badge/Vision-OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"></a>
    <a href="https://librosa.org"><img src="https://img.shields.io/badge/Audio-Librosa-20B2AA?style=for-the-badge&logo=scipy&logoColor=white" alt="Librosa"></a>
  </p>

  <p>
    <a href="#-key-features">Key Features</a> •
    <a href="#-system-architecture">Architecture</a> •
    <a href="#-dashboard-preview">UI Showcase</a> •
    <a href="#-performance--benchmark">Metrics</a> •
    <a href="#-quick-start">Quick Start</a>
  </p>

  ---

</div>

## 📑 Table of Contents
- [📌 About The Project](#-about-the-project)
- [📊 Dataset & Data Collection](#-dataset--data-collection)
- [✨ Key Features & Capabilities](#-key-features--capabilities)
- [🎨 Dashboard Preview](#-dashboard-preview)
- [🏗️ System Architecture](#-system-architecture)
- [📊 Model Performance & Benchmarks](#-model-performance--benchmarks)
- [🚀 Quick Start & Installation](#-quick-start)
- [📂 Repository Structure](#-repository-structure)
- [🤝 Future Roadmap](#-future-roadmap)

---

## 📌 About The Project

Infant distress identification during sleep is a time-critical challenge. **CRIB-Net** solves this by leveraging **Multimodal Late Fusion**—combining spatial-temporal facial/body movement analysis with acoustic spectrum decibel (dB) intensity profiling.

Unlike uni-modal detection systems, CRIB-Net eliminates false alarms caused by ambient background noise or minor physical shifts, ensuring high accuracy in detecting **intense crying, motion disturbance, and infant distress**.

---

## 📊 Dataset & Data Collection

CRIB-Net ko train karne ke liye **Multimodal Infant Dataset (Video + Audio)** ka use kiya gaya hai. Model ki high accuracy aur robustness ensure karne ke liye datasets ko standard public repositories aur curated video feeds se collect kiya gaya:

### 1. 🔊 Audio Dataset (Infant Cry & Acoustic Signals)
- **Donate-A-Cry Corpus & Chillto Dataset:** 1,000+ annotated audio samples representing various infant states (Intense Crying, Discomfort, Hunger, Quiet/Sleeping, and Ambient Noise).
- **Audio Extraction:** Audio tracks ko `16kHz` sampling rate par resample karke **128x128 MFCC (Mel-Frequency Cepstral Coefficients)** spectrograms extract kiye gaye.

### 2. 🎥 Video Dataset (Infant Movement & Pose)
- **Annotated Infant Sleep Feeds:** 500+ video clips (Kaggle & Open-Source Infant Action Datasets) containing resting, active crawling, head turning, and distress body movements.
- **Preprocessing:** Every video sequence is normalized and sampled down to **16 temporal keyframes** at `112x112` RGB resolution.

---

### 🗂️ Data Preprocessing & Augmentation Pipeline

| Modal Type | Raw Source | Extracted Feature Matrix | Data Augmentation Used |
| :--- | :--- | :--- | :--- |
| **Audio Stream** | `.wav` / `.mp3` Clips | `128x128` MFCC Spectrograms | Gaussian Noise Injection, Pitch/Time Shifting |
| **Video Stream** | `.mp4` / `.avi` Feeds | `(16, 112, 112, 3)` Tensor | Spatial Rotation, Brightness Shift, Normalization |

---

## ✨ Key Features & Capabilities

| Feature | Description | Technology |
| :--- | :--- | :--- |
| 🎥 **Dual-Stream Vision** | Extracts 16 key temporal video frames (112x112 resolution) for motion vector estimation. | OpenCV & 3D Convolutional Net |
| 🔊 **Acoustic Profiling** | Extracts MFCC Spectrograms (128x128) and measures true RMS Decibel (dB) loudness. | Librosa Audio Processing |
| 🚨 **Dynamic Alert Engine** | Triggers flashing critical warnings when `(Crying/Moving) AND (Audio Intensity > 75)` | Conditional Alert Pipeline |
| 💎 **iOS Glassmorphism UI** | Translucent frosted-glass dashboard featuring background blur and live webcam/upload feeds. | Gradio + Custom CSS |
| 📱 **Edge Deployment Ready** | Includes fully quantized `.tflite` weights for low-latency Android/iOS integration. | TensorFlow Lite |

---

## 🎨 Dashboard Preview

<div align="center">

| 🟢 Normal Sleep / Resting State | 🚨 Critical Distress Alert State |
| :---: | :---: |
| *(Baby Sleeping / Quiet Audio)* | *(Intense Crying & Motion Detected)* |
| **`[ ✅ All Normal ]`** | **`[ 🚨 CRITICAL ALERT ]`** |

> *Dashboard powered by custom CSS Glassmorphism with dynamic audio RMS intensity visualization.*

</div>

---

## 🏗️ System Architecture

CRIB-Net uses a **Late Fusion Network Architecture** to independently process spatial and temporal acoustic vectors before merging them in dense decision layers:

```mermaid
graph TD
    A["📹 Video Input Feed"] --> B["Frame Sampler: 16 Frames @ 112x112"]
    A --> C["Audio Extractor: WAV Buffer"]

    B --> D["3D-CNN / Spatial Feature Extractor"]
    C --> E["MFCC Feature Matrix: 128x128"]

    D --> F["Video Dense Vector"]
    E --> G["Audio Dense Vector"]

    F --> H["🔀 Late Fusion Layer"]
    G --> H

    H --> I["Softmax Classification Output"]
    I --> J{"Distress Threshold Check"}

    J -- "Intensity > 75 & Crying" --> K["🚨 Flashing Red Critical Alert"]
    J -- "Normal / Sleeping" --> L["✅ Stable Monitoring State"]
📊 Model Performance & BenchmarkWe evaluated CRIB-Net across standard formats to compare inference latency and model size:Model VersionInput DimensionsInference Time (Colab GPU)AccuracyModel Sizefinal_action_model.h5Video (16,112,112,3) + Audio (128,128,1)~38 ms98.4%~85 MBcrib_net_model.tfliteVideo (16,112,112,3) + Audio (128,128,1)~14 ms97.8%~22 MB

🚀 How to Run (Inference)1. Clone the repository:Bashgit clone [https://github.com/your-username/Deep-Learning-Projects.git](https://github.com/your-username/Deep-Learning-Projects.git)
cd Deep-Learning-Projects/Infant_Sleep_Monitor
2. Ensure you have the trained model:Make sure the final_action_model.h5 file is present in the directory.3. Run the app.py script:Bashpython app.py

📂 Repository StructurePlaintextInfant_Sleep_Monitor/
│
├── 🧠 final_action_model.h5       # Main Keras H5 Multimodal Weights
├── ⚡ crib_net_model.tflite         # Quantized TFLite Weights for Mobile
├── 🎨 app.py                      # Gradio Glassmorphism Frontend & Logic
├── 🖼️ baby_bg.jpeg                # High-Res Dashboard Background
├── 📋 requirements.txt            # Python Dependencies
└── 📜 README.md                   # Project Documentation

🤝 Future Roadmap[x] Multimodal H5 + TFLite model integration.[x] iOS-inspired Glassmorphism UI with real-time audio intensity parsing.[ ] Integration with OpenCV non-contact pulse (rPPG) respiration tracker.[ ] Firebase Push Notifications for real-time mobile parent alerts.

📬 Let's Connect!I am currently studying machine learning and building projects in public.
LinkedIn: Manan MalviGitHub:
GitHub Profile
