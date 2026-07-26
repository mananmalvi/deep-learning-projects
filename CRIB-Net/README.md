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
