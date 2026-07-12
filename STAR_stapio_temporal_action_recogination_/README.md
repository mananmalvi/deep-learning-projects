# STAR_stapio_temporal_action_recogination_

## 🌟 About The Project
**STAR (Spatio-Temporal Action Recognition)** is an advanced Deep Learning-based web application designed to classify human actions from video sequences. Unlike standard image classification models that only look at static pictures, STAR analyzes both **Spatial features** (what is in the frame) and **Temporal features** (how things are moving across multiple frames) to accurately predict the action being performed.

Currently, the model is fine-tuned to recognize specific actions such as:
* 🎯 Archery
* 💄 Applying Lipstick
* 👁️ Applying Eye Makeup

## 🧠 Model Architecture (The AI Engine)
This project utilizes a state-of-the-art **Time-Distributed CNN** architecture:
* **EfficientNetB0 (Base Model):** Used as a feature extractor. It scans each individual frame of the video to understand the spatial features (objects, people, background).
* **TimeDistributed Layer:** This is the heart of the temporal analysis. It allows the model to process a sequence of 10 consecutive frames simultaneously, grasping the motion and time dynamics.
* **GlobalAveragePooling3D & Dense Layers:** Acts as the Artificial Neural Network (ANN) at the end, consolidating the spatio-temporal data to output the final prediction using a softmax activation function.

## 🛠️ Python Libraries Used
The project heavily relies on the following industry-standard libraries:
* `tensorflow` / `keras`: For building, compiling, and training the deep learning model.
* `opencv-python` (`cv2`): For video processing, frame extraction, and image resizing.
* `numpy`: For matrix operations and reshaping video frame arrays.
* `gradio`: For building a seamless, interactive web frontend.
* `scikit-learn`: For splitting the dataset into training and validation sets.

## 🔄 Why We Used a Custom Data Generator?
Working with video datasets (like UCF101) is incredibly memory-intensive. Loading thousands of multi-frame videos directly into memory will instantly crash the system's RAM (especially on platforms like Google Colab). 

To solve this, we built a **Memory-Safe Custom Data Generator**. Instead of loading the entire dataset at once, the generator:
1. Fetches only a small batch of videos at a time.
2. Extracts and preprocesses the frames on the fly.
3. Feeds them to the model and immediately frees up the memory. 
This ensures zero RAM crashes and highly efficient training.

## 💻 Frontend UI (Powered by Gradio)
To make the model accessible and presentable, we developed a dark-themed, premium frontend interface using **Gradio**. 
* **User-Friendly:** * **No Local Setup Needed:** * **Real-time Results:** 

## 🚀 How to Run (Inference)
1. Clone the repository.
2. Ensure you have the trained `final_action_model.h5` file.
3. Run the `app.py` script:
   ```bash
   python app.py
   📬 Let's Connect!
I am currently studying machine learning and building projects in public.

LinkedIn: [https://github.com/]
GitHub: [https://www.linkedin.com/in/manan-malvi-7849b5382/]
