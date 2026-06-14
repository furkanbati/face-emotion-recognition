# Real-Time Facial Emotion Recognition (FER) using Custom CNN

A real-time Facial Emotion Recognition system that detects human faces from a live webcam feed and predicts their emotional states instantaneously. The project utilizes a custom Deep Convolutional Neural Network (CNN) trained via TensorFlow/Keras and deployed using OpenCV.

---

## 🚀 Features
* **Real-Time Detection (`detect.py`):** Uses OpenCV's Haar Cascade classifier to isolate faces from a live webcam feed, pre-processes the cropped regions, and overlays emotion predictions dynamically onto the video stream.
* **CPU-Optimized Architecture:** A lightweight yet deep CNN pipeline engineered specifically to achieve high frame rates during live inference on standard local hardware.
* **Smart Training Pipeline:** Implements dynamic learning rate scheduling (`ReduceLROnPlateau`) and automated model saving (`ModelCheckpoint`) to capture peak validation performance.

---

## 📊 Project Summary & Performance Analysis

The core system classifies $48 \times 48$ grayscale facial images into 7 emotional states: **Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral**. 

The model successfully converged at **Epoch 36**, achieving a solid overall test **Accuracy of 59%** across 7,178 evaluation images. For a lightweight architecture trained entirely on a CPU using a highly complex and noisy dataset, this represents a successful and robust baseline.

### 1. Classification Report

```text
--- Classification Report ---
              precision    recall  f1-score   support

       Angry       0.48      0.54      0.51       958
     Disgust       0.48      0.28      0.35       111
        Fear       0.50      0.21      0.30      1024
       Happy       0.79      0.85      0.82      1774
         Sad       0.49      0.65      0.56      1233
    Surprise       0.47      0.44      0.46      1247
     Neutral       0.70      0.72      0.71       831

    accuracy                           0.59      7178
   macro avg       0.56      0.53      0.53      7178
weighted avg       0.59      0.59      0.58      7178


2. Key Insights from Metrics

Live Strengths (Happy & Neutral): The system shows outstanding responsiveness in detect.py when encountering Happy (F1-Score: 0.82) and Neutral (F1-Score: 0.71) expressions. Distinct geometric facial cues like smile lines allowed the network to accurately catch 85% of happy faces.

The Intermediate Classes (Sad, Angry, Surprise): These classes show moderate performance. Sad (Recall: 0.65) is easily caught by the model, but it suffers from lower precision (0.49), implying it frequently misinterprets calm or expressionless faces as sad.

Live Challenges (Fear & Disgust): Fear (F1-Score: 0.30) has a critical bottleneck with only 0.21 recall, frequently triggering brief false positives with Surprise during real-time state changes due to shared traits like widened eyes. Disgust (F1-Score: 0.35) underperformed due to severe class imbalance (only 111 test samples).

🛠️ Installation & Setup
Clone this repository:

Bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
Install the required dependencies:

Bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn seaborn
Ensure your dataset or the trained model is placed in the root directory:

Plaintext
.
├── best_emotion_model.keras
├── detect.py
└── README.md
💻 How to Run (Real-Time Inference)
To deploy the real-time webcam detector using the absolute best weights saved during training, execute the production script:

Bash
python detect.py
Note: Press 'q' on your keyboard while focusing on the camera window to safely close the application.

🔮 Future Improvements
Class Imbalance Mitigation: Apply oversampling or class weighting techniques during compilation to boost the recognition of underrepresented classes like Disgust and Fear.

Architecture Scaling: Port the project to a GPU-accelerated environment to train deeper architectures like ResNet blocks with BatchNormalization.