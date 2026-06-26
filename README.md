# 🚀 Face IQ - Real-Time Face Emotion Recognition API

This project features a Convolutional Neural Network (CNN) model trained on the **CK+ dataset** for facial emotion recognition. It bridges the gap between AI development and production by wrapping the trained model in a high-performance **FastAPI** backend and containerizing the entire environment using **Docker** for seamless MLOps deployment.

---

## 🏗️ Project Architecture & Pipeline

The repository represents a complete machine learning lifecycle, split into two main phases:

1. **Research & Development (R&D):** Data preprocessing, CNN architecture design, and model training performed inside Jupyter Notebook.
2. **Production & MLOps:** Model serving via FastAPI and environment isolation using Docker to eliminate the "it works on my machine" problem.



---

## 📂 Project Structure

```text
face_iq/
├── data/                         # CK+ Dataset directory
├── FaceIQ_Training.ipynb         # Model training, analysis, and evaluation
├── best_emotion_model.keras      # Trained CNN model weights
├── detect.py                     # Local/Webcam real-time inference script
├── main.py                       # Production FastAPI backend application
├── Dockerfile                    # Docker build instructions
├── requirements.txt              # Production python dependencies
├── test_api.py                   # Automated integration/smoke test script
└── README.md                     # Project documentation
🛠️ Features
Asynchronous API: Powered by FastAPI for lightning-fast image processing endpoints.

Robust Preprocessing: Automatically handles grayscale conversion, image resizing (48x48), and batch dimension expansion pipeline-safe inside the API.

Dockerized Environment: Isolated TensorFlow and Keras dependencies, ensuring 100% reproducibility across any OS or Cloud provider.

Smoke Testing: Includes a pre-configured test_api.py script to instantly verify container health and prediction accuracy.

🏃‍♂️ How to Run with Docker
To build and run this microservice instantly without messing up your local Python environment, ensure you have Docker installed and run:

Bash
# 1. Build the Docker image
docker build -t face_iq_api .

# 2. Spin up the container (Maps port 8000)
docker run -d -p 8000:8000 --name face_iq_container face_iq_api
Once running, you can access the interactive Swagger API documentation at:
🔗 http://localhost:8000/docs

🧪 Testing the API
You can easily verify the model server's live predictions using the included automated integration script. Just make sure you have a sample.jpg in your directory and run:

Bash
python test_api.py
📊 Sample API Output Response
When successfully pinged, the API returns the predicted dominant emotion along with the confidence score and the full probability breakdown:

JSON
{
  "emotion": "Fear",
  "confidence": 0.6642,
  "all_predictions": {
    "Angry": 0.0583,
    "Disgust": 0.0010,
    "Fear": 0.6642,
    "Happy": 0.0187,
    "Neutral": 0.0370,
    "Sad": 0.1458,
    "Surprise": 0.0746
  }
}
💡 Developed by Furkan Batı as a comprehensive end-to-end MLOps showcase.
