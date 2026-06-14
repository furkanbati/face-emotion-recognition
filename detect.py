import cv2
import numpy as np
import tensorflow as tf

# ==========================================
# 1. Load the Trained Model & Setup Categories
# ==========================================
# Load your absolute best model saved by the checkpoint
model = tf.keras.models.load_model('best_emotion_model.keras')

# Define the emotion labels in alphabetical order (matching image_dataset_from_directory)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load OpenCV's built-in face detector (Haar Cascade)
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ==========================================
# 2. Camera Loop and Real-Time Prediction
# ==========================================
# Open the webcam (0 is usually the default built-in camera)
cap = cv2.VideoCapture(0)

print("Press 'q' to exit the camera application.")

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a bounding box around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Crop the face region out of the grayscale frame
        roi_gray = gray_frame[y:y+h, x:x+w]
        
        # Preprocess the cropped face to match the model's expected input
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = np.array(roi_gray, dtype='float32')
        img_pixels = np.expand_dims(img_pixels, axis=0)  # Add batch dimension -> (1, 48, 48)
        img_pixels = np.expand_dims(img_pixels, axis=-1) # Add channel dimension -> (1, 48, 48, 1)
        
        # Note: We don't manually divide by 255 here because our model 
        # has a built-in layers.Rescaling(1./255) layer! Perfect!

        # Make a prediction
        predictions = model.predict(img_pixels, verbose=0)
        max_index = int(np.argmax(predictions))
        predicted_emotion = emotion_labels[max_index]

        # Display the emotion label right above the bounding box
        cv2.putText(frame, predicted_emotion, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the final output frame
    cv2.imshow('Real-Time Emotion Detector', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up and release assets
cap.release()
cv2.destroyAllWindows()