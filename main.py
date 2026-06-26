import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

app = FastAPI(title="Face IQ - Emotion Detection API")

# Duygu etiketleri
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def build_model():
    """Model mimarisini manuel kuruyoruz ki Keras versiyon hatalarından tamamen kurtulalım."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(48, 48, 1)),
        
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    return model

# Modeli oluştur ve ağırlıkları yükle
model = build_model()
try:
    # Model dosyanın adını buraya tam olarak yaz (Örn: best_emotion_model.keras veya .h5 fark etmez)
    model.load_weights("best_emotion_model.keras") 
    print("✅ Model ağırlıkları başarıyla yüklendi!")
except Exception as e:
    print(f"⚠️ Ağırlıklar yüklenirken bir uyarı oluştu, ham load deneniyor: {e}")
    # Eğer mimari uyuşmazsa doğrudan load_model'a geri dönecek bir fallback
    model = tf.keras.models.load_model("best_emotion_model.keras", compile=False)

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    # 1. Resmi oku
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('L') # Gri tonlama (Grayscale)
    
    # 2. Boyutlandır (48x48)
    image = image.resize((48, 48))
    
    # 3. Preprocessing (Normalize etme ve boyutu genişletme)
    image_array = np.array(image, dtype="float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Shape: (1, 48, 48)
    image_array = np.expand_dims(image_array, axis=-1) # Shape: (1, 48, 48, 1)
    
    # 4. Tahmin Et
    predictions = model.predict(image_array)[0]
    max_idx = np.argmax(predictions)
    
    # 5. Tüm olasılıkları hazırla
    all_predictions = {EMOTIONS[i]: float(predictions[i]) for i in range(len(EMOTIONS))}
    
    return {
        "emotion": EMOTIONS[max_idx],
        "confidence": float(predictions[max_idx]),
        "all_predictions": all_predictions
    }