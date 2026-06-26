import requests

# Docker üzerinde çalışan API'mizin adresi
URL = "http://127.0.0.1:8000/predict"

# Test etmek istediğin fotoğrafın bilgisayarındaki tam yolu
# Örnek: "test.jpg" veya klasöründeki gerçek bir resim adı
IMAGE_PATH = "test_resmi.png" 

try:
    # Dosyayı ikili (binary) modda açıyoruz
    with open(IMAGE_PATH, "rb") as f:
        files = {"file": (IMAGE_PATH, f, "image/jpeg")}
        
        print("🚀 Docker üzerindeki API'ye istek gönderiliyor...")
        response = requests.post(URL, files=files)
        
        # Sonucu ekrana yazdırıyoruz
        if response.status_code == 200:
            print("✅ Tahmin Başarılı!")
            print(response.json())
        else:
            print(f"❌ Hata Kodu: {response.status_code}")
            print(response.text)

except FileNotFoundError:
    print(f"❌ Hata: '{IMAGE_PATH}' adında bir dosya bulunamadı. Lütfen klasördeki bir resim adını yazın.")
except Exception as e:
    print(f"💥 Bir hata oluştu: {e}")