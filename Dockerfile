# 1. Resmi hafif bir Python imajı taban olarak alınıyor
FROM python:3.10-slim

# 2. Konteyner içinde çalışacağımız ana klasörü belirliyoruz
WORKDIR /app

# 3. Sistem bağımlılıklarını güncelliyoruz (Resim işleme kütüphaneleri için gerekebilir)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Kütüphane listemizi konteyner içine kopyalıyoruz
COPY requirements.txt .

# 5. Kütüphaneleri konteyner içinde sıfırdan kuruyoruz
RUN pip install --no-cache-dir -r requirements.txt

# 6. Projedeki tüm kodları, ağırlıkları ve dosyaları konteynere aktarıyoruz
COPY . .

# 7. Dış dünyaya 8000 portunu açıyoruz
EXPOSE 8000

# 8. Konteyner ayağa kalktığında otomatik çalışacak komut
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]