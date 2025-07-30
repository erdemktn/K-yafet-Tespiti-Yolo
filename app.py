from ultralytics import YOLO
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Rengi isme çevirme (basit versiyon)
def bgr_to_color_name(b, g, r):
    if r <= 120 and g <= 120 and b <= 120:
        return "Siyah"
    elif r > 200 and 165 < g <= 245 and 165 <= b <= 245:
        return "Beyaz"
    elif 165 < r < 200 and 145 < g < 170 and 120 < b < 150:
        return "Kahverengi"
    elif 180 < r < 200 and 170 < g < 190 and 145 < b < 170:
        return "Yesil"
    elif 165 < r < 200 and 50 < g < 80 and 60 < b < 90:
        return "Kirmizi"
    elif 95 <= r < 200 and 120 < g < 200 and 120 < b < 225:
        return "Mavi"
    elif 165 <= r < 240 and 160 < g < 220 and 85 < b < 195:
        return "Sari"
    else:
        return f"R:{r} G:{g} B:{b}"  # Debug için sayıları yaz

# Başlık
st.title("🎯 YOLOv8 ile Nesne ve Renk Tespiti")

# Model yükle
model = YOLO("best.pt")

# Görsel yükle
uploaded_file = st.file_uploader("Bir görsel yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Görseli oku
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # BGR yap
    img = cv2.resize(img, (1100, 1000))

    # Model tahmini
    results = model(img)[0]

    # Her nesne için
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        label = model.names[class_id]

        roi = img[y1:y2, x1:x2]
        if roi.size > 0:
            avg_color = np.mean(roi, axis=(0, 1))
            b, g, r = avg_color.astype(int)
            color_name = bgr_to_color_name(b, g, r)
        else:
            color_name = "Bilinmiyor"

        # Kutu ve yazı
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(img, f"{label}-{color_name}", (x1, y1 + 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), 3)

    # Sonucu göster
    result_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(result_rgb, caption="Tahmin Sonucu", use_column_width=True)
