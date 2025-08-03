from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import gradio as gr

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
    elif 170 <= r < 210 and 90 < g < 175 and 85 < b < 195:
        return "Pembe"
    else:
        return f"R:{r} G:{g} B:{b}"  # Debug için sayıları yaz

# Modeli yükle (bir kez çalışacak)
model = YOLO("best.pt")

def detect_and_color(image: Image.Image):
    # PIL Image -> numpy BGR formatı
    img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (1100, 1000))

    # Model tahmini
    results = model(img)[0]

    # Her kutu için renk ve label ekle
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

        # Kutu ve yazı çizimi
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
        # Etiket metni
        text = f"{label}-{color_name}"
        
        # Yazının boyutlarını al (genişlik, yükseklik)
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, 2)
        
        # Arka plan kutusunu çiz (dolu dikdörtgen)
        cv2.rectangle(img,
                      (x1, y1),  # Sol üst köşe
                      (x1 + text_width, y1 + text_height + baseline),  # Sağ alt köşe
                      (0, 0, 0),  # Arka plan rengi (siyah)
                      -1)  # -1 => dolu dikdörtgen
        
        # Yazıyı kutunun üstüne çiz
        cv2.putText(img, text,
                    (x1, y1 + text_height),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    2,  # font boyutu
                    (255, 255, 255), 2)  # Yazı rengi: beyaz

    # Sonucu RGB olarak döndür
    result_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)

# Gradio arayüzü
iface = gr.Interface(
    fn=detect_and_color,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="🎯Kiyafet ve Renk Tespiti",
    description="""Ürünler: Pantolon, Gömlek, Tshirt, Kazak, Etek.
\nRenkler: Siyah, Beyaz, Kahverengi, Kırmızı, Yeşil, Sarı, Mavi, Pembe
\nVeri Sayısı: 1200
\nModel: YOLOv8 modeli ile resimler etiketlenerek eğitilmiştir.
\nKullanılan Teknolojiler: Python, YOLO, Gradio, OpenCV, Numpy, Pillow
\n⚠️ Yapmış olduğunuz denemelerde ışık ve arka plan açılarına göre tahminler değişiklik gösterebilir."""

)

if __name__ == "__main__":
    iface.launch()
