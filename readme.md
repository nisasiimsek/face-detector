# Mini Projem: Yüz Algılama Modeli

Merhaba, ben Nisa Şimşek. Bu mini projede kendi yüz fotoğraflarımı kullanarak YOLOv8 tabanlı bir yüz algılama modeli geliştirdim. Aşağıda projeyi nasıl kurduğum, eğittiğim ve test ettiğim adımlar yer alıyor.

## 1. Veri Toplama ve Etiketleme
- Elimdeki fotoğraflarımı augmentasyon ile çoğalttım.
- Roboflow’da **my_face** etiketiyle yeni bir proje oluşturdum.
- Fotoğrafları yükleyip bounding box ile **my_face** etiketiyle etiketledim.
- Veri setini YOLO formatında indirdim ve `face/` klasörü altına yerleştirdim.

## 2. Ortam Kurulumu
Bu projenin model eğitimini Google Colab üzerinden gerçekleştirdim. Sonrasında eğittiğim modeli indirerek kendi bilgisayarımda çalıştırdım.

## 3. Model Eğitimi
- `data.yaml` içinde:
  ```yaml
  train: train/images
  val: valid/images
  nc: 1
  names: ['my_face']
  ```
- Eğitim komutum:
  ```python
  model = YOLO('yolov8n.pt')
  model.train(
    data='/content/drive/MyDrive/face/Face_detectorns18-2/data.yaml',
    epochs=50, imgsz=640, batch=16, device=0,
    name='face_full_train'
  )
  ```
- Eğitim tamamlandıktan sonra ağırlıklar `face_detector\face\runs\detect\face_full_train\weights\best.pt` içine kaydediliyor.

## Gereksinimler
```
ultralytics
opencv-python
torch
torchvision
```
