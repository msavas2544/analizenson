# API Dokümantasyonu

## VideoAnalysisWorker Sınıfı

### Kullanım
```python
from analiz import VideoAnalysisWorker

worker = VideoAnalysisWorker(video_path, sensitivity)
worker.start()
```

### Metodlar

#### `_ultra_analyze_frames(cap, model, video_info)`
- **Açıklama**: Video karelerini analiz eder
- **Parametreler**:
  - `cap`: OpenCV VideoCapture objesi
  - `model`: YOLO modeli
  - `video_info`: Video bilgileri
- **Dönüş**: `(detected_objects, detected_frames_list)`

#### `_is_valid_person_detection(detection, frame_width, frame_height)`
- **Açıklama**: Tespit edilen nesnenin geçerli olup olmadığını kontrol eder
- **Parametreler**:
  - `detection`: Tespit objesi
  - `frame_width`: Kare genişliği
  - `frame_height`: Kare yüksekliği
- **Dönüş**: `bool`

## ReportGenerator Sınıfı

### Kullanım
```python
from analiz import ReportGenerator

generator = ReportGenerator(video_path, events, detected_objects, video_info, sensitivity)
generator.generate_excel_report(output_path)
```

### Metodlar

#### `generate_excel_report(output_path)`
- **Açıklama**: Excel raporu oluşturur
- **Parametreler**: `output_path` (string)
- **Dönüş**: `bool`

#### `generate_word_report(output_path)`
- **Açıklama**: Word raporu oluşturur
- **Parametreler**: `output_path` (string)
- **Dönüş**: `bool`

#### `generate_charts(output_dir)`
- **Açıklama**: Grafik raporları oluşturur
- **Parametreler**: `output_dir` (string)
- **Dönüş**: `bool`

## VideoAnalysisApp Sınıfı

### Kullanım
```python
from analiz import VideoAnalysisApp
import sys
from PyQt5.QtWidgets import QApplication

app = QApplication(sys.argv)
window = VideoAnalysisApp()
window.show()
sys.exit(app.exec_())
```

### Önemli Metodlar

#### `load_video()`
- **Açıklama**: Video dosyası yükleme diyalogu açar
- **Dönüş**: `bool`

#### `start_analysis()`
- **Açıklama**: Video analizini başlatır
- **Dönüş**: `None`

#### `on_analysis_complete(detected_objects, events, video_info)`
- **Açıklama**: Analiz tamamlandığında çağrılır
- **Parametreler**:
  - `detected_objects`: Tespit edilen nesneler
  - `events`: Bulunan olaylar
  - `video_info`: Video bilgileri

## Konfigürasyon

### config.ini
```ini
[ANALYSIS]
confidence_threshold = 0.5
iou_threshold = 0.5
max_detections_per_frame = 10

[GPU]
auto_detect = true
fallback_to_cpu = true
```

## Hata Yönetimi

### Yaygın Hatalar
- `ModuleNotFoundError`: Kütüphane eksik
- `FileNotFoundError`: Video dosyası bulunamadı
- `RuntimeError`: GPU hatası

### Hata Yakalama
```python
try:
    worker = VideoAnalysisWorker(video_path, sensitivity)
    worker.start()
except Exception as e:
    print(f"Hata: {e}")
```
