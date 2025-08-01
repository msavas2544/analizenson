# Örnek Kullanım

Bu klasör örnek video dosyaları ve kullanım senaryolarını içerir.

## 📁 Klasör Yapısı

```
examples/
├── README.md              # Bu dosya
├── ornek_kullanim.py      # Basit kullanım örneği
├── toplu_analiz.py        # Toplu analiz örneği
├── rapor_ornegi.py        # Rapor oluşturma örneği
└── videos/                # Örnek video dosyaları (eklenecek)
    ├── kisa_video.mp4
    ├── uzun_video.mp4
    └── test_video.mp4
```

## 🎯 Örnekler

### 1. Basit Kullanım
```python
# ornek_kullanim.py dosyasına bakın
# Tek video analizi örneği
```

### 2. Toplu Analiz
```python
# toplu_analiz.py dosyasına bakın
# Çoklu video analizi örneği
```

### 3. Rapor Oluşturma
```python
# rapor_ornegi.py dosyasına bakın
# Excel, Word, grafik raporu oluşturma
```

## 🔧 Test Senaryoları

### Senaryo 1: Kısa Video (< 30 saniye)
- **Dosya**: `kisa_video.mp4`
- **Beklenen**: Hızlı analiz, az olay
- **Hassasiyet**: YÜKSEK

### Senaryo 2: Uzun Video (> 5 dakika)
- **Dosya**: `uzun_video.mp4`
- **Beklenen**: Uzun analiz, çok olay
- **Hassasiyet**: ORTA

### Senaryo 3: Test Video (çeşitli senaryolar)
- **Dosya**: `test_video.mp4`
- **Beklenen**: Karmaşık tespit durumları
- **Hassasiyet**: ULTRA MAX

## 📊 Performans Testleri

### CPU Testi
```bash
python ornek_kullanim.py --cpu-only
```

### GPU Testi
```bash
python ornek_kullanim.py --gpu
```

### Benchmark
```bash
python benchmark.py
```

## 💡 İpuçları

1. **Küçük videolarla başlayın**: İlk testlerde kısa videolar kullanın
2. **Farklı hassasiyetleri deneyin**: Her video için optimal ayarı bulun
3. **Rapor formatlarını test edin**: Excel, Word, grafik raporlarını deneyin
4. **GPU vs CPU**: Performans farkını gözlemleyin

## 🚀 Gelişmiş Örnekler

### Özelleştirilmiş Analiz
```python
# Sadece belirli bölgeyi analiz etme
# Özel filtreleme kuralları
# Gerçek zamanlı analiz
```

### Entegrasyon Örnekleri
```python
# Web API entegrasyonu
# Veritabanı kayıt
# Email bildirim
```

---

**Not**: Örnek video dosyaları telif hakkı sebebiyle dahil edilmemiştir. Kendi test videolarınızı ekleyebilirsiniz.
