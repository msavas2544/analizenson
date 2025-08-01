# M.SAVAŞ Video Analiz Sistemi

**Gelişmiş AI destekli video analiz ve insan tespiti sistemi**

## 🚀 Özellikler

- **Ultra Performans Modu**: GPU destekli hızlı analiz
- **Gelişmiş İnsan Tespiti**: YOLO v8 tabanlı hassas tespit
- **Çoklu Rapor Formatı**: Excel, Word ve grafik raporları
- **Profesyonel Arayüz**: Modern PyQt5 GUI
- **Batch İşlem**: Çoklu video analizi
- **Gerçek Zamanlı İzleme**: Canlı video görüntüleme

## 📋 Sistem Gereksinimleri

- **Python**: 3.8 veya üzeri
- **İşletim Sistemi**: Windows 10/11, macOS, Linux
- **RAM**: Minimum 8GB (16GB önerilir)
- **GPU**: CUDA uyumlu GPU (opsiyonel, performans için)
- **Disk**: 2GB boş alan

## 🛠️ Kurulum

### Windows

1. **Python Kurulumu**
   ```bash
   # Python 3.8+ yükleyin: https://python.org/downloads/
   ```

2. **Proje Klasörünü İndirin**
   ```bash
   git clone <repo-url>
   cd "m.savas analiz"
   ```

3. **Sanal Ortam Oluşturun**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

4. **Bağımlılıkları Yükleyin**
   ```bash
   pip install -r requirements.txt
   ```

### macOS / Linux

1. **Python Kurulumu**
   ```bash
   # macOS için Homebrew ile:
   brew install python3
   
   # Ubuntu/Debian için:
   sudo apt-get install python3 python3-pip
   ```

2. **Proje Klasörünü İndirin**
   ```bash
   git clone <repo-url>
   cd "m.savas analiz"
   ```

3. **Sanal Ortam Oluşturun**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Bağımlılıkları Yükleyin**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Kullanım

### Temel Çalıştırma

```bash
# Windows
python analiz.py

# macOS/Linux
python3 analiz.py
```

### Hızlı Başlangıç

1. **Programı Başlatın**
2. **Video Yükleyin**: "Video Yükle" butonuna tıklayın
3. **Hassasiyet Seçin**: ULTRA MAX önerilir
4. **Analizi Başlatın**: "Analizi Başlat" butonuna tıklayın
5. **Rapor Oluşturun**: Analiz tamamlandıktan sonra rapor butonlarını kullanın

## 📊 Rapor Türleri

### Excel Raporu (.xlsx)
- Detaylı tespit tablosu
- Zaman serisi grafikleri
- İstatistiksel özetler
- Pasta grafikleri

### Word Raporu (.docx)
- Profesyonel rapor formatı
- Görsellerle desteklenen analiz
- Özet ve öneriler

### Grafik Raporları (.png)
- Zaman çizelgesi
- Radar grafikleri
- Isı haritaları
- Dashboard görünümleri

## 🔧 Gelişmiş Ayarlar

### Hassasiyet Seviyeleri

- **DÜŞÜK**: Temel tespit, hızlı işlem
- **ORTA**: Dengeli performans
- **YÜKSEK**: Hassas tespit
- **ULTRA MAX**: En hassas mod (önerilir)

### GPU Desteği

CUDA uyumlu GPU varsa otomatik olarak aktif olur. CPU modu da desteklenir.

## 🐛 Sorun Giderme

### Yaygın Sorunlar

1. **ModuleNotFoundError**
   ```bash
   pip install -r requirements.txt
   ```

2. **CUDA Hatası**
   ```bash
   # CPU moduna geçiş
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

3. **PyQt5 Hatası**
   ```bash
   # Linux için ek paketler
   sudo apt-get install python3-pyqt5
   ```

### Performans İyileştirme

- **GPU Kullanımı**: CUDA yükleyin
- **RAM Optimizasyonu**: Büyük videolar için parçalara bölerek analiz
- **Disk Alanı**: Temp dosyaları temizleyin

## 📝 Lisans

Bu proje MIT lisansı altında dağıtılmaktadır.

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun
3. Commit yapın
4. Push edin
5. Pull request oluşturun

## 📞 Destek

- GitHub Issues: Sorunlar ve öneriler için
- Email: Teknik destek için

## 🏆 Yazarlar

- **M.SAVAŞ**: Ana geliştirici
- **AI Assistant**: Geliştirme desteği

---

**Not**: Bu sistem gelişmiş AI teknolojileri kullanmaktadır. En iyi performans için GPU destekli sistem önerilir.
