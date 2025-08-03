# M.SAVAŞ Video Analiz Sistemi

**Gelişmiş AI destekli video analiz ve insan tespiti sistemi**

## 🚀 Özellikler

### 🎯 Temel Özellikler
- **Ultra Performans Modu**: GPU destekli hızlı analiz
- **Gelişmiş İnsan Tespiti**: YOLO v8 tabanlı hassas tespit
- **Çoklu Rapor Formatı**: Excel, Word ve grafik raporları
- **Profesyonel Arayüz**: Modern PyQt5 GUI
- **Batch İşlem**: Çoklu video analizi
- **Gerçek Zamanlı İzleme**: Canlı video görüntüleme

### 🎨 Yeni UI Özellikleri (v2.0)
- **3 Panelli Tasarım**: 
  - Sol Panel: Video kontrolleri ve rapor butonları
  - Orta Panel: Video önizleme (genişleyebilir)
  - Sağ Panel: Ayarlar ve log görüntüleme
- **Geliştirilmiş Butonlar**: Büyük boyutlu (45px) ve hover efektli
- **Singleton Pattern**: Tek uygulama örneği garantisi
- **Zorla Çıkış**: Güvenli uygulama sonlandırma
- **Yeniden Başlatma**: Tek tıkla uygulama yenileme

### 📋 Geliştirilmiş Raporlama
- **Word Raporu**: Resimler 2x2 tablo formatında düzenli görünüm
- **Tablo Tabanlı Layout**: Border'lı ve organize edilmiş görsel sunumu
- **Gelişmiş Formatlar**: Profesyonel rapor şablonları

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

### 🖥️ Arayüz Düzeni (3 Panel)

#### Sol Panel - Video Kontrolleri ve Raporlar
- **Video İşlemleri**: Yükle, başlat, durdur butonları
- **Canlı Kamera**: Kamera başlatma/durdurma
- **Batch İşlem**: Çoklu video analizi
- **Rapor Butonları**: Word, Grafik ve Tüm Raporlar (büyük boyutlu)

#### Orta Panel - Video Önizleme
- **Ana Görüntü**: Video oynatma alanı (genişleyebilir)
- **Tespit Görünümü**: Nesne tespiti overlay'i
- **Tam Ekran**: Video önizleme için optimize edilmiş alan

#### Sağ Panel - Ayarlar ve Loglar
- **Hassasiyet Ayarları**: 4 seviyeli hassasiyet kontrolü
- **Nesne Seçimi**: Tespit edilecek objeler (checkbox'lar)
- **Log Görüntüleme**: Sistem logları ve durum mesajları
- **Sistem Kontrolleri**: Zorla çıkış ve yeniden başlatma

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
- **Yeni Tablo Formatı**: Resimler 2x2 tablo düzeninde
- **Border Destekli**: Organize edilmiş görsel sunumu
- **Profesyonel rapor formatı**: Geliştirilmiş layout
- **Görsellerle desteklenen analiz**: Tablo içinde düzenli resimler
- **Özet ve öneriler**: Kapsamlı analiz sonuçları

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

### 🔒 Sistem Güvenliği

#### Singleton Pattern
- **Tek Örnek**: Aynı anda sadece bir uygulama çalışır
- **Kaynak Koruması**: Çoklu açılım engellenir
- **Güvenli Başlatma**: Otomatik örnek kontrolü

#### Güvenli Çıkış Sistemi
- **Zorla Çıkış**: Acil durum sonlandırması
- **Yeniden Başlatma**: Hızlı uygulama yenileme
- **Kaynak Temizliği**: Güvenli bellek yönetimi

### GPU Desteği

CUDA uyumlu GPU varsa otomatik olarak aktif olur. CPU modu da desteklenir.

## 🆕 Sürüm Güncellemeleri

### v2.0 - UI Yenileme ve Gelişmiş Özellikler

#### 🎨 Arayüz İyileştirmeleri
- **3 Panelli Tasarım**: Sol-Orta-Sağ panel düzeni
- **Responsive Layout**: Orta panel genişleyebilir tasarım
- **Büyük Butonlar**: 45px yükseklik, hover efektli
- **Gelişmiş Renkler**: Modern renk paleti ve gradient'lar

#### 🔧 Teknik İyileştirmeler
- **Singleton Pattern**: Çoklu açılım engelleme
- **Zorla Çıkış**: Güvenli uygulama sonlandırma
- **Kaynak Yönetimi**: Gelişmiş bellek optimizasyonu
- **YOLO Parametreleri**: TARGET_CLASSES ve ACTIVE_CLASSES düzeltmeleri

#### 📄 Rapor Geliştirmeleri
- **Tablo Tabanlı Word**: 2x2 resim düzeni
- **Border Desteği**: XML parse ile border ekleme
- **Organize Layout**: Düzenli görsel sunumu

#### 🚀 Performans
- **Batch Processing**: Gelişmiş çoklu video işleme
- **Hata Düzeltmeleri**: Daha stabil çalışma
- **UI Optimizasyonu**: Daha hızlı arayüz tepkisi

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
