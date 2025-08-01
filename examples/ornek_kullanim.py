#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.SAVAŞ Video Analiz Sistemi - Örnek Kullanım

Bu script temel video analizi işlemlerini gösterir.
"""

import sys
import os
import argparse
from pathlib import Path

# Ana dizini path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from analiz import VideoAnalysisWorker, ReportGenerator
    from PyQt5.QtCore import QObject, pyqtSignal
    from PyQt5.QtWidgets import QApplication
except ImportError as e:
    print(f"Gerekli kütüphaneler yüklenemiyor: {e}")
    print("Lütfen 'pip install -r requirements.txt' komutunu çalıştırın")
    sys.exit(1)

class OrnekAnaliz(QObject):
    """Örnek analiz sınıfı"""
    
    def __init__(self):
        super().__init__()
        self.app = QApplication(sys.argv)
    
    def tek_video_analiz(self, video_path, sensitivity="ULTRA MAX"):
        """Tek video analizi örneği"""
        
        print(f"🎥 Video analizi başlatılıyor: {video_path}")
        print(f"📊 Hassasiyet: {sensitivity}")
        
        # Video dosyasının varlığını kontrol et
        if not os.path.exists(video_path):
            print(f"❌ Video dosyası bulunamadı: {video_path}")
            return False
        
        # Analiz worker'ı oluştur
        worker = VideoAnalysisWorker(video_path, sensitivity)
        
        # Sonuçları saklamak için
        self.analiz_sonucu = None
        self.olaylar = None
        self.video_bilgisi = None
        
        # Signal bağlantısı
        worker.analysis_complete.connect(self.analiz_tamamlandi)
        
        # Analizi başlat
        worker.start()
        
        # Analiz tamamlanana kadar bekle
        while worker.isRunning():
            self.app.processEvents()
        
        return True
    
    def analiz_tamamlandi(self, detected_objects, events, video_info):
        """Analiz tamamlandığında çağrılır"""
        
        self.analiz_sonucu = detected_objects
        self.olaylar = events
        self.video_bilgisi = video_info
        
        print(f"✅ Analiz tamamlandı!")
        print(f"📈 Tespit edilen olay sayısı: {len(events)}")
        print(f"👥 Tespit edilen nesne sayısı: {len(detected_objects)}")
        
        # Olayları listele
        if events:
            print("\n📋 Bulunan olaylar:")
            for i, (start, end) in enumerate(events, 1):
                print(f"  {i}. Olay: {start:.2f}s - {end:.2f}s ({end-start:.2f}s)")
        
        # Rapor oluştur
        self.rapor_olustur()
    
    def rapor_olustur(self):
        """Analiz sonuçlarından rapor oluştur"""
        
        if not self.analiz_sonucu:
            print("❌ Analiz sonucu bulunamadı")
            return
        
        # Rapor dizinini oluştur
        rapor_dir = Path("reports/ornek_analiz")
        rapor_dir.mkdir(parents=True, exist_ok=True)
        
        # ReportGenerator oluştur
        generator = ReportGenerator(
            video_path="ornek_video.mp4",
            events=self.olaylar,
            detected_objects=self.analiz_sonucu,
            video_info=self.video_bilgisi,
            sensitivity="ULTRA MAX"
        )
        
        print("\n📄 Raporlar oluşturuluyor...")
        
        # Excel raporu
        excel_path = rapor_dir / "analiz_raporu.xlsx"
        if generator.generate_excel_report(str(excel_path)):
            print(f"✅ Excel raporu: {excel_path}")
        else:
            print("❌ Excel raporu oluşturulamadı")
        
        # Word raporu
        word_path = rapor_dir / "analiz_raporu.docx"
        if generator.generate_word_report(str(word_path)):
            print(f"✅ Word raporu: {word_path}")
        else:
            print("❌ Word raporu oluşturulamadı")
        
        # Grafik raporları
        if generator.generate_charts(str(rapor_dir)):
            print(f"✅ Grafik raporları: {rapor_dir}")
        else:
            print("❌ Grafik raporları oluşturulamadı")
        
        print(f"\n🎉 Tüm raporlar hazır: {rapor_dir}")

def main():
    """Ana fonksiyon"""
    
    parser = argparse.ArgumentParser(description='M.SAVAŞ Video Analiz Sistemi - Örnek Kullanım')
    parser.add_argument('video', help='Analiz edilecek video dosyası')
    parser.add_argument('--sensitivity', default='ULTRA MAX', 
                       choices=['DÜŞÜK', 'ORTA', 'YÜKSEK', 'ULTRA MAX'],
                       help='Analiz hassasiyeti')
    parser.add_argument('--cpu-only', action='store_true', 
                       help='Sadece CPU kullan (GPU\'yu devre dışı bırak)')
    
    args = parser.parse_args()
    
    # CPU modu ayarla
    if args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("🔧 CPU modu etkinleştirildi")
    
    # Analiz nesnesini oluştur
    analiz = OrnekAnaliz()
    
    # Tek video analizi yap
    if analiz.tek_video_analiz(args.video, args.sensitivity):
        print("🎯 Analiz başarıyla tamamlandı")
    else:
        print("❌ Analiz başarısız")
        sys.exit(1)

if __name__ == "__main__":
    main()
