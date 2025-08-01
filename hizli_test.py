import sys
import os

# Test kütüphaneleri
def test_libraries():
    print("🔍 Kütüphane testleri başlıyor...")
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV eksik")
        return False
    
    try:
        from PyQt5.QtWidgets import QApplication
        print("✅ PyQt5 mevcut")
    except ImportError:
        print("❌ PyQt5 eksik")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError:
        print("❌ NumPy eksik")
        return False
    
    print("🎉 Temel kütüphaneler hazır!")
    return True

if __name__ == "__main__":
    if test_libraries():
        print("\n🚀 Uygulama başlatılıyor...")
        try:
            # Dosya yolunu kontrol et
            current_dir = os.path.dirname(os.path.abspath(__file__))
            analiz_path = os.path.join(current_dir, "analiz.py")
            
            if not os.path.exists(analiz_path):
                print(f"❌ analiz.py dosyası bulunamadı: {analiz_path}")
                input("Çıkmak için Enter'a basın...")
                sys.exit(1)
            
            # Analiz modülünü import et ve çalıştır
            sys.path.insert(0, current_dir)
            import analiz
            analiz.main()
            
        except Exception as e:
            print(f"❌ Uygulama başlatılamadı: {e}")
            import traceback
            traceback.print_exc()
            input("Çıkmak için Enter'a basın...")
    else:
        print("\n❌ Gerekli kütüphaneler eksik!")
        print("Şu komutları çalıştırın:")
        print("pip install opencv-python PyQt5 numpy")
        input("Çıkmak için Enter'a basın...")
