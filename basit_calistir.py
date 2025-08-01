#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M.SAVAŞ Video Analiz Sistemi - Basit Başlatıcı
Bu dosya gerekli kütüphaneleri kontrol eder ve uygulamayı başlatır.
"""

import sys
import os
import subprocess

def check_and_install_requirements():
    """Gerekli kütüphaneleri kontrol eder ve eksikleri yükler"""
    required_packages = [
        'PyQt5',
        'opencv-python', 
        'numpy',
        'matplotlib',
        'openpyxl',
        'python-docx'
    ]
    
    print("🔍 Gerekli kütüphaneler kontrol ediliyor...")
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
                print(f"✅ OpenCV yüklü: {cv2.__version__}")
            elif package == 'PyQt5':
                from PyQt5.QtWidgets import QApplication
                print(f"✅ PyQt5 yüklü")
            elif package == 'numpy':
                import numpy as np
                print(f"✅ NumPy yüklü: {np.__version__}")
            elif package == 'matplotlib':
                import matplotlib
                print(f"✅ Matplotlib yüklü: {matplotlib.__version__}")
            elif package == 'openpyxl':
                import openpyxl
                print(f"✅ OpenPyXL yüklü: {openpyxl.__version__}")
            elif package == 'python-docx':
                import docx
                print(f"✅ Python-docx yüklü")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} eksik")
    
    if missing_packages:
        print(f"\n📦 Eksik kütüphaneler yükleniyor: {', '.join(missing_packages)}")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ {package} başarıyla yüklendi")
            except subprocess.CalledProcessError as e:
                print(f"❌ {package} yüklenemedi: {e}")
                return False
    
    print("✅ Tüm kütüphaneler hazır!")
    return True

def main():
    """Ana başlatıcı fonksiyon"""
    print("🚀 M.SAVAŞ Video Analiz Sistemi Başlatılıyor...")
    print("=" * 50)
    
    # Kütüphaneleri kontrol et
    if not check_and_install_requirements():
        print("❌ Gerekli kütüphaneler yüklenemedi!")
        input("Çıkmak için Enter tuşuna basın...")
        return
    
    print("\n🎯 Uygulama başlatılıyor...")
    
    try:
        # Ana uygulamayı başlat
        from analiz import main as app_main
        app_main()
    except ImportError as e:
        print(f"❌ Uygulama dosyası yüklenemedi: {e}")
        print("💡 analiz.py dosyasının aynı klasörde olduğundan emin olun.")
    except Exception as e:
        print(f"❌ Uygulama başlatılamadı: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nÇıkmak için Enter tuşuna basın...")

if __name__ == "__main__":
    main()
