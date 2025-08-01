#!/bin/bash

echo "========================================"
echo "M.SAVAS Video Analiz Sistemi Başlatılıyor..."
echo "========================================"
echo

if [ ! -d "venv" ]; then
    echo "HATA: Sanal ortam bulunamadı!"
    echo "Lütfen önce './kurulum.sh' dosyasını çalıştırın."
    exit 1
fi

echo "Sanal ortam aktif ediliyor..."
source venv/bin/activate

echo "Program başlatılıyor..."
python3 analiz.py

echo
echo "Program kapandı."
