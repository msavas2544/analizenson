#!/bin/bash

echo "========================================"
echo "M.SAVAS Video Analiz Sistemi Kurulum"
echo "========================================"
echo

echo "[1/5] Python versiyonu kontrol ediliyor..."
if ! command -v python3 &> /dev/null; then
    echo "HATA: Python3 bulunamadi!"
    echo "macOS için: brew install python3"
    echo "Ubuntu için: sudo apt-get install python3 python3-pip"
    exit 1
fi

echo "[2/5] Sanal ortam oluşturuluyor..."
if [ -d "venv" ]; then
    echo "Sanal ortam zaten mevcut, siliniyor..."
    rm -rf venv
fi
python3 -m venv venv

echo "[3/5] Sanal ortam aktif ediliyor..."
source venv/bin/activate

echo "[4/5] Kütüphaneler yükleniyor..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[5/5] Kurulum tamamlandı!"
echo
echo "========================================"
echo "KULLANIM:"
echo "1. Terminal'de: ./calistir.sh"
echo "2. Veya: source venv/bin/activate && python3 analiz.py"
echo "========================================"
echo
