@echo off
echo ========================================
echo M.SAVAS Video Analiz Sistemi Kurulum
echo ========================================
echo.

echo [1/5] Python versiyonu kontrol ediliyor...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo HATA: Python bulunamadi! Lutfen Python 3.8+ yukleyin.
    echo Python indirme: https://python.org/downloads/
    pause
    exit /b 1
)

echo [2/5] Sanal ortam olusturuluyor...
if exist venv (
    echo Sanal ortam zaten mevcut, siliniyor...
    rmdir /s /q venv
)
python -m venv venv
if %errorlevel% neq 0 (
    echo HATA: Sanal ortam olusturulamadi!
    pause
    exit /b 1
)

echo [3/5] Sanal ortam aktif ediliyor...
call venv\Scripts\activate.bat

echo [4/5] Kutuphaneler yukleniyor...
pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo HATA: Kutuphaneler yuklenemedi!
    pause
    exit /b 1
)

echo [5/5] Kurulum tamamlandi!
echo.
echo ========================================
echo KULLANIM:
echo 1. "calistir.bat" dosyasini cift tiklayin
echo 2. Veya terminal'de: python analiz.py
echo ========================================
echo.
pause
