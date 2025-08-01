@echo off
echo ========================================
echo M.SAVAS Video Analiz Sistemi Baslatiliyor...
echo ========================================
echo.

if not exist venv (
    echo HATA: Sanal ortam bulunamadi!
    echo Lutfen once "kurulum.bat" dosyasini calistirin.
    pause
    exit /b 1
)

echo Sanal ortam aktif ediliyor...
call venv\Scripts\activate.bat

echo Program baslatiliyor...
python analiz.py

echo.
echo Program kapandi.
pause
