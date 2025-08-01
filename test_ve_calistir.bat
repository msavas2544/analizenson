@echo off
echo M.SAVAS Video Analiz Sistemi Test
echo ===================================
cd /d "%~dp0"
echo.
echo Python testi yapiliyor...
python -c "print('Python calisiyor!')"
if errorlevel 1 (
    echo HATA: Python bulunamadi!
    pause
    exit /b 1
)
echo.
echo PyQt5 testi yapiliyor...
python -c "from PyQt5.QtWidgets import QApplication; print('PyQt5 OK')"
if errorlevel 1 (
    echo HATA: PyQt5 bulunamadi!
    echo Lutfen su komutu calistirin: pip install PyQt5
    pause
    exit /b 1
)
echo.
echo OpenCV testi yapiliyor...
python -c "import cv2; print('OpenCV OK -', cv2.__version__)"
if errorlevel 1 (
    echo HATA: OpenCV bulunamadi!
    echo Lutfen su komutu calistirin: pip install opencv-python
    pause
    exit /b 1
)
echo.
echo Tum testler basarili! Uygulama baslatiliyor...
echo.
python analiz.py
pause
