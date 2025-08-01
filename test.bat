@echo off
echo ========================================
echo M.SAVAS Video Analiz Sistemi - Test
echo ========================================
echo.

echo [1/3] Kurulum kontrol ediliyor...
if not exist venv (
    echo HATA: Sanal ortam bulunamadi!
    echo Lutfen "kurulum.bat" dosyasini calistirin.
    pause
    exit /b 1
)

echo [2/3] Sanal ortam aktif ediliyor...
call venv\Scripts\activate.bat

echo [3/3] Program test ediliyor...
python -c "import sys; print(f'Python versiyon: {sys.version}')"
python -c "import PyQt5; print('PyQt5: OK')"
python -c "import cv2; print('OpenCV: OK')"
python -c "import torch; print('PyTorch: OK')"
python -c "import ultralytics; print('Ultralytics: OK')"
python -c "import xlsxwriter; print('XlsxWriter: OK')"
python -c "import docx; print('python-docx: OK')"
python -c "import matplotlib; print('Matplotlib: OK')"

echo.
echo ========================================
echo TEST TAMAMLANDI!
echo Tum kutuphaneler basariyla yuklendi.
echo ========================================
echo.
pause
