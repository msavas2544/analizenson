@echo off
echo.
echo ========================================
echo   M.SAVAS - Rapor Kutuphaneleri Yukleyici
echo ========================================
echo.

echo Gerekli kutuphaneler kontrol ediliyor...
echo.

echo [1/5] xlsxwriter yukleniyor...
C:\Users\atabe\AppData\Local\Programs\Python\Python311\python.exe -m pip install xlsxwriter --quiet
if %errorlevel% equ 0 (
    echo     ✓ xlsxwriter basariyla yuklendi
) else (
    echo     ✗ xlsxwriter yuklenemedi
)

echo [2/5] python-docx yukleniyor...
C:\Users\atabe\AppData\Local\Programs\Python\Python311\python.exe -m pip install python-docx --quiet
if %errorlevel% equ 0 (
    echo     ✓ python-docx basariyla yuklendi
) else (
    echo     ✗ python-docx yuklenemedi
)

echo [3/5] openpyxl yukleniyor...
C:\Users\atabe\AppData\Local\Programs\Python\Python311\python.exe -m pip install openpyxl --quiet
if %errorlevel% equ 0 (
    echo     ✓ openpyxl basariyla yuklendi
) else (
    echo     ✗ openpyxl yuklenemedi
)

echo [4/5] matplotlib yukleniyor...
C:\Users\atabe\AppData\Local\Programs\Python\Python311\python.exe -m pip install matplotlib --quiet
if %errorlevel% equ 0 (
    echo     ✓ matplotlib basariyla yuklendi
) else (
    echo     ✗ matplotlib yuklenemedi
)

echo [5/5] seaborn yukleniyor...
C:\Users\atabe\AppData\Local\Programs\Python\Python311\python.exe -m pip install seaborn --quiet
if %errorlevel% equ 0 (
    echo     ✓ seaborn basariyla yuklendi
) else (
    echo     ✗ seaborn yuklenemedi
)

echo.
echo ========================================
echo   Kurulum tamamlandi!
echo ========================================
echo.
echo Excel ve Word raporlarini artik olusturabilirsiniz.
echo Uygulama tam ekranda da duzgun gorunecektir.
echo.
pause
