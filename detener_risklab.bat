@echo off
chcp 65001 >nul
title RiskLab · Detener servicios

echo.
echo  Deteniendo RiskLab...
echo.

REM Matar el proceso uvicorn (backend)
taskkill /f /im uvicorn.exe >nul 2>&1
echo  [OK] Backend FastAPI detenido.

REM Matar el proceso streamlit (frontend)
taskkill /f /fi "WINDOWTITLE eq RiskLab-Dashboard*" >nul 2>&1
for /f "tokens=2" %%i in ('tasklist /fi "IMAGENAME eq streamlit.exe" /fo table /nh 2^>nul') do (
    taskkill /f /pid %%i >nul 2>&1
)
echo  [OK] Dashboard Streamlit detenido.

echo.
echo  Todos los servicios detenidos.
pause
