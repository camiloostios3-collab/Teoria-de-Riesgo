@echo off
chcp 65001 >nul
title RiskLab · USTA

echo.
echo  ╔══════════════════════════════════════════╗
echo  ║       RiskLab · USTA  —  Iniciando       ║
echo  ╚══════════════════════════════════════════╝
echo.

REM ── Verificar que existe el entorno virtual ────────────────────────────────
if not exist ".venv\Scripts\activate.bat" (
    echo  [ERROR] No se encontro el entorno virtual .venv
    echo  Ejecuta primero:  python -m venv .venv
    echo                    .venv\Scripts\activate
    echo                    pip install -r backend\requirements.txt
    pause
    exit /b 1
)

REM ── Activar entorno virtual ────────────────────────────────────────────────
call .venv\Scripts\activate.bat

REM ── Verificar que uvicorn esta instalado ──────────────────────────────────
where uvicorn >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] uvicorn no esta instalado en el entorno virtual.
    echo  Ejecuta:  pip install -r backend\requirements.txt
    pause
    exit /b 1
)

REM ── Levantar el backend FastAPI en una ventana nueva ──────────────────────
echo  [1/2] Iniciando backend FastAPI en http://localhost:8000 ...
start "RiskLab - Backend API" cmd /k "title RiskLab-Backend ^& cd /d "%~dp0backend" ^& uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

REM ── Esperar 3 segundos para que el backend arranque ───────────────────────
echo  Esperando que el backend arranque...
timeout /t 3 /nobreak >nul

REM ── Levantar el frontend Streamlit en una ventana nueva ───────────────────
echo  [2/2] Iniciando dashboard Streamlit en http://localhost:8501 ...
start "RiskLab - Dashboard" cmd /k "title RiskLab-Dashboard ^& cd /d "%~dp0" ^& streamlit run app.py --server.port 8501"

REM ── Abrir el navegador ────────────────────────────────────────────────────
echo.
echo  Abriendo el dashboard en el navegador...
timeout /t 4 /nobreak >nul
start "" http://localhost:8501

echo.
echo  ╔══════════════════════════════════════════════════════════════╗
echo  ║  Backend:   http://localhost:8000  (docs en /docs)          ║
echo  ║  Dashboard: http://localhost:8501                           ║
echo  ║                                                              ║
echo  ║  Para detener: cierra las dos ventanas negras               ║
echo  ╚══════════════════════════════════════════════════════════════╝
echo.
pause
