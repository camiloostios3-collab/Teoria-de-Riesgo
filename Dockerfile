# ─── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11.9-slim-bookworm AS builder

WORKDIR /app

# Instalar dependencias de sistema necesarias para compilar paquetes C
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev && \
    rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ─── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11.9-slim-bookworm AS runtime

WORKDIR /app

# Copiar paquetes instalados desde el builder
COPY --from=builder /install /usr/local

# Copiar código fuente del backend
COPY backend/ ./

# Crear directorio de datos persistentes
RUN mkdir -p /app/data

# Variables de entorno por defecto (sobreescribir en docker-compose o en runtime)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
