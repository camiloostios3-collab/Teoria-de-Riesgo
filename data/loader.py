"""
data/loader.py
Descarga datos de mercado desde yfinance con caché Streamlit.
Robusto ante distintos formatos de columnas que devuelve yfinance según versión.
"""

import time
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

TICKERS   = ["AAPL", "JPM", "XOM", "JNJ", "AMZN"]
BENCHMARK = "^GSPC"
SECTOR_MAP = {
    "AAPL": "Tecnología",
    "JPM" : "Financiero",
    "XOM" : "Energía",
    "JNJ" : "Salud",
    "AMZN": "Consumo discrecional",
}
TICKER_COLORS = {
    "AAPL": "#A89060",
    "JPM" : "#3D8B6E",
    "XOM" : "#3A6B8A",
    "JNJ" : "#5A4E7A",
    "AMZN": "#8B4A4A",
}


def _extract_close(raw: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    Extrae SOLO las columnas de precio de cierre del DataFrame de yfinance.

    yfinance puede devolver distintos formatos según la versión y si se pide
    uno o varios tickers:
      - MultiIndex (field, ticker): raw["Close"] → DataFrame con tickers como cols
      - MultiIndex (ticker, field): raw.xs("Close", level=1, axis=1)
      - Columnas planas = tickers (multi_level_index=False funcionó bien)
      - Columnas planas = campos OHLCV (un solo ticker, multi_level_index=False)
    """
    if raw.empty:
        return pd.DataFrame()

    # ── Caso 1: MultiIndex ──────────────────────────────────────────────────
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0).unique().tolist()
        level1 = raw.columns.get_level_values(1).unique().tolist()

        if "Close" in level0:
            # Formato (field, ticker) — el más común
            return raw["Close"]
        elif "Close" in level1:
            # Formato (ticker, field) — algunas versiones de yfinance
            return raw.xs("Close", axis=1, level=1)
        else:
            logger.warning("MultiIndex sin 'Close'. Nivel 0: %s", level0[:5])
            return pd.DataFrame()

    # ── Caso 2: columnas planas ─────────────────────────────────────────────
    cols = set(raw.columns.tolist())

    # ¿Las columnas son tickers directamente? (multi_level_index=False correcto)
    ticker_cols = [t for t in tickers if t in cols]
    if ticker_cols:
        return raw[ticker_cols]

    # ¿Las columnas son campos OHLCV? (single ticker o versión antigua)
    ohlcv_fields = {"Open", "High", "Low", "Close", "Volume",
                    "Adj Close", "Dividends", "Stock Splits"}
    if "Close" in cols and len(cols.intersection(ohlcv_fields)) >= 3:
        # Un solo ticker devuelve OHLCV plano
        return raw[["Close"]]

    logger.warning("No se pudo identificar columnas de cierre. Cols: %s", list(cols)[:8])
    return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def get_prices(years: int = 3) -> pd.DataFrame:
    """
    Descarga precios de cierre ajustado para TICKERS + BENCHMARK.
    Retorna DataFrame con índice DatetimeIndex y columnas = tickers.
    Incluye caché de 30 min y reintentos exponenciales.
    """
    tickers_all = TICKERS + [BENCHMARK]
    start = (datetime.today() - timedelta(days=365 * years + 30)).strftime("%Y-%m-%d")

    for attempt in range(1, 4):
        try:
            # Sin multi_level_index=False → yfinance siempre devuelve MultiIndex
            # para múltiples tickers, lo cual es el caso más predecible.
            raw = yf.download(
                tickers_all,
                start=start,
                auto_adjust=True,
                progress=False,
            )
            raw.index.name = "Date"

            if raw.empty:
                raise ValueError("yfinance devolvió DataFrame vacío")

            prices = _extract_close(raw, tickers_all)

            if prices.empty:
                raise ValueError("No se pudieron extraer precios de cierre")

            prices = prices.dropna(how="all").ffill()

            # Verificación mínima: al menos 20 filas para que los cálculos funcionen
            if len(prices) < 20:
                raise ValueError(f"Muy pocas filas descargadas: {len(prices)}")

            return prices

        except Exception as exc:
            logger.warning("get_prices intento %d/3: %s", attempt, exc)
            if attempt < 3:
                time.sleep(2 ** attempt)

    raise ConnectionError("No se pudieron descargar precios tras 3 intentos.")


@st.cache_data(ttl=1800, show_spinner=False)
def get_ohlcv(ticker: str, years: int = 3) -> pd.DataFrame:
    """
    Descarga OHLCV completo de UN solo ticker (para indicadores técnicos).
    Retorna DataFrame vacío en caso de error.
    """
    start = (datetime.today() - timedelta(days=365 * years + 30)).strftime("%Y-%m-%d")
    for attempt in range(1, 4):
        try:
            df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
            df.index.name = "Date"
            if df.empty:
                raise ValueError(f"Sin datos para {ticker}")
            # Un solo ticker siempre devuelve columnas OHLCV planas
            return df.dropna()
        except Exception as exc:
            logger.warning("get_ohlcv %s intento %d/3: %s", ticker, attempt, exc)
            if attempt < 3:
                time.sleep(2 ** attempt)
    return pd.DataFrame()


def get_returns(prices: pd.DataFrame | None = None, log: bool = True) -> pd.DataFrame:
    """
    Calcula rendimientos simples o logarítmicos.
    Si prices es None, descarga los precios del portafolio.
    """
    if prices is None:
        prices = get_prices()
    if prices.empty:
        return pd.DataFrame()
    if log:
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()


@st.cache_data(ttl=1800, show_spinner=False)
def get_risk_free_rate() -> dict:
    """
    Obtiene la tasa libre de riesgo anual desde ^IRX (T-Bill 3M).
    Retorna un diccionario con 'annual', 'daily', 'display', 'source' y 'date'.
    Si falla, usa 5.25% como valor de referencia.
    """
    try:
        irx = yf.download("^IRX", period="5d", progress=False, auto_adjust=True)
        irx.index.name = "Date"
        close_series = irx["Close"].dropna() if "Close" in irx.columns else pd.Series()
        if close_series.empty:
            raise ValueError("^IRX sin datos de Close")
        latest = float(close_series.iloc[-1])
        return {
            "annual" : latest / 100,
            "daily"  : latest / 100 / 252,
            "display": f"{latest:.2f}%",
            "source" : "Yahoo Finance · ^IRX",
            "date"   : irx.index[-1].strftime("%Y-%m-%d"),
        }
    except Exception as exc:
        logger.warning("get_risk_free_rate falló: %s — usando 5.25%%", exc)
        return {
            "annual" : 0.0525,
            "daily"  : 0.0525 / 252,
            "display": "5.25%",
            "source" : "Referencia (FRED)",
            "date"   : datetime.today().strftime("%Y-%m-%d"),
        }
