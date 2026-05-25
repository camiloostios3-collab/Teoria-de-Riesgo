"""
backend/app/config.py
Configuración centralizada mediante BaseSettings (pydantic-settings).
Todas las variables se cargan desde el archivo .env; nunca se hardcodean.
"""

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── API keys externas ─────────────────────────────────────────────────────
    alpha_vantage_key: str = Field(default="", description="API key de Alpha Vantage")
    fred_api_key: str = Field(default="", description="API key de FRED (St. Louis Fed)")
    finnhub_key: str = Field(default="", description="API key de Finnhub")

    # ── Portafolio por defecto ─────────────────────────────────────────────────
    default_tickers: List[str] = Field(
        default=["AAPL", "JPM", "XOM", "JNJ", "AMZN"],
        description="Lista de tickers que conforman el portafolio por defecto",
    )
    benchmark: str = Field(default="^GSPC", description="Ticker del índice de referencia")
    default_years: int = Field(default=3, ge=1, le=10, description="Años de historia por defecto")

    # ── Parámetros de riesgo ───────────────────────────────────────────────────
    var_confidence_default: float = Field(
        default=0.95, ge=0.90, le=0.9999,
        description="Nivel de confianza por defecto para el VaR",
    )
    mc_simulations: int = Field(
        default=10_000, ge=1_000, le=100_000,
        description="Número de simulaciones Monte Carlo",
    )
    rf_fallback: float = Field(
        default=0.0525,
        description="Tasa libre de riesgo de respaldo cuando la API no responde",
    )

    # ── Indicadores técnicos ───────────────────────────────────────────────────
    sma_short: int = Field(default=20, ge=5, le=200)
    sma_long: int = Field(default=50, ge=10, le=500)
    ema_period: int = Field(default=20, ge=5, le=200)
    rsi_period: int = Field(default=14, ge=2, le=50)
    macd_fast: int = Field(default=12, ge=2, le=50)
    macd_slow: int = Field(default=26, ge=5, le=200)
    macd_signal: int = Field(default=9, ge=2, le=50)
    bb_period: int = Field(default=20, ge=5, le=200)
    bb_std: float = Field(default=2.0, ge=0.5, le=4.0)
    stoch_k: int = Field(default=14, ge=5, le=50)
    stoch_d: int = Field(default=3, ge=1, le=10)

    # ── CORS y seguridad ──────────────────────────────────────────────────────────
    allowed_origins: List[str] = Field(
        default=["*"],
        description=(
            "Orígenes CORS permitidos. En producción usa: "
            '["https://tu-frontend.onrender.com"]'
        ),
    )

    # ── Base de datos ─────────────────────────────────────────────────────────────
    database_url: str = Field(
        default="sqlite:///./risklab.db",
        description="URL de conexión SQLAlchemy. Para SQLite: sqlite:///./risklab.db",
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    """Retorna una instancia cacheada de Settings (singleton)."""
    return Settings()
