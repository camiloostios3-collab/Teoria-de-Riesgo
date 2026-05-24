"""
backend/app/db_models.py
Modelos ORM de SQLAlchemy — tablas persistentes en SQLite.
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from .database import Base


class Asset(Base):
    """Activo financiero registrado en el sistema."""

    __tablename__ = "assets"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(10), unique=True, nullable=False, index=True)
    nombre = Column(String(100), nullable=False)
    sector = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    prices = relationship("Price", back_populates="asset", cascade="all, delete-orphan")


class Price(Base):
    """Precio de cierre diario de un activo."""

    __tablename__ = "prices"

    id = Column(Integer, primary_key=True, index=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    fecha = Column(String(10), nullable=False)  # YYYY-MM-DD
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    asset = relationship("Asset", back_populates="prices")


class Portfolio(Base):
    """Portafolio guardado por el usuario."""

    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String(100), nullable=False)
    descripcion = Column(Text, nullable=True)
    tickers = Column(Text, nullable=False)   # JSON string: ["AAPL","JPM"]
    pesos = Column(Text, nullable=False)     # JSON string: [0.6,0.4]
    capital = Column(Float, default=100_000.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PredictionLog(Base):
    """Registro de predicciones realizadas por el modelo ML."""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(10), nullable=False)
    features_json = Column(Text, nullable=False)   # JSON de las features de entrada
    prediction = Column(Float, nullable=False)      # retorno predicho (%)
    model_version = Column(String(20), default="v1.0")
    created_at = Column(DateTime, default=datetime.utcnow)


class SignalLog(Base):
    """Registro histórico de señales de compra/venta generadas."""

    __tablename__ = "signal_logs"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(10), nullable=False)
    clasificacion = Column(String(30), nullable=False)
    score_compuesto = Column(Float, nullable=False)
    señales_json = Column(Text, nullable=False)     # JSON con detalle de señales
    created_at = Column(DateTime, default=datetime.utcnow)
