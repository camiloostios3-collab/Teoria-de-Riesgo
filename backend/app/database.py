"""
backend/app/database.py
Configuración de SQLAlchemy + generador de sesión para Depends().
La URL de la BD se toma de la variable de entorno DATABASE_URL (o Settings).
"""

import os
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from typing import Generator

# Toma DATABASE_URL del entorno para que funcione dentro de Docker/Render/CI.
# Fallback a SQLite local para desarrollo sin .env.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./risklab.db")

_connect_args: dict = {}
if DATABASE_URL.startswith("sqlite"):
    _connect_args = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,  # solo necesario para SQLite
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    """Generador de sesión de base de datos para inyección con Depends()."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Crea todas las tablas si no existen."""
    from . import db_models  # noqa: F401 — importar para registrar modelos
    Base.metadata.create_all(bind=engine)
