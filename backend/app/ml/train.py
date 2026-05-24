"""
backend/app/ml/train.py
Pipeline de entrenamiento del modelo ML para predicción de retornos.

Uso:
    cd backend
    python -m app.ml.train

Genera:  app/ml/model.joblib
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Configuración ────────────────────────────────────────────────────────────

TICKERS = ["AAPL", "JPM", "XOM", "JNJ", "AMZN", "MSFT", "GOOGL", "NVDA"]
YEARS = 5
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")


# ── Construcción de features ─────────────────────────────────────────────────

def build_features(returns: pd.Series) -> pd.DataFrame:
    """
    Genera features a partir de rendimientos logarítmicos diarios:
    - media_5d, media_20d: medias móviles de retornos
    - vol_5d, vol_20d: volatilidades móviles
    - momentum_5d: suma de retornos últimos 5 días
    - rsi_proxy: RSI simplificado de 14 períodos
    """
    df = pd.DataFrame({"ret": returns})

    df["media_5d"] = df["ret"].rolling(5).mean()
    df["media_20d"] = df["ret"].rolling(20).mean()
    df["vol_5d"] = df["ret"].rolling(5).std()
    df["vol_20d"] = df["ret"].rolling(20).std()
    df["momentum_5d"] = df["ret"].rolling(5).sum()

    delta = df["ret"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_proxy"] = 100 - 100 / (1 + gain / loss.replace(0, 1e-8))

    df["target"] = df["ret"].shift(-1)  # predecir el retorno del día siguiente

    return df.dropna()


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """Descarga datos y construye X, y para entrenamiento."""
    all_X, all_y = [], []
    start = (datetime.today() - timedelta(days=YEARS * 365 + 30)).strftime("%Y-%m-%d")

    for ticker in TICKERS:
        logger.info("Descargando %s ...", ticker)
        try:
            df = yf.download(ticker, start=start, auto_adjust=True,
                             progress=False, multi_level_index=False)
            if df.empty:
                logger.warning("Sin datos para %s", ticker)
                continue
            close = df["Close"].squeeze()
            log_ret = np.log(close / close.shift(1)).dropna()
            feat = build_features(log_ret)

            feature_cols = ["media_5d", "media_20d", "vol_5d", "vol_20d", "momentum_5d", "rsi_proxy"]
            X = feat[feature_cols].values
            y = feat["target"].values * 100  # en %

            all_X.append(X)
            all_y.append(y)
        except Exception as exc:
            logger.warning("Error con %s: %s", ticker, exc)

    if not all_X:
        raise RuntimeError("No se pudo descargar datos para ningún ticker")

    return np.vstack(all_X), np.concatenate(all_y)


def train_and_save() -> None:
    """Entrena el modelo y lo guarda en MODEL_PATH."""
    logger.info("Cargando datos para %d tickers ...", len(TICKERS))
    X, y = load_data()
    logger.info("Dataset: %d muestras, %d features", len(X), X.shape[1])

    # Validación cruzada de series temporales
    tscv = TimeSeriesSplit(n_splits=5)
    maes, r2s = [], []

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        pipeline.fit(X[train_idx], y[train_idx])
        preds = pipeline.predict(X[test_idx])
        mae = mean_absolute_error(y[test_idx], preds)
        r2 = r2_score(y[test_idx], preds)
        maes.append(mae)
        r2s.append(r2)
        logger.info("Fold %d — MAE: %.4f %%, R²: %.4f", fold, mae, r2)

    logger.info("CV promedio — MAE: %.4f %% ± %.4f, R²: %.4f", np.mean(maes), np.std(maes), np.mean(r2s))

    # Entrenar sobre todo el dataset
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)
    logger.info("Modelo guardado en %s", MODEL_PATH)


if __name__ == "__main__":
    train_and_save()
