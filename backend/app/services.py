"""
backend/app/services.py
Clases de servicio que encapsulan toda la lógica de negocio:
  - DataService          → descarga y cacheo de datos (yfinance)
  - TechnicalIndicators  → SMA, EMA, RSI, MACD, Bollinger, Estocástico
  - RiskCalculator       → VaR (3 métodos), CVaR, Kupiec
  - CAPMCalculator       → Beta, Alpha de Jensen, rendimiento esperado
  - PortfolioAnalyzer    → Frontera eficiente, Mínima Varianza, Máx. Sharpe
  - SignalGenerator      → Señales automatizadas de compra/venta
  - MacroService         → Indicadores macroeconómicos vía APIs
  - GARCHService         → Modelos de volatilidad condicional
  - EWMAService          → Volatilidad EWMA con λ configurable
  - YieldCurveService    → Curva Nelson-Siegel desde FRED
  - BondService          → Duración de Macaulay, modificada y convexidad
  - OptionPricer         → Black-Scholes, 5 Greeks, volatilidad implícita
  - StressTester         → Pruebas de estrés con 3 escenarios
  - MLPredictor          → Predicción ML (Singleton) con registro en BD
"""

import functools
import json
import logging
import math
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from arch import arch_model
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# ── Mapa de metadatos de activos ──────────────────────────────────────────────

SECTOR_MAP: Dict[str, str] = {
    "AAPL": "Tecnología",    "MSFT": "Tecnología",   "GOOGL": "Tecnología",
    "META": "Tecnología",    "NVDA": "Tecnología",   "AMZN": "Consumo discrecional",
    "JPM": "Financiero",     "BAC": "Financiero",    "GS": "Financiero",
    "XOM": "Energía",        "CVX": "Energía",
    "JNJ": "Salud",          "PFE": "Salud",         "UNH": "Salud",
    "WMT": "Consumo básico",
}

NOMBRE_MAP: Dict[str, str] = {
    "AAPL": "Apple Inc.",          "JPM": "JPMorgan Chase",
    "XOM": "Exxon Mobil",          "JNJ": "Johnson & Johnson",
    "AMZN": "Amazon.com",          "MSFT": "Microsoft Corp.",
    "GOOGL": "Alphabet Inc.",      "META": "Meta Platforms",
    "NVDA": "NVIDIA Corp.",        "BAC": "Bank of America",
    "GS": "Goldman Sachs",         "CVX": "Chevron Corp.",
    "PFE": "Pfizer Inc.",          "UNH": "UnitedHealth Group",
    "WMT": "Walmart Inc.",
}

COLOR_MAP: Dict[str, str] = {
    "AAPL": "#A89060", "JPM": "#3D8B6E", "XOM": "#3A6B8A",
    "JNJ": "#5A4E7A",  "AMZN": "#8B4A4A",
}

# ── Caché en memoria (TTL = 30 min) ──────────────────────────────────────────

_cache: Dict[str, Tuple[Any, float]] = {}
_CACHE_TTL = 1800


def _cache_get(key: str) -> Optional[Any]:
    if key in _cache:
        value, ts = _cache[key]
        if time.time() - ts < _CACHE_TTL:
            return value
    return None


def _cache_set(key: str, value: Any) -> None:
    _cache[key] = (value, time.time())


# ═══════════════════════════════════════════════════════════════════════════════
#  Decorador personalizado: mide y registra el tiempo de ejecución
# ═══════════════════════════════════════════════════════════════════════════════

def timed(func):
    """Decorador que registra el tiempo de ejecución de un método de servicio."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.debug("%s completado en %.3fs", func.__qualname__, elapsed)
        return result
    return wrapper


# ═══════════════════════════════════════════════════════════════════════════════
#  DataService — descarga y cacheo de datos de mercado
# ═══════════════════════════════════════════════════════════════════════════════

class DataService:
    """Servicio central de datos: descarga precios de yfinance con caché interno."""

    def __init__(self, benchmark: str = "^GSPC") -> None:
        self.benchmark = benchmark

    def _download(self, tickers: List[str], start: str) -> pd.DataFrame:
        """Descarga datos con multi_level_index=False para columnas planas."""
        df = yf.download(
            tickers if len(tickers) > 1 else tickers[0],
            start=start,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )
        df.columns.name = None
        df.index.name = "Date"
        return df

    @timed
    def get_prices(self, tickers: List[str], years: int = 3) -> pd.DataFrame:
        """Retorna DataFrame de precios de cierre para los tickers dados (+benchmark).

        Incluye caché interno de 30 minutos y reintentos exponenciales.
        """
        key = f"prices:{'|'.join(sorted(tickers))}:{years}"
        cached = _cache_get(key)
        if cached is not None:
            return cached

        all_tickers = list(dict.fromkeys(tickers + [self.benchmark]))
        start = (datetime.today() - timedelta(days=years * 365 + 30)).strftime("%Y-%m-%d")

        for attempt in range(1, 4):
            try:
                raw = self._download(all_tickers, start)
                if raw.empty:
                    raise ValueError("yfinance devolvió DataFrame vacío")

                if isinstance(raw.columns, pd.MultiIndex):
                    prices = raw["Close"]
                else:
                    # Columnas planas (multi_level_index=False activo)
                    close_cols = [c for c in raw.columns if c in all_tickers]
                    if close_cols:
                        prices = raw[close_cols]
                    else:
                        prices = raw

                prices = prices.dropna(how="all").ffill()
                _cache_set(key, prices)
                return prices
            except Exception as exc:
                logger.warning("Intento %d/3 fallido: %s", attempt, exc)
                time.sleep(2 ** attempt)

        raise ConnectionError("No se pudieron descargar precios tras 3 intentos.")

    @timed
    def get_ohlcv(self, ticker: str, years: int = 3) -> pd.DataFrame:
        """Retorna OHLCV completo de un solo ticker (para indicadores técnicos)."""
        key = f"ohlcv:{ticker}:{years}"
        cached = _cache_get(key)
        if cached is not None:
            return cached

        start = (datetime.today() - timedelta(days=years * 365 + 30)).strftime("%Y-%m-%d")
        for attempt in range(1, 4):
            try:
                df = self._download([ticker], start)
                if df.empty:
                    raise ValueError(f"Sin datos para {ticker}")
                # Si llegara con MultiIndex, aplanar
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns.tolist()]
                result = df.dropna()
                _cache_set(key, result)
                return result
            except Exception as exc:
                logger.warning("Intento %d/3 para %s: %s", attempt, ticker, exc)
                time.sleep(2 ** attempt)

        raise ConnectionError(f"No se pudo descargar OHLCV de {ticker}.")

    @staticmethod
    def get_returns(prices: pd.DataFrame, log: bool = True) -> pd.DataFrame:
        """Calcula rendimientos simples o logarítmicos a partir de precios."""
        if log:
            return np.log(prices / prices.shift(1)).dropna()
        return prices.pct_change().dropna()

    @timed
    def get_risk_free_rate(self) -> float:
        """Obtiene la tasa libre de riesgo anual desde ^IRX (T-Bill 3M)."""
        key = "rf_rate"
        cached = _cache_get(key)
        if cached is not None:
            return cached
        try:
            irx = yf.download(
                "^IRX", period="5d", progress=False,
                auto_adjust=True, multi_level_index=False,
            )
            irx.columns.name = None
            val = float(irx["Close"].dropna().iloc[-1]) / 100
            _cache_set(key, val)
            return val
        except Exception:
            return 0.0525


# ═══════════════════════════════════════════════════════════════════════════════
#  TechnicalIndicators — cálculo de indicadores técnicos
# ═══════════════════════════════════════════════════════════════════════════════

class TechnicalIndicators:
    """Calcula indicadores técnicos estándar sobre series de precios."""

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(
        series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line, macd_line - signal_line

    @staticmethod
    def bollinger_bands(
        series: pd.Series, period: int = 20, std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        mid = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        return mid + std_dev * std, mid, mid - std_dev * std

    @staticmethod
    def stochastic(
        high: pd.Series, low: pd.Series, close: pd.Series,
        k_period: int = 14, d_period: int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        lowest = low.rolling(window=k_period).min()
        highest = high.rolling(window=k_period).max()
        k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
        return k, k.rolling(window=d_period).mean()

    @timed
    def compute_all(self, ohlcv: pd.DataFrame, settings: Any) -> Dict[str, Any]:
        """Computa todos los indicadores y los retorna como listas de dicts."""
        close = ohlcv["Close"].squeeze()
        high = ohlcv["High"].squeeze()
        low = ohlcv["Low"].squeeze()
        dates = ohlcv.index.strftime("%Y-%m-%d").tolist()

        def to_list(s: pd.Series) -> List[Optional[float]]:
            return [None if pd.isna(v) else round(float(v), 6) for v in s]

        prices_l = to_list(close)
        sma_s_l = to_list(self.sma(close, settings.sma_short))
        sma_l_l = to_list(self.sma(close, settings.sma_long))
        ema_l = to_list(self.ema(close, settings.ema_period))
        rsi_l = to_list(self.rsi(close, settings.rsi_period))
        ml, sl, hl = self.macd(close, settings.macd_fast, settings.macd_slow, settings.macd_signal)
        macd_l, sig_l, hist_l = to_list(ml), to_list(sl), to_list(hl)
        bb_up, bb_mid, bb_lo = self.bollinger_bands(close, settings.bb_period, settings.bb_std)
        bb_up_l, bb_mid_l, bb_lo_l = to_list(bb_up), to_list(bb_mid), to_list(bb_lo)
        k_s, d_s = self.stochastic(high, low, close, settings.stoch_k, settings.stoch_d)
        k_l, d_l = to_list(k_s), to_list(d_s)

        n = len(dates)
        return {
            "sma_corta":   [{"fecha": dates[i], "precio": prices_l[i], "sma": sma_s_l[i]} for i in range(n)],
            "sma_larga":   [{"fecha": dates[i], "precio": prices_l[i], "sma": sma_l_l[i]} for i in range(n)],
            "ema":         [{"fecha": dates[i], "precio": prices_l[i], "ema": ema_l[i]} for i in range(n)],
            "rsi":         [{"fecha": dates[i], "rsi": rsi_l[i]} for i in range(n)],
            "macd":        [{"fecha": dates[i], "macd": macd_l[i], "señal": sig_l[i], "histograma": hist_l[i]} for i in range(n)],
            "bollinger":   [{"fecha": dates[i], "precio": prices_l[i], "superior": bb_up_l[i], "media": bb_mid_l[i], "inferior": bb_lo_l[i]} for i in range(n)],
            "estocastico": [{"fecha": dates[i], "k": k_l[i], "d": d_l[i]} for i in range(n)],
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  CAPMCalculator — Beta, Alpha de Jensen, rendimiento esperado
# ═══════════════════════════════════════════════════════════════════════════════

class CAPMCalculator:
    """Calcula el modelo CAPM para cada activo del portafolio."""

    @timed
    def compute(
        self,
        returns: pd.DataFrame,
        benchmark_col: str,
        tickers: List[str],
        rf: float,
    ) -> Dict[str, Any]:
        market = returns[benchmark_col].dropna()
        ann_market = float(market.mean() * 252)
        market_premium = ann_market - rf
        activos: List[Dict[str, Any]] = []

        for ticker in tickers:
            if ticker not in returns.columns:
                continue
            asset = returns[ticker].dropna()
            idx = asset.index.intersection(market.index)
            x, y = market.loc[idx].values, asset.loc[idx].values

            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            beta = float(slope)
            t_crit = stats.t.ppf(0.975, df=len(x) - 2)

            if beta > 1.2:
                clasificacion = "Agresivo"
            elif beta < 0.8:
                clasificacion = "Defensivo"
            else:
                clasificacion = "Neutro"

            activos.append({
                "ticker": ticker,
                "beta": round(beta, 4),
                "beta_ic_inferior": round(beta - t_crit * std_err, 4),
                "beta_ic_superior": round(beta + t_crit * std_err, 4),
                "alpha_jensen": round(float(intercept * 252), 6),
                "r_cuadrado": round(float(r_value ** 2), 4),
                "p_value": round(float(p_value), 6),
                "retorno_esperado_capm": round(rf + beta * market_premium, 6),
                "clasificacion": clasificacion,
            })

        return {
            "tasa_libre_riesgo": round(rf, 6),
            "prima_mercado": round(market_premium, 6),
            "retorno_mercado": round(ann_market, 6),
            "activos": activos,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  RiskCalculator — VaR (3 métodos), CVaR, Kupiec
# ═══════════════════════════════════════════════════════════════════════════════

class RiskCalculator:
    """Calcula métricas de riesgo: VaR paramétrico, histórico, Monte Carlo y CVaR."""

    @timed
    def compute_var(
        self,
        returns: pd.DataFrame,
        tickers: List[str],
        weights: List[float],
        confidence: float,
        capital: float,
        mc_simulations: int = 10_000,
    ) -> Dict[str, Any]:
        w = np.array(weights)
        port_rets = returns[tickers].dropna() @ w

        mu = float(port_rets.mean())
        sigma = float(port_rets.std())
        z = stats.norm.ppf(1 - confidence)

        # ── Método 1: Paramétrico ──────────────────────────────────────────────
        var_p = -(mu + z * sigma)
        cvar_p = -(mu - sigma * stats.norm.pdf(z) / (1 - confidence))

        # ── Método 2: Histórico ───────────────────────────────────────────────
        var_h = float(-np.percentile(port_rets, (1 - confidence) * 100))
        tail_h = port_rets[port_rets <= -var_h]
        cvar_h = float(-tail_h.mean()) if len(tail_h) > 0 else var_h

        # ── Método 3: Monte Carlo ─────────────────────────────────────────────
        rng = np.random.default_rng(42)
        sim = rng.normal(mu, sigma, mc_simulations)
        var_mc = float(-np.percentile(sim, (1 - confidence) * 100))
        tail_mc = sim[sim <= -var_mc]
        cvar_mc = float(-tail_mc.mean()) if len(tail_mc) > 0 else var_mc

        ann = np.sqrt(252)
        resultados = [
            {
                "metodo": "Paramétrico",
                "var_diario_pct": round(var_p, 6),
                "var_diario_usd": round(var_p * capital, 2),
                "var_anual_pct": round(var_p * ann, 6),
                "cvar_diario_pct": round(cvar_p, 6),
                "cvar_diario_usd": round(cvar_p * capital, 2),
                "confianza": confidence,
            },
            {
                "metodo": "Histórico",
                "var_diario_pct": round(var_h, 6),
                "var_diario_usd": round(var_h * capital, 2),
                "var_anual_pct": round(var_h * ann, 6),
                "cvar_diario_pct": round(cvar_h, 6),
                "cvar_diario_usd": round(cvar_h * capital, 2),
                "confianza": confidence,
            },
            {
                "metodo": "Monte Carlo",
                "var_diario_pct": round(var_mc, 6),
                "var_diario_usd": round(var_mc * capital, 2),
                "var_anual_pct": round(var_mc * ann, 6),
                "cvar_diario_pct": round(cvar_mc, 6),
                "cvar_diario_usd": round(cvar_mc * capital, 2),
                "confianza": confidence,
            },
        ]

        # ── Test de Kupiec ────────────────────────────────────────────────────
        n = len(port_rets)
        violations = int((port_rets < -var_h).sum())
        exp_rate = 1 - confidence
        act_rate = violations / n if n > 0 else 0.0

        if 0 < violations < n:
            lr = -2 * (
                violations * np.log(exp_rate / act_rate)
                + (n - violations) * np.log((1 - exp_rate) / (1 - act_rate))
            )
        else:
            lr = 0.0
        pval_kup = float(1 - stats.chi2.cdf(lr, df=1))

        kupiec = {
            "n_violaciones": violations,
            "tasa_violacion": round(act_rate, 6),
            "tasa_esperada": round(exp_rate, 6),
            "lr_statistic": round(float(lr), 4),
            "p_value": round(pval_kup, 4),
            "resultado": (
                "No rechaza H₀ — modelo bien calibrado"
                if pval_kup > 0.05
                else "Rechaza H₀ — modelo subestima el riesgo"
            ),
        }

        return {
            "tickers": tickers,
            "pesos": weights,
            "capital": capital,
            "confianza": confidence,
            "resultados": resultados,
            "kupiec": kupiec,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  PortfolioAnalyzer — Frontera eficiente de Markowitz
# ═══════════════════════════════════════════════════════════════════════════════

class PortfolioAnalyzer:
    """Simulación de Monte Carlo + optimización para la frontera eficiente."""

    @timed
    def compute_frontier(
        self,
        returns: pd.DataFrame,
        tickers: List[str],
        n_portfolios: int,
        rf: float,
    ) -> Dict[str, Any]:
        rets = returns[tickers].dropna()
        mean_ann = rets.mean() * 252
        cov_ann = rets.cov() * 252
        n = len(tickers)

        # ── Simulación aleatoria ───────────────────────────────────────────────
        rng = np.random.default_rng(42)
        port_r, port_v, port_s, port_w = [], [], [], []
        for _ in range(n_portfolios):
            w = rng.dirichlet(np.ones(n))
            r = float(w @ mean_ann)
            v = float(np.sqrt(w @ cov_ann @ w))
            s = (r - rf) / v if v > 0 else 0.0
            port_r.append(round(r, 6))
            port_v.append(round(v, 6))
            port_s.append(round(s, 6))
            port_w.append(w.round(6).tolist())

        bounds = [(0, 1)] * n
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        w0 = np.ones(n) / n

        # ── Portafolio de mínima varianza ──────────────────────────────────────
        def vol(w: np.ndarray) -> float:
            return float(np.sqrt(w @ cov_ann @ w))

        mv = minimize(vol, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        mv_w = mv.x
        mv_r = float(mv_w @ mean_ann)
        mv_v = vol(mv_w)
        mv_s = (mv_r - rf) / mv_v if mv_v > 0 else 0.0

        # ── Portafolio de máximo Sharpe ───────────────────────────────────────
        def neg_sharpe(w: np.ndarray) -> float:
            r = w @ mean_ann
            v = np.sqrt(w @ cov_ann @ w)
            return -(r - rf) / v if v > 0 else 0.0

        ms = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        ms_w = ms.x
        ms_r = float(ms_w @ mean_ann)
        ms_v = vol(ms_w)
        ms_s = (ms_r - rf) / ms_v if ms_v > 0 else 0.0

        optimos = [
            {
                "nombre": "Mínima Varianza",
                "retorno": round(mv_r, 6), "volatilidad": round(mv_v, 6), "sharpe": round(mv_s, 6),
                "composicion": {tickers[i]: round(float(mv_w[i]), 4) for i in range(n)},
            },
            {
                "nombre": "Máximo Sharpe",
                "retorno": round(ms_r, 6), "volatilidad": round(ms_v, 6), "sharpe": round(ms_s, 6),
                "composicion": {tickers[i]: round(float(ms_w[i]), 4) for i in range(n)},
            },
        ]

        return {
            "tickers": tickers,
            "n_simulados": n_portfolios,
            "portafolios": [
                {"retorno": port_r[i], "volatilidad": port_v[i], "sharpe": port_s[i], "pesos": port_w[i]}
                for i in range(n_portfolios)
            ],
            "optimos": optimos,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  SignalGenerator — señales automatizadas de compra/venta
# ═══════════════════════════════════════════════════════════════════════════════

class SignalGenerator:
    """Genera señales de compra/venta evaluando 5 indicadores técnicos."""

    def __init__(self, ti: TechnicalIndicators) -> None:
        self._ti = ti

    @timed
    def compute_signals(
        self, ohlcv: pd.DataFrame, ticker: str, settings: Any
    ) -> Dict[str, Any]:
        close = ohlcv["Close"].squeeze()
        high = ohlcv["High"].squeeze()
        low = ohlcv["Low"].squeeze()

        # ── 1. MACD ────────────────────────────────────────────────────────────
        ml, sl, _ = self._ti.macd(close, settings.macd_fast, settings.macd_slow, settings.macd_signal)
        macd_val, sig_val = float(ml.iloc[-1]), float(sl.iloc[-1])
        if macd_val > sig_val:
            señal_macd, score_macd = "COMPRA", 1
        elif macd_val < sig_val:
            señal_macd, score_macd = "VENTA", -1
        else:
            señal_macd, score_macd = "NEUTRAL", 0

        # ── 2. RSI ─────────────────────────────────────────────────────────────
        rsi_s = self._ti.rsi(close, settings.rsi_period)
        rsi_val = float(rsi_s.iloc[-1])
        if rsi_val > 70:
            señal_rsi, score_rsi = "VENTA", -1
        elif rsi_val < 30:
            señal_rsi, score_rsi = "COMPRA", 1
        else:
            señal_rsi, score_rsi = "NEUTRAL", 0

        # ── 3. Bollinger ───────────────────────────────────────────────────────
        bb_up, _, bb_lo = self._ti.bollinger_bands(close, settings.bb_period, settings.bb_std)
        precio = float(close.iloc[-1])
        if precio >= float(bb_up.iloc[-1]):
            señal_bb, score_bb = "VENTA", -1
        elif precio <= float(bb_lo.iloc[-1]):
            señal_bb, score_bb = "COMPRA", 1
        else:
            señal_bb, score_bb = "NEUTRAL", 0

        # ── 4. Cruce de medias (Golden/Death Cross) ───────────────────────────
        sma_s = self._ti.sma(close, settings.sma_short)
        sma_l = self._ti.sma(close, settings.sma_long)
        sma_s_cur, sma_l_cur = float(sma_s.iloc[-1]), float(sma_l.iloc[-1])
        sma_s_prv = float(sma_s.iloc[-2]) if len(sma_s) >= 2 else sma_s_cur
        sma_l_prv = float(sma_l.iloc[-2]) if len(sma_l) >= 2 else sma_l_cur

        if sma_s_cur > sma_l_cur and sma_s_prv <= sma_l_prv:
            señal_cross, score_cross = "COMPRA", 1.0
        elif sma_s_cur < sma_l_cur and sma_s_prv >= sma_l_prv:
            señal_cross, score_cross = "VENTA", -1.0
        elif sma_s_cur > sma_l_cur:
            señal_cross, score_cross = "SESGO_ALCISTA", 0.5
        else:
            señal_cross, score_cross = "SESGO_BAJISTA", -0.5

        # ── 5. Estocástico ─────────────────────────────────────────────────────
        k, d = self._ti.stochastic(high, low, close, settings.stoch_k, settings.stoch_d)
        k_val, d_val = float(k.iloc[-1]), float(d.iloc[-1])
        if k_val < 20 and k_val > d_val:
            señal_stoch, score_stoch = "COMPRA", 1
        elif k_val > 80 and k_val < d_val:
            señal_stoch, score_stoch = "VENTA", -1
        else:
            señal_stoch, score_stoch = "NEUTRAL", 0

        total = score_macd + score_rsi + score_bb + score_cross + score_stoch
        if total >= 2:
            clasificacion, color = "COMPRA FUERTE",  "#16a34a"
        elif total >= 0.5:
            clasificacion, color = "SESGO ALCISTA",  "#65a30d"
        elif total <= -2:
            clasificacion, color = "VENTA FUERTE",   "#dc2626"
        elif total <= -0.5:
            clasificacion, color = "SESGO BAJISTA",  "#ea580c"
        else:
            clasificacion, color = "NEUTRAL",        "#6b7280"

        return {
            "ticker": ticker,
            "señales": [
                {"indicador": "MACD",            "señal": señal_macd,  "valor_actual": round(macd_val, 6), "descripcion": f"MACD={macd_val:.4f} | Señal={sig_val:.4f}"},
                {"indicador": "RSI",             "señal": señal_rsi,   "valor_actual": round(rsi_val, 2),  "descripcion": f"RSI={rsi_val:.1f} (30=sobreventa, 70=sobrecompra)"},
                {"indicador": "Bollinger",        "señal": señal_bb,    "valor_actual": round(precio, 4),   "descripcion": f"Precio={precio:.2f} | Banda sup={float(bb_up.iloc[-1]):.2f}"},
                {"indicador": "Cruce de Medias",  "señal": señal_cross, "valor_actual": round(sma_s_cur, 4),"descripcion": f"SMA{settings.sma_short}={sma_s_cur:.2f} | SMA{settings.sma_long}={sma_l_cur:.2f}"},
                {"indicador": "Estocástico",      "señal": señal_stoch, "valor_actual": round(k_val, 2),    "descripcion": f"%K={k_val:.1f} | %D={d_val:.1f}"},
            ],
            "score_compuesto": round(total / 5, 4),
            "clasificacion": clasificacion,
            "color": color,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  MacroService — indicadores macroeconómicos vía APIs
# ═══════════════════════════════════════════════════════════════════════════════

class MacroService:
    """Obtiene indicadores macroeconómicos actualizados desde Yahoo Finance."""

    MACRO_TICKERS: Dict[str, Tuple[str, str, str]] = {
        "^IRX":    ("Tasa Libre de Riesgo (T-Bill 3M)", "%",   "Yahoo Finance"),
        "^TNX":    ("Tasa del Tesoro 10 Años",           "%",   "Yahoo Finance"),
        "^VIX":    ("Índice de Volatilidad VIX",          "pts", "Yahoo Finance"),
        "USDCOP=X":("Tasa de Cambio USD/COP",             "COP", "Yahoo Finance"),
        "EURUSD=X":("Tasa EUR/USD",                       "USD", "Yahoo Finance"),
        "GC=F":    ("Oro (Gold Futures)",                 "USD/oz", "Yahoo Finance"),
    }

    @timed
    def get_macro_indicators(self, rf_fallback: float = 0.0525) -> Dict[str, Any]:
        rf = rf_fallback
        indicadores: List[Dict[str, Any]] = []
        timestamp = datetime.now().isoformat()

        for sym, (nombre, unidad, fuente) in self.MACRO_TICKERS.items():
            try:
                raw = yf.download(
                    sym, period="5d", progress=False,
                    auto_adjust=True, multi_level_index=False,
                )
                raw.columns.name = None
                if not raw.empty and "Close" in raw.columns:
                    val = float(raw["Close"].dropna().iloc[-1])
                    if sym == "^IRX":
                        rf = val / 100
                    indicadores.append({
                        "nombre": nombre,
                        "valor": round(val, 4),
                        "unidad": unidad,
                        "fuente": fuente,
                        "descripcion": f"Último valor: {val:.4f} {unidad}",
                    })
                else:
                    raise ValueError("DataFrame vacío")
            except Exception as exc:
                logger.warning("Macro %s no disponible: %s", sym, exc)
                indicadores.append({
                    "nombre": nombre, "valor": 0.0, "unidad": unidad,
                    "fuente": fuente, "descripcion": "No disponible",
                })

        return {
            "tasa_libre_riesgo": round(rf, 6),
            "indicadores": indicadores,
            "timestamp": timestamp,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  GARCHService — modelos de volatilidad condicional (endpoint adicional)
# ═══════════════════════════════════════════════════════════════════════════════

class GARCHService:
    """Ajusta y compara modelos ARCH/GARCH para un activo."""

    SPECS: List[Dict[str, Any]] = [
        {"nombre": "ARCH(1)",        "vol": "ARCH",  "p": 1, "q": 0, "o": 0},
        {"nombre": "GARCH(1,1)",     "vol": "GARCH", "p": 1, "q": 1, "o": 0},
        {"nombre": "GJR-GARCH(1,1)", "vol": "GARCH", "p": 1, "q": 1, "o": 1},
        {"nombre": "EGARCH(1,1)",    "vol": "EGARCH","p": 1, "q": 1, "o": 1},
    ]

    @timed
    def fit_models(self, returns: pd.Series) -> Dict[str, Any]:
        r = returns * 100  # escalar para estabilidad numérica
        resultados: List[Dict[str, Any]] = []
        mejor_aic = np.inf
        mejor_nombre = ""

        for spec in self.SPECS:
            try:
                am = arch_model(
                    r, vol=spec["vol"],
                    p=spec["p"], q=spec["q"], o=spec["o"],
                    dist="t",
                )
                res = am.fit(disp="off", show_warning=False)
                cond_vol = res.conditional_volatility
                pronostico = res.forecast(horizon=1)
                vol_1d_scaled = float(np.sqrt(pronostico.variance.iloc[-1, 0]))
                vol_1d = vol_1d_scaled / 100
                vol_ann = vol_1d * np.sqrt(252)

                params = res.params
                omega = float(params.get("omega", 0))
                alpha = float(params.get("alpha[1]", 0))
                beta_ = float(params.get("beta[1]", np.nan)) if "beta[1]" in params else None
                gamma = float(params.get("gamma[1]", np.nan)) if "gamma[1]" in params else None
                persist = (alpha + float(beta_) if beta_ is not None else None)

                resultados.append({
                    "nombre": spec["nombre"],
                    "aic": round(float(res.aic), 4),
                    "bic": round(float(res.bic), 4),
                    "log_likelihood": round(float(res.loglikelihood), 4),
                    "omega": round(omega, 8),
                    "alpha": round(alpha, 6),
                    "beta": round(beta_, 6) if beta_ is not None else None,
                    "gamma": round(gamma, 6) if gamma is not None else None,
                    "persistencia": round(persist, 6) if persist is not None else None,
                    "pronostico_vol_1d": round(vol_1d, 8),
                    "pronostico_vol_anual": round(vol_ann, 6),
                })

                if float(res.aic) < mejor_aic:
                    mejor_aic = float(res.aic)
                    mejor_nombre = spec["nombre"]
                    resid_std = cond_vol.values

            except Exception as exc:
                logger.warning("GARCH spec %s falló: %s", spec["nombre"], exc)

        # Diagnóstico de residuos del mejor modelo
        jb_stat, jb_p = 0.0, 1.0
        if len(resultados) > 0 and "resid_std" in dir():
            try:
                jb_stat, jb_p = stats.jarque_bera(resid_std)
            except Exception:
                pass

        return {
            "especificaciones": resultados,
            "mejor_modelo": mejor_nombre,
            "jarque_bera_residuos": round(float(jb_stat), 4),
            "jarque_bera_pvalue": round(float(jb_p), 6),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  EWMAService — Volatilidad EWMA con λ configurable
# ═══════════════════════════════════════════════════════════════════════════════

class EWMAService:
    """Calcula volatilidad EWMA (Exponentially Weighted Moving Average)."""

    @timed
    def compute(self, returns: pd.Series, lam: float = 0.94) -> Dict[str, Any]:
        """
        Calcula la varianza EWMA: σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}
        λ=0.94 es el valor RiskMetrics para datos diarios.
        """
        r = returns.values
        n = len(r)
        var_ewma = np.zeros(n)
        var_ewma[0] = float(np.var(r[:20])) if n >= 20 else float(np.var(r))

        for t in range(1, n):
            var_ewma[t] = lam * var_ewma[t - 1] + (1 - lam) * r[t - 1] ** 2

        vol_ewma = np.sqrt(var_ewma)
        fechas = [str(d.date()) for d in returns.index]

        # Pronóstico a 1 día (último valor)
        vol_1d = float(vol_ewma[-1])
        vol_anual = vol_1d * math.sqrt(252)

        serie = [
            {"fecha": fechas[i], "volatilidad_diaria": round(float(vol_ewma[i]), 8)}
            for i in range(n)
        ]

        return {
            "lambda": lam,
            "volatilidad_1d": round(vol_1d, 8),
            "volatilidad_anual": round(vol_anual, 6),
            "n_observaciones": n,
            "serie": serie,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  YieldCurveService — Curva de rendimiento Nelson-Siegel desde FRED
# ═══════════════════════════════════════════════════════════════════════════════

class YieldCurveService:
    """Obtiene tasas del Tesoro desde FRED y ajusta el modelo Nelson-Siegel."""

    # Series FRED: plazo en años → serie ID
    FRED_SERIES: Dict[float, str] = {
        0.25: "DGS3MO",
        1.0:  "DGS1",
        2.0:  "DGS2",
        5.0:  "DGS5",
        10.0: "DGS10",
        30.0: "DGS30",
    }
    FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

    @timed
    def get_rates(self, fred_api_key: str = "") -> Dict[float, float]:
        """Descarga la última observación de cada plazo desde FRED."""
        rates: Dict[float, float] = {}
        for plazo, series_id in self.FRED_SERIES.items():
            try:
                if fred_api_key:
                    params = {
                        "series_id": series_id,
                        "api_key": fred_api_key,
                        "file_type": "json",
                        "sort_order": "desc",
                        "limit": 5,
                    }
                    resp = requests.get(self.FRED_URL, params=params, timeout=10)
                    resp.raise_for_status()
                    obs = resp.json().get("observations", [])
                    for o in obs:
                        if o["value"] != ".":
                            rates[plazo] = float(o["value"]) / 100
                            break
                else:
                    # Fallback: yfinance no tiene DGS series; usar tasas de referencia
                    rates[plazo] = self._yf_fallback(plazo)
            except Exception as exc:
                logger.warning("FRED %s no disponible: %s — usando fallback", series_id, exc)
                rates[plazo] = self._yf_fallback(plazo)
        return rates

    @staticmethod
    def _yf_fallback(plazo: float) -> float:
        """Tasas de referencia aproximadas cuando FRED no está disponible."""
        # Curva típica US Treasuries (aproximación)
        base = {0.25: 0.053, 1.0: 0.052, 2.0: 0.049, 5.0: 0.047, 10.0: 0.046, 30.0: 0.048}
        return base.get(plazo, 0.045)

    @staticmethod
    def nelson_siegel(plazo: float, b0: float, b1: float, b2: float, tau: float) -> float:
        """Modelo Nelson-Siegel: y(T) = β₀ + β₁·(1-e^{-T/τ})/(T/τ) + β₂·[(1-e^{-T/τ})/(T/τ) - e^{-T/τ}]"""
        if plazo <= 0:
            return b0 + b1
        x = plazo / tau
        factor1 = (1 - math.exp(-x)) / x
        factor2 = factor1 - math.exp(-x)
        return b0 + b1 * factor1 + b2 * factor2

    @timed
    def fit(self, rates: Dict[float, float]) -> Dict[str, Any]:
        """Ajusta Nelson-Siegel y retorna parámetros + curva interpolada."""
        plazos = sorted(rates.keys())
        yields = [rates[p] for p in plazos]

        def residuals(params: np.ndarray) -> float:
            b0, b1, b2, tau = params
            if tau <= 0 or b0 <= 0:
                return 1e10
            pred = [self.nelson_siegel(p, b0, b1, b2, tau) for p in plazos]
            return float(sum((p - y) ** 2 for p, y in zip(pred, yields)))

        x0 = np.array([0.05, -0.02, 0.01, 1.5])
        result = minimize(residuals, x0, method="Nelder-Mead",
                          options={"maxiter": 5000, "xatol": 1e-8})
        b0, b1, b2, tau = result.x
        tau = max(tau, 0.01)

        # Curva interpolada en plazos estándar
        plazos_curva = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
        curva = [
            {"plazo": p, "tasa_ns": round(self.nelson_siegel(p, b0, b1, b2, tau), 6),
             "tasa_observada": round(rates.get(p, float("nan")), 6) if p in rates else None}
            for p in plazos_curva
        ]

        return {
            "parametros": {
                "beta_0": round(float(b0), 6),
                "beta_1": round(float(b1), 6),
                "beta_2": round(float(b2), 6),
                "tau": round(float(tau), 6),
            },
            "tasas_observadas": {str(p): round(r, 6) for p, r in rates.items()},
            "curva": curva,
            "rmse": round(math.sqrt(result.fun / len(plazos)), 6),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  BondService — Duración de Macaulay, Modificada y Convexidad
# ═══════════════════════════════════════════════════════════════════════════════

class BondService:
    """Calcula métricas de renta fija para un bono de cupón fijo."""

    @timed
    def compute(
        self,
        face_value: float,
        coupon_rate: float,
        ytm: float,
        periods: int,
        frequency: int = 2,
    ) -> Dict[str, Any]:
        """
        Calcula precio, duración Macaulay, duración modificada y convexidad.

        Args:
            face_value: Valor nominal del bono (USD)
            coupon_rate: Tasa de cupón anual (fracción, ej: 0.05 = 5%)
            ytm: Yield to maturity anual (fracción)
            periods: Número total de períodos de pago (semestres si freq=2)
            frequency: Pagos por año (2=semestral, 1=anual, 4=trimestral)
        """
        c = face_value * coupon_rate / frequency  # cupón por período
        y = ytm / frequency                         # yield por período

        precio = 0.0
        macaulay_num = 0.0
        convexidad_num = 0.0

        flujos = []
        for t in range(1, periods + 1):
            cf = c + (face_value if t == periods else 0)
            pv = cf / (1 + y) ** t
            t_years = t / frequency
            precio += pv
            macaulay_num += t_years * pv
            convexidad_num += t_years * (t_years + 1 / frequency) * pv
            flujos.append({
                "periodo": t,
                "año": round(t_years, 4),
                "flujo": round(cf, 4),
                "pv_flujo": round(pv, 4),
            })

        macaulay = macaulay_num / precio
        modified = macaulay / (1 + y)
        convexidad = convexidad_num / (precio * (1 + y) ** 2)

        # Sensibilidad DV01 (cambio en precio por 1 pb en ytm)
        dv01 = modified * precio * 0.0001

        return {
            "precio": round(precio, 4),
            "duracion_macaulay": round(macaulay, 6),
            "duracion_modificada": round(modified, 6),
            "convexidad": round(convexidad, 6),
            "dv01": round(dv01, 4),
            "flujos": flujos,
            "parametros": {
                "face_value": face_value,
                "coupon_rate": coupon_rate,
                "ytm": ytm,
                "periods": periods,
                "frequency": frequency,
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  OptionPricer — Black-Scholes, 5 Greeks, Volatilidad Implícita
# ═══════════════════════════════════════════════════════════════════════════════

class OptionPricer:
    """Valúa opciones europeas con Black-Scholes y calcula los 5 Greeks."""

    @staticmethod
    def _d1_d2(S: float, K: float, r: float, sigma: float, T: float) -> Tuple[float, float]:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2

    @timed
    def price(
        self,
        S: float,      # Precio actual del subyacente
        K: float,      # Precio de ejercicio
        r: float,      # Tasa libre de riesgo (anual)
        sigma: float,  # Volatilidad anual
        T: float,      # Tiempo al vencimiento (años)
        option_type: str = "call",
    ) -> Dict[str, Any]:
        """Precio Black-Scholes y 5 Greeks (Delta, Gamma, Vega, Theta, Rho)."""
        if T <= 0:
            intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
            return self._zero_time(intrinsic, S, K, option_type)

        d1, d2 = self._d1_d2(S, K, r, sigma, T)
        nd1, nd2 = stats.norm.cdf(d1), stats.norm.cdf(d2)
        nd1n, nd2n = stats.norm.cdf(-d1), stats.norm.cdf(-d2)
        pdf_d1 = stats.norm.pdf(d1)

        if option_type == "call":
            precio = S * nd1 - K * math.exp(-r * T) * nd2
            delta = nd1
            theta = (
                -(S * pdf_d1 * sigma) / (2 * math.sqrt(T))
                - r * K * math.exp(-r * T) * nd2
            ) / 365
            rho = K * T * math.exp(-r * T) * nd2 / 100
        else:  # put
            precio = K * math.exp(-r * T) * nd2n - S * nd1n
            delta = nd1 - 1
            theta = (
                -(S * pdf_d1 * sigma) / (2 * math.sqrt(T))
                + r * K * math.exp(-r * T) * nd2n
            ) / 365
            rho = -K * T * math.exp(-r * T) * nd2n / 100

        gamma = pdf_d1 / (S * sigma * math.sqrt(T))
        vega = S * pdf_d1 * math.sqrt(T) / 100

        return {
            "precio": round(precio, 4),
            "delta": round(delta, 6),
            "gamma": round(gamma, 6),
            "vega": round(vega, 6),
            "theta": round(theta, 6),
            "rho": round(rho, 6),
            "d1": round(d1, 6),
            "d2": round(d2, 6),
            "tipo": option_type,
            "parametros": {"S": S, "K": K, "r": r, "sigma": sigma, "T": T},
        }

    def implied_volatility(
        self,
        market_price: float,
        S: float,
        K: float,
        r: float,
        T: float,
        option_type: str = "call",
        tol: float = 1e-6,
        max_iter: int = 100,
    ) -> float:
        """Volatilidad implícita via método de Newton-Raphson."""
        sigma = 0.3  # valor inicial
        for _ in range(max_iter):
            result = self.price(S, K, r, sigma, T, option_type)
            bs_price = result["precio"]
            vega = result["vega"] * 100  # vega en dólares por unidad de sigma
            if abs(vega) < 1e-10:
                break
            diff = bs_price - market_price
            if abs(diff) < tol:
                break
            sigma -= diff / vega
            sigma = max(sigma, 1e-5)
        return round(float(sigma), 6)

    @staticmethod
    def _zero_time(intrinsic: float, S: float, K: float, option_type: str) -> Dict[str, Any]:
        delta = 1.0 if (option_type == "call" and S > K) else (0.0 if option_type == "call" else -1.0)
        return {
            "precio": round(intrinsic, 4),
            "delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0,
            "d1": 0.0, "d2": 0.0, "tipo": option_type,
            "parametros": {"S": S, "K": K, "r": 0.0, "sigma": 0.0, "T": 0.0},
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  StressTester — Pruebas de estrés con 3 escenarios históricos/hipotéticos
# ═══════════════════════════════════════════════════════════════════════════════

class StressTester:
    """Aplica escenarios de estrés a un portafolio y calcula el impacto en P&L."""

    # Escenarios: shocks diarios por activo (fracción)
    ESCENARIOS = {
        "shock_tasas": {
            "descripcion": "Shock de tasas de interés +200 pb (estilo Taper Tantrum 2013)",
            "equity_shock": -0.08,
            "volatility_mult": 1.5,
            "bond_shock": -0.05,
        },
        "crash_mercado": {
            "descripcion": "Crash de mercado -20 % (estilo COVID-19 Mar 2020)",
            "equity_shock": -0.20,
            "volatility_mult": 3.0,
            "bond_shock": 0.03,
        },
        "shock_volatilidad": {
            "descripcion": "Explosión de volatilidad +150 % (estilo VIX spike Oct 2008)",
            "equity_shock": -0.12,
            "volatility_mult": 2.5,
            "bond_shock": 0.01,
        },
    }

    @timed
    def compute(
        self,
        returns: pd.DataFrame,
        tickers: List[str],
        weights: List[float],
        capital: float,
    ) -> Dict[str, Any]:
        """Aplica los 3 escenarios de estrés y compara con métricas base."""
        w = np.array(weights)
        port_rets = returns[tickers].dropna() @ w
        sigma_base = float(port_rets.std())
        mu_base = float(port_rets.mean())

        # VaR histórico base al 95%
        var_base = float(-np.percentile(port_rets, 5))

        resultados = []
        for nombre, params in self.ESCENARIOS.items():
            shock = float(params["equity_shock"])
            vol_mult = float(params["volatility_mult"])

            # P&L del portafolio bajo el escenario
            pnl_pct = shock
            pnl_usd = pnl_pct * capital

            # VaR estresado: asume que la volatilidad escala con el multiplicador
            sigma_stress = sigma_base * vol_mult
            z95 = stats.norm.ppf(0.05)
            var_stress = -(mu_base + z95 * sigma_stress)
            var_stress_usd = var_stress * capital

            resultados.append({
                "escenario": nombre,
                "descripcion": params["descripcion"],
                "shock_equity_pct": round(shock * 100, 2),
                "multiplicador_volatilidad": vol_mult,
                "pnl_pct": round(pnl_pct * 100, 2),
                "pnl_usd": round(pnl_usd, 2),
                "var_estresado_pct": round(var_stress * 100, 4),
                "var_estresado_usd": round(var_stress_usd, 2),
                "incremento_var_pct": round((var_stress - var_base) / var_base * 100, 2) if var_base > 0 else 0.0,
            })

        return {
            "capital": capital,
            "var_base_pct": round(var_base * 100, 4),
            "var_base_usd": round(var_base * capital, 2),
            "volatilidad_base_anual": round(sigma_base * math.sqrt(252) * 100, 4),
            "escenarios": resultados,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  MLPredictor — Singleton para predicción de retornos con modelo pre-entrenado
# ═══════════════════════════════════════════════════════════════════════════════

class MLPredictor:
    """
    Singleton que carga el modelo ML una sola vez y expone predict().
    El modelo predice el retorno diario (%) del activo dado un vector de features.
    """

    _instance: Optional["MLPredictor"] = None
    _model = None
    _model_path: str = ""

    def __new__(cls, model_path: str = "app/ml/model.joblib") -> "MLPredictor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model_path = model_path
            cls._instance._load_model()
        return cls._instance

    def _load_model(self) -> None:
        import os
        try:
            import joblib
            if os.path.exists(self._model_path):
                self._model = joblib.load(self._model_path)
                logger.info("Modelo ML cargado desde %s", self._model_path)
            else:
                logger.warning("Modelo ML no encontrado en %s — usando predictor dummy", self._model_path)
                self._model = None
        except ImportError:
            logger.warning("joblib no instalado — usando predictor dummy")
            self._model = None

    @timed
    def predict(
        self,
        ticker: str,
        returns: pd.Series,
    ) -> Dict[str, Any]:
        """
        Genera features técnicas a partir de la serie de retornos y predice.
        Features: [media_5d, media_20d, vol_5d, vol_20d, momentum_5d, rsi_14d]
        """
        features = self._build_features(returns)

        if self._model is not None:
            import numpy as np
            X = np.array([[
                features["media_5d"],
                features["media_20d"],
                features["vol_5d"],
                features["vol_20d"],
                features["momentum_5d"],
                features["rsi_proxy"],
            ]])
            prediction = float(self._model.predict(X)[0])
        else:
            # Predicción dummy: media móvil 5d + ruido
            prediction = round(features["media_5d"] * 100, 4)

        return {
            "ticker": ticker,
            "prediccion_retorno_pct": round(prediction, 4),
            "features": features,
            "modelo": "RandomForest v1.0" if self._model is not None else "Dummy (media 5d)",
        }

    @staticmethod
    def _build_features(returns: pd.Series) -> Dict[str, float]:
        r = returns.dropna()
        media_5 = float(r.tail(5).mean()) if len(r) >= 5 else float(r.mean())
        media_20 = float(r.tail(20).mean()) if len(r) >= 20 else float(r.mean())
        vol_5 = float(r.tail(5).std()) if len(r) >= 5 else float(r.std())
        vol_20 = float(r.tail(20).std()) if len(r) >= 20 else float(r.std())
        mom_5 = float(r.tail(5).sum()) if len(r) >= 5 else float(r.sum())

        # RSI proxy (simplificado)
        delta = r.diff().dropna()
        gain = delta.clip(lower=0).tail(14).mean() if len(delta) >= 14 else 0.0
        loss = (-delta.clip(upper=0)).tail(14).mean() if len(delta) >= 14 else 1e-8
        rsi = 100 - 100 / (1 + (gain / max(loss, 1e-8)))

        return {
            "media_5d": round(media_5, 8),
            "media_20d": round(media_20, 8),
            "vol_5d": round(vol_5, 8),
            "vol_20d": round(vol_20, 8),
            "momentum_5d": round(mom_5, 8),
            "rsi_proxy": round(float(rsi), 4),
        }
