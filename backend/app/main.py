"""
backend/app/main.py
Aplicación FastAPI principal con todos los endpoints del proyecto RiskLab.
Ejecutar con: uvicorn app.main:app --reload  (desde la carpeta backend/)

Endpoints:
  GET  /health                   — estado de la API
  GET  /activos                  — activos disponibles
  GET  /precios/{ticker}         — precios históricos
  GET  /rendimientos/{ticker}    — rendimientos y estadísticos
  GET  /indicadores/{ticker}     — indicadores técnicos
  POST /volatilidad              — EWMA con λ configurable
  POST /var                      — VaR (3 métodos), CVaR, Kupiec
  GET  /capm                     — Beta, Alpha de Jensen, CAPM
  POST /frontera-eficiente       — frontera de Markowitz + portafolios óptimos
  GET  /alertas                  — señales automáticas de compra/venta
  GET  /macro                    — indicadores macroeconómicos
  POST /garch                    — modelos ARCH/GARCH
  GET  /curva-rendimiento        — curva Nelson-Siegel desde FRED
  POST /bono/duracion            — duración Macaulay, modificada y convexidad
  POST /opcion/precio            — Black-Scholes + 5 Greeks + vol. implícita
  POST /stress                   — pruebas de estrés (3 escenarios)
  POST /predict                  — predicción ML con registro en BD
  GET  /portafolios              — listar portafolios guardados
  POST /portafolios              — guardar portafolio
  DELETE /portafolios/{id}       — eliminar portafolio
"""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Callable, List, Optional

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from scipy import stats
from sqlalchemy.orm import Session

from .config import Settings, get_settings
from .database import get_db, init_db
from .dependencies import (
    get_bond_service,
    get_capm_calculator,
    get_data_service,
    get_ewma_service,
    get_garch_service,
    get_macro_service,
    get_ml_predictor,
    get_option_pricer,
    get_portfolio_analyzer,
    get_risk_calculator,
    get_signal_generator,
    get_stress_tester,
    get_technical_indicators,
    get_yield_curve_service,
)
from .db_models import PredictionLog, Portfolio
from .models import (
    ActivoInfo,
    ActivosResponse,
    AlertaActivo,
    AlertasResponse,
    BonoResponse,
    BondRequest,
    CAPMResponse,
    CurvaPunto,
    CurvaResponse,
    EscenarioStress,
    EstadisticosRendimientos,
    EWMARequest,
    EWMAResponse,
    EWMAPunto,
    FlujosBono,
    FronteraEficienteResponse,
    FronteraRequest,
    GARCHRequest,
    GARCHResponse,
    IndicadoresResponse,
    MacroIndicador,
    MacroResponse,
    OptionRequest,
    OptionResponse,
    PrecioPunto,
    PreciosResponse,
    PredictRequest,
    PredictResponse,
    PortfolioCreateRequest,
    PortfolioResponse,
    RendimientoPunto,
    RendimientosResponse,
    SeñalItem,
    StressRequest,
    StressResponse,
    VaRRequest,
    VaRResponse,
)
from .services import (
    BondService,
    CAPMCalculator,
    DataService,
    EWMAService,
    GARCHService,
    MacroService,
    MLPredictor,
    NOMBRE_MAP,
    OptionPricer,
    PortfolioAnalyzer,
    RiskCalculator,
    SECTOR_MAP,
    COLOR_MAP,
    SignalGenerator,
    StressTester,
    TechnicalIndicators,
    YieldCurveService,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Executor para correr funciones síncronas (yfinance) en hilo ───────────────
_executor = ThreadPoolExecutor(max_workers=8)


async def run_sync(func: Callable, *args, **kwargs):
    """Ejecuta una función bloqueante en el thread pool para no bloquear el event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, lambda: func(*args, **kwargs))


# ════════════════════════════════════════════════════════════════════════════════
#  Aplicación FastAPI
# ════════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa la base de datos al arrancar y limpia recursos al cerrar."""
    init_db()
    logger.info("Base de datos SQLite inicializada")
    yield
    logger.info("Aplicación cerrada")


app = FastAPI(
    title="RiskLab · USTA — API de Análisis de Riesgo Financiero",
    description=(
        "Backend para el proyecto integrador de **Teoría del Riesgo** (USTA). "
        "Provee endpoints para indicadores técnicos, rendimientos, modelos GARCH, "
        "VaR/CVaR, CAPM, Markowitz, curva Nelson-Siegel, duración de bonos, "
        "Black-Scholes, stress testing, ML y contexto macroeconómico."
    ),
    version="2.0.0",
    contact={"name": "Teoría del Riesgo · USTA"},
    license_info={"name": "MIT"},
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

_settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    # Lee los orígenes permitidos desde Settings/env (ALLOWED_ORIGINS=["https://..."])
    # Por defecto es ["*"] para facilitar el desarrollo local.
    allow_origins=_settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Manejador global de errores no capturados ─────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Captura errores inesperados y retorna 500 con mensaje controlado."""
    logger.exception("Error no capturado en %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Error interno del servidor. Inténtalo de nuevo en unos momentos.",
            "path": str(request.url),
        },
    )


# ════════════════════════════════════════════════════════════════════════════════
#  GET /health
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/", include_in_schema=False)
async def root():
    """Redirige al navegador a la documentación interactiva."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Sistema"])
async def health_check(db: Session = Depends(get_db)):
    """Verifica que la API y la base de datos están activos."""
    # Valida que SQLite responde — detecta disco lleno o archivo corrupto
    try:
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as exc:
        logger.error("Health check: BD no disponible: %s", exc)
        db_status = "error"

    status = "ok" if db_status == "ok" else "degraded"
    return {
        "status": status,
        "database": db_status,
        "timestamp": datetime.now().isoformat(),
        "version": app.version,          # toma la versión definida en FastAPI()
    }


# ════════════════════════════════════════════════════════════════════════════════
#  GET /activos
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/activos", response_model=ActivosResponse, tags=["Portafolio"])
async def get_activos(settings: Settings = Depends(get_settings)):
    """Lista los activos disponibles en el portafolio con sector y color asignado."""
    activos = [
        ActivoInfo(
            ticker=t,
            nombre=NOMBRE_MAP.get(t, t),
            sector=SECTOR_MAP.get(t, "Desconocido"),
            color=COLOR_MAP.get(t, "#6366F1"),
        )
        for t in settings.default_tickers
    ]
    return ActivosResponse(activos=activos, benchmark=settings.benchmark, total=len(activos))


# ════════════════════════════════════════════════════════════════════════════════
#  GET /precios/{ticker}
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/precios/{ticker}", response_model=PreciosResponse, tags=["Datos de Mercado"])
async def get_precios(
    ticker: str,
    years: int = Query(default=3, ge=1, le=10, description="Años de historia"),
    data_svc: DataService = Depends(get_data_service),
):
    """Retorna precios diarios de cierre ajustado de un activo."""
    ticker = ticker.upper()
    try:
        ohlcv = await run_sync(data_svc.get_ohlcv, ticker, years)
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error inesperado: {exc}")

    if ohlcv.empty:
        raise HTTPException(status_code=404, detail=f"No se encontraron datos para '{ticker}'")

    close = ohlcv["Close"].squeeze()
    datos = [
        PrecioPunto(fecha=str(idx.date()), precio=round(float(v), 4))
        for idx, v in close.items()
    ]
    return PreciosResponse(ticker=ticker, datos=datos, n_observaciones=len(datos))


# ════════════════════════════════════════════════════════════════════════════════
#  GET /rendimientos/{ticker}
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/rendimientos/{ticker}", response_model=RendimientosResponse, tags=["Análisis de Retornos"])
async def get_rendimientos(
    ticker: str,
    years: int = Query(default=3, ge=1, le=10),
    data_svc: DataService = Depends(get_data_service),
):
    """Retorna rendimientos diarios (simples y logarítmicos) y estadísticos completos."""
    ticker = ticker.upper()
    try:
        ohlcv = await run_sync(data_svc.get_ohlcv, ticker, years)
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    if ohlcv.empty:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' no encontrado")

    close = ohlcv["Close"].squeeze()
    simple = close.pct_change().dropna()
    log_ret = np.log(close / close.shift(1)).dropna()
    common = simple.index.intersection(log_ret.index)

    datos = [
        RendimientoPunto(
            fecha=str(idx.date()),
            rendimiento_simple=round(float(simple.loc[idx]), 8),
            rendimiento_log=round(float(log_ret.loc[idx]), 8),
        )
        for idx in common
    ]

    jb_stat, jb_p = stats.jarque_bera(log_ret.values)
    sw_stat, sw_p = stats.shapiro(log_ret.values[:5000])

    est = EstadisticosRendimientos(
        media_diaria=round(float(log_ret.mean()), 8),
        media_anual=round(float(log_ret.mean() * 252), 6),
        volatilidad_diaria=round(float(log_ret.std()), 8),
        volatilidad_anual=round(float(log_ret.std() * np.sqrt(252)), 6),
        asimetria=round(float(stats.skew(log_ret.values)), 6),
        curtosis_exceso=round(float(stats.kurtosis(log_ret.values)), 6),
        minimo=round(float(log_ret.min()), 8),
        maximo=round(float(log_ret.max()), 8),
        jarque_bera_stat=round(float(jb_stat), 4),
        jarque_bera_pvalue=round(float(jb_p), 6),
        shapiro_stat=round(float(sw_stat), 6),
        shapiro_pvalue=round(float(sw_p), 6),
        es_normal_jb=bool(jb_p > 0.05),
        es_normal_sw=bool(sw_p > 0.05),
    )

    return RendimientosResponse(ticker=ticker, estadisticos=est, datos=datos, n_observaciones=len(datos))


# ════════════════════════════════════════════════════════════════════════════════
#  GET /indicadores/{ticker}
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/indicadores/{ticker}", response_model=IndicadoresResponse, tags=["Análisis Técnico"])
async def get_indicadores(
    ticker: str,
    years: int = Query(default=2, ge=1, le=5),
    data_svc: DataService = Depends(get_data_service),
    ti: TechnicalIndicators = Depends(get_technical_indicators),
    settings: Settings = Depends(get_settings),
):
    """Retorna SMA, EMA, RSI, MACD, Bandas de Bollinger y Oscilador Estocástico."""
    ticker = ticker.upper()
    try:
        ohlcv = await run_sync(data_svc.get_ohlcv, ticker, years)
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    if ohlcv.empty:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' no encontrado")

    try:
        result = await run_sync(ti.compute_all, ohlcv, settings)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error calculando indicadores: {exc}")

    return IndicadoresResponse(ticker=ticker, **result)


# ════════════════════════════════════════════════════════════════════════════════
#  POST /var
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/var", response_model=VaRResponse, tags=["Riesgo"])
async def post_var(
    request: VaRRequest,
    data_svc: DataService = Depends(get_data_service),
    risk_calc: RiskCalculator = Depends(get_risk_calculator),
    settings: Settings = Depends(get_settings),
):
    """
    Calcula VaR y CVaR con tres métodos:
    - **Paramétrico**: asume distribución normal.
    - **Histórico**: distribución empírica de rendimientos.
    - **Monte Carlo**: 10 000+ escenarios simulados.

    Incluye backtesting con el **test de Kupiec**.
    """
    try:
        prices = await run_sync(data_svc.get_prices, request.tickers, request.years)
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    available = [t for t in request.tickers if t in prices.columns]
    missing = [t for t in request.tickers if t not in prices.columns]
    if missing:
        raise HTTPException(status_code=404, detail=f"Tickers no encontrados en yfinance: {missing}")

    returns = data_svc.get_returns(prices[available])
    if len(returns) < 30:
        raise HTTPException(status_code=422, detail="Historial insuficiente (mínimo 30 observaciones)")

    result = await run_sync(
        risk_calc.compute_var,
        returns, request.tickers, request.weights,
        request.confidence, request.capital, settings.mc_simulations,
    )
    return VaRResponse(**result)


# ════════════════════════════════════════════════════════════════════════════════
#  GET /capm
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/capm", response_model=CAPMResponse, tags=["CAPM"])
async def get_capm(
    tickers: Optional[str] = Query(
        default=None,
        description="Tickers separados por coma (ej: AAPL,JPM,XOM). "
                    "Si se omite, usa el portafolio por defecto.",
    ),
    years: int = Query(default=3, ge=1, le=10),
    data_svc: DataService = Depends(get_data_service),
    capm_calc: CAPMCalculator = Depends(get_capm_calculator),
    settings: Settings = Depends(get_settings),
):
    """
    Calcula **Beta**, **Alpha de Jensen** y rendimiento esperado CAPM para cada activo.
    La tasa libre de riesgo se obtiene automáticamente desde Yahoo Finance (^IRX).
    """
    ticker_list: List[str] = (
        [t.upper().strip() for t in tickers.split(",")]
        if tickers
        else settings.default_tickers
    )

    try:
        prices = await run_sync(data_svc.get_prices, ticker_list + [settings.benchmark], years)
        rf = await run_sync(data_svc.get_risk_free_rate)
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    returns = data_svc.get_returns(prices)
    result = await run_sync(capm_calc.compute, returns, settings.benchmark, ticker_list, rf)
    return CAPMResponse(**result)


# ════════════════════════════════════════════════════════════════════════════════
#  POST /frontera-eficiente
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/frontera-eficiente", response_model=FronteraEficienteResponse, tags=["Optimización"])
async def post_frontera_eficiente(
    request: FronteraRequest,
    data_svc: DataService = Depends(get_data_service),
    pa: PortfolioAnalyzer = Depends(get_portfolio_analyzer),
    settings: Settings = Depends(get_settings),
):
    """
    Simula *n_portfolios* combinaciones aleatorias de pesos y calcula la **frontera eficiente**.
    Retorna también los portafolios de **mínima varianza** y **máximo ratio de Sharpe**.
    """
    try:
        prices = await run_sync(data_svc.get_prices, request.tickers, request.years)
        rf = await run_sync(data_svc.get_risk_free_rate)
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    available = [t for t in request.tickers if t in prices.columns]
    if len(available) < 2:
        raise HTTPException(
            status_code=422,
            detail=f"Se requieren al menos 2 activos. Disponibles: {available}",
        )

    returns = data_svc.get_returns(prices[available])
    result = await run_sync(pa.compute_frontier, returns, available, request.n_portfolios, rf)
    return FronteraEficienteResponse(**result)


# ════════════════════════════════════════════════════════════════════════════════
#  GET /alertas
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/alertas", response_model=AlertasResponse, tags=["Señales"])
async def get_alertas(
    tickers: Optional[str] = Query(
        default=None,
        description="Tickers separados por coma. Si se omite, usa el portafolio por defecto.",
    ),
    data_svc: DataService = Depends(get_data_service),
    sig_gen: SignalGenerator = Depends(get_signal_generator),
    settings: Settings = Depends(get_settings),
):
    """
    Evalúa 5 indicadores técnicos por activo y retorna señales automáticas:
    MACD · RSI · Bollinger · Cruce de Medias · Estocástico.
    Incluye un **score compuesto** y clasificación semáforo.
    """
    ticker_list: List[str] = (
        [t.upper().strip() for t in tickers.split(",")]
        if tickers
        else settings.default_tickers
    )

    activos: List[AlertaActivo] = []
    timestamp = datetime.now().isoformat()

    for ticker in ticker_list:
        try:
            ohlcv = await run_sync(data_svc.get_ohlcv, ticker, 2)
            result = await run_sync(sig_gen.compute_signals, ohlcv, ticker, settings)
            activos.append(
                AlertaActivo(
                    ticker=result["ticker"],
                    señales=[SeñalItem(**s) for s in result["señales"]],
                    score_compuesto=result["score_compuesto"],
                    clasificacion=result["clasificacion"],
                    color=result["color"],
                )
            )
        except Exception as exc:
            logger.warning("Señales no disponibles para %s: %s", ticker, exc)

    return AlertasResponse(activos=activos, timestamp=timestamp)


# ════════════════════════════════════════════════════════════════════════════════
#  GET /macro
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/macro", response_model=MacroResponse, tags=["Macro"])
async def get_macro(
    macro_svc: MacroService = Depends(get_macro_service),
    settings: Settings = Depends(get_settings),
):
    """
    Retorna indicadores macroeconómicos actualizados vía API:
    Tasa libre de riesgo (^IRX) · T10Y (^TNX) · VIX · USD/COP · EUR/USD · Oro.
    """
    try:
        result = await run_sync(macro_svc.get_macro_indicators, settings.rf_fallback)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Error obteniendo datos macro: {exc}")

    return MacroResponse(
        tasa_libre_riesgo=result["tasa_libre_riesgo"],
        indicadores=[MacroIndicador(**i) for i in result["indicadores"]],
        timestamp=result["timestamp"],
    )


# ════════════════════════════════════════════════════════════════════════════════
#  POST /garch  (endpoint adicional — Módulo 3)
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/garch", response_model=GARCHResponse, tags=["Volatilidad"])
async def post_garch(
    request: GARCHRequest,
    data_svc: DataService = Depends(get_data_service),
    garch_svc: GARCHService = Depends(get_garch_service),
):
    """
    Ajusta y compara cuatro especificaciones de volatilidad condicional:
    **ARCH(1)** · **GARCH(1,1)** · **GJR-GARCH(1,1)** · **EGARCH(1,1)**.
    Retorna AIC, BIC, Log-Likelihood, parámetros y pronóstico de volatilidad a 1 día.
    """
    try:
        ohlcv = await run_sync(data_svc.get_ohlcv, request.ticker, request.years)
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    if ohlcv.empty:
        raise HTTPException(status_code=404, detail=f"Ticker '{request.ticker}' no encontrado")

    log_ret = np.log(ohlcv["Close"].squeeze() / ohlcv["Close"].squeeze().shift(1)).dropna()
    if len(log_ret) < 60:
        raise HTTPException(status_code=422, detail="Historial insuficiente para ajustar GARCH (mínimo 60 obs)")

    result = await run_sync(garch_svc.fit_models, log_ret)
    return GARCHResponse(ticker=request.ticker, **result)


# ════════════════════════════════════════════════════════════════════════════════
#  POST /volatilidad  — EWMA con λ configurable
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/volatilidad", response_model=EWMAResponse, tags=["Volatilidad"])
async def post_volatilidad(
    request: EWMARequest,
    data_svc: DataService = Depends(get_data_service),
    ewma_svc: EWMAService = Depends(get_ewma_service),
):
    """
    Calcula la **volatilidad EWMA** con factor de decaimiento λ configurable.
    La fórmula recursiva es: σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}.
    λ=0.94 es el valor estándar de **RiskMetrics** para datos diarios.
    """
    try:
        ohlcv = await run_sync(data_svc.get_ohlcv, request.ticker, request.years)
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    if ohlcv.empty:
        raise HTTPException(status_code=404, detail=f"Ticker '{request.ticker}' no encontrado")

    log_ret = np.log(ohlcv["Close"].squeeze() / ohlcv["Close"].squeeze().shift(1)).dropna()
    if len(log_ret) < 20:
        raise HTTPException(status_code=422, detail="Historial insuficiente (mínimo 20 obs)")

    result = await run_sync(ewma_svc.compute, log_ret, request.lam)
    return EWMAResponse(
        ticker=request.ticker,
        lam=result["lambda"],
        volatilidad_1d=result["volatilidad_1d"],
        volatilidad_anual=result["volatilidad_anual"],
        n_observaciones=result["n_observaciones"],
        serie=[EWMAPunto(**p) for p in result["serie"]],
    )


# ════════════════════════════════════════════════════════════════════════════════
#  GET /curva-rendimiento — Curva Nelson-Siegel desde FRED
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/curva-rendimiento", response_model=CurvaResponse, tags=["Renta Fija"])
async def get_curva_rendimiento(
    yc_svc: YieldCurveService = Depends(get_yield_curve_service),
    settings: Settings = Depends(get_settings),
):
    """
    Obtiene tasas del Tesoro de EE.UU. desde FRED y ajusta el modelo **Nelson-Siegel**:
    y(T) = β₀ + β₁·factor₁(T,τ) + β₂·factor₂(T,τ)

    Retorna parámetros del modelo y la curva interpolada de 3M a 30Y.
    """
    try:
        rates = await run_sync(yc_svc.get_rates, settings.fred_api_key)
        result = await run_sync(yc_svc.fit, rates)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Error obteniendo curva de rendimiento: {exc}")

    return CurvaResponse(
        parametros=result["parametros"],
        tasas_observadas=result["tasas_observadas"],
        curva=[CurvaPunto(**p) for p in result["curva"]],
        rmse=result["rmse"],
    )


# ════════════════════════════════════════════════════════════════════════════════
#  POST /bono/duracion — Duración y Convexidad de un Bono
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/bono/duracion", response_model=BonoResponse, tags=["Renta Fija"])
async def post_bono_duracion(
    request: BondRequest,
    bond_svc: BondService = Depends(get_bond_service),
):
    """
    Calcula el **precio**, **duración de Macaulay**, **duración modificada**,
    **convexidad** y **DV01** de un bono de cupón fijo.

    - Duración de Macaulay: promedio ponderado del tiempo de los flujos.
    - Duración modificada: sensibilidad del precio ante cambios en YTM.
    - Convexidad: curvatura de la relación precio-YTM.
    - DV01: cambio en precio por 1 punto base (0.01 %) en el YTM.
    """
    result = await run_sync(
        bond_svc.compute,
        request.face_value,
        request.coupon_rate,
        request.ytm,
        request.periods,
        request.frequency,
    )
    return BonoResponse(
        precio=result["precio"],
        duracion_macaulay=result["duracion_macaulay"],
        duracion_modificada=result["duracion_modificada"],
        convexidad=result["convexidad"],
        dv01=result["dv01"],
        flujos=[FlujosBono(**f) for f in result["flujos"]],
        parametros=result["parametros"],
    )


# ════════════════════════════════════════════════════════════════════════════════
#  POST /opcion/precio — Black-Scholes + 5 Greeks + Volatilidad Implícita
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/opcion/precio", response_model=OptionResponse, tags=["Derivados"])
async def post_opcion_precio(
    request: OptionRequest,
    pricer: OptionPricer = Depends(get_option_pricer),
):
    """
    Valúa una opción europea (call/put) con el modelo **Black-Scholes**.

    Calcula los **5 Greeks**:
    - **Delta (Δ)**: sensibilidad al precio del subyacente
    - **Gamma (Γ)**: tasa de cambio del delta
    - **Vega (ν)**: sensibilidad a la volatilidad (por 1 %)
    - **Theta (Θ)**: decaimiento temporal (por día calendario)
    - **Rho (ρ)**: sensibilidad a la tasa libre de riesgo (por 1 %)

    Si se proporciona `market_price`, también calcula la **volatilidad implícita**
    usando el método de **Newton-Raphson**.
    """
    result = await run_sync(pricer.price, request.S, request.K, request.r, request.sigma, request.T, request.option_type)

    iv = None
    if request.market_price is not None:
        iv = await run_sync(
            pricer.implied_volatility,
            request.market_price, request.S, request.K, request.r, request.T, request.option_type,
        )

    return OptionResponse(
        precio=result["precio"],
        delta=result["delta"],
        gamma=result["gamma"],
        vega=result["vega"],
        theta=result["theta"],
        rho=result["rho"],
        d1=result["d1"],
        d2=result["d2"],
        tipo=result["tipo"],
        volatilidad_implicita=iv,
        parametros=result["parametros"],
    )


# ════════════════════════════════════════════════════════════════════════════════
#  POST /stress — Pruebas de estrés (3 escenarios)
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/stress", response_model=StressResponse, tags=["Riesgo"])
async def post_stress(
    request: StressRequest,
    data_svc: DataService = Depends(get_data_service),
    stress_svc: StressTester = Depends(get_stress_tester),
):
    """
    Aplica **3 escenarios de estrés** al portafolio:
    1. **Shock de tasas** (+200 pb — estilo Taper Tantrum 2013)
    2. **Crash de mercado** (-20 % — estilo COVID-19 Marzo 2020)
    3. **Explosión de volatilidad** (+150 % — estilo VIX spike 2008)

    Para cada escenario calcula el P&L estimado y el VaR estresado.
    """
    try:
        prices = await run_sync(data_svc.get_prices, request.tickers, request.years)
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    available = [t for t in request.tickers if t in prices.columns]
    if len(available) < 1:
        raise HTTPException(status_code=404, detail=f"Ningún ticker disponible: {request.tickers}")

    returns = data_svc.get_returns(prices[available])
    result = await run_sync(
        stress_svc.compute, returns, available, request.weights, request.capital
    )
    return StressResponse(
        capital=result["capital"],
        var_base_pct=result["var_base_pct"],
        var_base_usd=result["var_base_usd"],
        volatilidad_base_anual=result["volatilidad_base_anual"],
        escenarios=[EscenarioStress(**e) for e in result["escenarios"]],
    )


# ════════════════════════════════════════════════════════════════════════════════
#  POST /predict — Predicción ML con registro en BD
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/predict", response_model=PredictResponse, tags=["Machine Learning"])
async def post_predict(
    request: PredictRequest,
    data_svc: DataService = Depends(get_data_service),
    predictor: MLPredictor = Depends(get_ml_predictor),
    db: Session = Depends(get_db),
):
    """
    Genera una **predicción de retorno** para el siguiente día usando un modelo
    de Machine Learning pre-entrenado (RandomForest o Gradient Boosting).

    Features utilizadas: media 5d, media 20d, volatilidad 5d, volatilidad 20d,
    momentum 5d, RSI proxy 14d.

    Cada predicción queda registrada en la tabla `prediction_logs` de SQLite.
    """
    try:
        ohlcv = await run_sync(data_svc.get_ohlcv, request.ticker, request.years)
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    if ohlcv.empty:
        raise HTTPException(status_code=404, detail=f"Ticker '{request.ticker}' no encontrado")

    log_ret = np.log(ohlcv["Close"].squeeze() / ohlcv["Close"].squeeze().shift(1)).dropna()
    result = await run_sync(predictor.predict, request.ticker, log_ret)

    # Persistir en BD
    log_entry = PredictionLog(
        ticker=request.ticker,
        features_json=json.dumps(result["features"]),
        prediction=result["prediccion_retorno_pct"],
        model_version="v1.0",
    )
    db.add(log_entry)
    db.commit()

    return PredictResponse(
        ticker=result["ticker"],
        prediccion_retorno_pct=result["prediccion_retorno_pct"],
        features=result["features"],
        modelo=result["modelo"],
    )


# ════════════════════════════════════════════════════════════════════════════════
#  CRUD /portafolios — Portafolios guardados en BD
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/portafolios", response_model=List[PortfolioResponse], tags=["Portafolio"])
async def get_portafolios(db: Session = Depends(get_db)):
    """Lista todos los portafolios guardados en la base de datos."""
    portafolios = db.query(Portfolio).order_by(Portfolio.created_at.desc()).all()
    return [
        PortfolioResponse(
            id=p.id,
            nombre=p.nombre,
            descripcion=p.descripcion,
            tickers=json.loads(p.tickers),
            pesos=json.loads(p.pesos),
            capital=p.capital,
            created_at=p.created_at.isoformat(),
            updated_at=p.updated_at.isoformat() if p.updated_at else p.created_at.isoformat(),
        )
        for p in portafolios
    ]


@app.post("/portafolios", response_model=PortfolioResponse, status_code=201, tags=["Portafolio"])
async def create_portafolio(
    request: PortfolioCreateRequest,
    db: Session = Depends(get_db),
):
    """Crea y persiste un nuevo portafolio en la base de datos SQLite."""
    p = Portfolio(
        nombre=request.nombre,
        descripcion=request.descripcion,
        tickers=json.dumps(request.tickers),
        pesos=json.dumps(request.pesos),
        capital=request.capital,
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return PortfolioResponse(
        id=p.id,
        nombre=p.nombre,
        descripcion=p.descripcion,
        tickers=json.loads(p.tickers),
        pesos=json.loads(p.pesos),
        capital=p.capital,
        created_at=p.created_at.isoformat(),
        updated_at=p.updated_at.isoformat() if p.updated_at else p.created_at.isoformat(),
    )


@app.delete("/portafolios/{portafolio_id}", status_code=204, tags=["Portafolio"])
async def delete_portafolio(portafolio_id: int, db: Session = Depends(get_db)):
    """Elimina un portafolio de la base de datos por su ID."""
    p = db.query(Portfolio).filter(Portfolio.id == portafolio_id).first()
    if p is None:
        raise HTTPException(status_code=404, detail=f"Portafolio {portafolio_id} no encontrado")
    db.delete(p)
    db.commit()
