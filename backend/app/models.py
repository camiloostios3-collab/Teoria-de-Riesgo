"""
backend/app/models.py
Modelos Pydantic v2 para request y response de todos los endpoints.
Se usan Field() con descripciones para enriquecer la documentación /docs.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ═══════════════════════════════════════════════════════════════════════════════
#  REQUEST MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class VaRRequest(BaseModel):
    """Parámetros para calcular VaR y CVaR de un portafolio."""

    tickers: List[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Lista de tickers del portafolio (ej: ['AAPL', 'JPM'])",
        examples=[["AAPL", "JPM", "XOM"]],
    )
    weights: List[float] = Field(
        ...,
        min_length=1,
        description="Pesos del portafolio. Deben ser no negativos y sumar exactamente 1.0",
        examples=[[0.4, 0.35, 0.25]],
    )
    confidence: float = Field(
        default=0.95,
        ge=0.90,
        le=0.9999,
        description="Nivel de confianza del VaR (0.95 → 95 %)",
    )
    capital: float = Field(
        default=100_000.0,
        gt=0,
        description="Capital invertido en USD para calcular VaR en dólares",
    )
    years: int = Field(default=3, ge=1, le=10, description="Años de historia para el cálculo")

    @field_validator("tickers")
    @classmethod
    def normalizar_tickers(cls, v: List[str]) -> List[str]:
        cleaned: List[str] = []
        for t in v:
            t = t.upper().strip()
            if not (1 <= len(t) <= 10):
                raise ValueError(f"Ticker '{t}' inválido: debe tener entre 1 y 10 caracteres")
            cleaned.append(t)
        return cleaned

    @model_validator(mode="after")
    def validar_pesos(self) -> "VaRRequest":
        if len(self.tickers) != len(self.weights):
            raise ValueError(
                f"tickers ({len(self.tickers)}) y weights ({len(self.weights)}) "
                "deben tener la misma longitud"
            )
        if any(w < 0 for w in self.weights):
            raise ValueError("Todos los pesos deben ser >= 0")
        total = sum(self.weights)
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Los pesos deben sumar 1.0; suma actual: {total:.6f}")
        return self


class FronteraRequest(BaseModel):
    """Parámetros para construir la frontera eficiente de Markowitz."""

    tickers: List[str] = Field(
        ...,
        min_length=2,
        max_length=15,
        description="Al menos 2 tickers para construir la frontera",
        examples=[["AAPL", "JPM", "XOM", "JNJ", "AMZN"]],
    )
    n_portfolios: int = Field(
        default=10_000,
        ge=1_000,
        le=50_000,
        description="Número de portafolios aleatorios a simular",
    )
    years: int = Field(default=3, ge=1, le=10, description="Años de historia")

    @field_validator("tickers")
    @classmethod
    def normalizar_tickers(cls, v: List[str]) -> List[str]:
        return [t.upper().strip() for t in v]


class GARCHRequest(BaseModel):
    """Parámetros para ajustar modelos ARCH/GARCH."""

    ticker: str = Field(..., description="Ticker del activo a modelar", examples=["AAPL"])
    years: int = Field(default=3, ge=1, le=10)

    @field_validator("ticker")
    @classmethod
    def normalizar_ticker(cls, v: str) -> str:
        return v.upper().strip()


# ═══════════════════════════════════════════════════════════════════════════════
#  RESPONSE MODELS — Activos
# ═══════════════════════════════════════════════════════════════════════════════

class ActivoInfo(BaseModel):
    ticker: str
    nombre: str
    sector: str
    color: str


class ActivosResponse(BaseModel):
    activos: List[ActivoInfo]
    benchmark: str
    total: int


# ═══════════════════════════════════════════════════════════════════════════════
#  RESPONSE MODELS — Precios
# ═══════════════════════════════════════════════════════════════════════════════

class PrecioPunto(BaseModel):
    fecha: str
    precio: float


class PreciosResponse(BaseModel):
    ticker: str
    moneda: str = "USD"
    frecuencia: str = "diaria"
    datos: List[PrecioPunto]
    n_observaciones: int


# ═══════════════════════════════════════════════════════════════════════════════
#  RESPONSE MODELS — Rendimientos
# ═══════════════════════════════════════════════════════════════════════════════

class EstadisticosRendimientos(BaseModel):
    media_diaria: float
    media_anual: float
    volatilidad_diaria: float
    volatilidad_anual: float
    asimetria: float
    curtosis_exceso: float
    minimo: float
    maximo: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    shapiro_stat: float
    shapiro_pvalue: float
    es_normal_jb: bool = Field(description="True si no se rechaza normalidad (p > 0.05, Jarque-Bera)")
    es_normal_sw: bool = Field(description="True si no se rechaza normalidad (p > 0.05, Shapiro-Wilk)")


class RendimientoPunto(BaseModel):
    fecha: str
    rendimiento_simple: float
    rendimiento_log: float


class RendimientosResponse(BaseModel):
    ticker: str
    tipo: str = "logaritmico"
    estadisticos: EstadisticosRendimientos
    datos: List[RendimientoPunto]
    n_observaciones: int


# ═══════════════════════════════════════════════════════════════════════════════
#  RESPONSE MODELS — Indicadores técnicos
# ═══════════════════════════════════════════════════════════════════════════════

class IndicadoresResponse(BaseModel):
    ticker: str
    sma_corta: List[Dict[str, Any]]
    sma_larga: List[Dict[str, Any]]
    ema: List[Dict[str, Any]]
    rsi: List[Dict[str, Any]]
    macd: List[Dict[str, Any]]
    bollinger: List[Dict[str, Any]]
    estocastico: List[Dict[str, Any]]


# ═══════════════════════════════════════════════════════════════════════════════
#  RESPONSE MODELS — VaR y CVaR
# ═══════════════════════════════════════════════════════════════════════════════

class VaRResultado(BaseModel):
    metodo: str
    var_diario_pct: float = Field(description="VaR diario como fracción (ej: 0.02 = 2 %)")
    var_diario_usd: float = Field(description="VaR diario en USD")
    var_anual_pct: float = Field(description="VaR anualizado (√252 × VaR diario)")
    cvar_diario_pct: float = Field(description="Expected Shortfall diario como fracción")
    cvar_diario_usd: float = Field(description="Expected Shortfall diario en USD")
    confianza: float


class KupiecResult(BaseModel):
    n_violaciones: int
    tasa_violacion: float
    tasa_esperada: float
    lr_statistic: float
    p_value: float
    resultado: str = Field(description="Interpretación del test de Kupiec")


class VaRResponse(BaseModel):
    tickers: List[str]
    pesos: List[float]
    capital: float
    confianza: float
    resultados: List[VaRResultado]
    kupiec: Optional[KupiecResult] = None


# ═══════════════════════════════════════════════════════════════════════════════
#  RESPONSE MODELS — CAPM
# ═══════════════════════════════════════════════════════════════════════════════

class ActivoCAPM(BaseModel):
    ticker: str
    beta: float
    beta_ic_inferior: float
    beta_ic_superior: float
    alpha_jensen: float = Field(description="Alpha de Jensen anualizado")
    r_cuadrado: float
    p_value: float
    retorno_esperado_capm: float = Field(description="E[R] = Rf + β × (Rm - Rf)")
    clasificacion: str = Field(description="Agresivo (β>1.2), Neutro (0.8≤β≤1.2), Defensivo (β<0.8)")


class CAPMResponse(BaseModel):
    tasa_libre_riesgo: float
    prima_mercado: float
    retorno_mercado: float
    activos: List[ActivoCAPM]


# ═══════════════════════════════════════════════════════════════════════════════
#  RESPONSE MODELS — Frontera eficiente
# ═══════════════════════════════════════════════════════════════════════════════

class PortafolioPunto(BaseModel):
    retorno: float
    volatilidad: float
    sharpe: float
    pesos: List[float]


class PortafolioOptimo(BaseModel):
    nombre: str
    retorno: float
    volatilidad: float
    sharpe: float
    composicion: Dict[str, float] = Field(description="Ticker → peso en el portafolio")


class FronteraEficienteResponse(BaseModel):
    tickers: List[str]
    n_simulados: int
    portafolios: List[PortafolioPunto]
    optimos: List[PortafolioOptimo]


# ═══════════════════════════════════════════════════════════════════════════════
#  RESPONSE MODELS — Señales y alertas
# ═══════════════════════════════════════════════════════════════════════════════

class SeñalItem(BaseModel):
    indicador: str
    señal: str = Field(description="COMPRA, VENTA o NEUTRAL")
    valor_actual: float
    descripcion: str


class AlertaActivo(BaseModel):
    ticker: str
    señales: List[SeñalItem]
    score_compuesto: float = Field(description="Score normalizado en [-1, 1]")
    clasificacion: str = Field(
        description="COMPRA FUERTE, SESGO ALCISTA, NEUTRAL, SESGO BAJISTA, VENTA FUERTE"
    )
    color: str = Field(description="Color hex asociado a la clasificación")


class AlertasResponse(BaseModel):
    activos: List[AlertaActivo]
    timestamp: str


# ═══════════════════════════════════════════════════════════════════════════════
#  RESPONSE MODELS — Macro
# ═══════════════════════════════════════════════════════════════════════════════

class MacroIndicador(BaseModel):
    nombre: str
    valor: float
    unidad: str
    fuente: str
    descripcion: str


class MacroResponse(BaseModel):
    tasa_libre_riesgo: float = Field(description="Rf anual como fracción (ej: 0.0525 = 5.25 %)")
    indicadores: List[MacroIndicador]
    timestamp: str


# ═══════════════════════════════════════════════════════════════════════════════
#  RESPONSE MODELS — GARCH (endpoint adicional)
# ═══════════════════════════════════════════════════════════════════════════════

class GARCHEspecificacion(BaseModel):
    nombre: str
    aic: float
    bic: float
    log_likelihood: float
    omega: float
    alpha: float
    beta: Optional[float] = None
    gamma: Optional[float] = None
    persistencia: Optional[float] = None
    pronostico_vol_1d: float = Field(description="Pronóstico de volatilidad a 1 día (diaria)")
    pronostico_vol_anual: float = Field(description="Pronóstico de volatilidad anualizado")


class GARCHResponse(BaseModel):
    ticker: str
    especificaciones: List[GARCHEspecificacion]
    mejor_modelo: str = Field(description="Nombre del modelo con menor AIC")
    jarque_bera_residuos: float
    jarque_bera_pvalue: float


# ═══════════════════════════════════════════════════════════════════════════════
#  REQUEST MODELS — nuevos módulos
# ═══════════════════════════════════════════════════════════════════════════════

class EWMARequest(BaseModel):
    """Parámetros para calcular volatilidad EWMA."""
    ticker: str = Field(..., description="Ticker del activo", examples=["AAPL"])
    years: int = Field(default=3, ge=1, le=10)
    lam: float = Field(
        default=0.94, ge=0.01, le=0.9999,
        description="Factor de decaimiento λ (0.94 recomendado por RiskMetrics)",
    )

    @field_validator("ticker")
    @classmethod
    def upper_ticker(cls, v: str) -> str:
        return v.upper().strip()


class BondRequest(BaseModel):
    """Parámetros para calcular duración y convexidad de un bono."""
    face_value: float = Field(default=1000.0, gt=0, description="Valor nominal (USD)")
    coupon_rate: float = Field(..., ge=0, le=1, description="Tasa de cupón anual (fracción)")
    ytm: float = Field(..., ge=0, le=1, description="Yield to maturity anual (fracción)")
    periods: int = Field(..., ge=1, le=120, description="Número total de períodos de pago")
    frequency: int = Field(default=2, description="Pagos por año (1=anual, 2=semestral, 4=trimestral)")


class OptionRequest(BaseModel):
    """Parámetros para valuar una opción europea con Black-Scholes."""
    S: float = Field(..., gt=0, description="Precio actual del subyacente (USD)")
    K: float = Field(..., gt=0, description="Precio de ejercicio (strike) en USD")
    r: float = Field(default=0.05, ge=0, le=0.5, description="Tasa libre de riesgo anual")
    sigma: float = Field(..., gt=0, le=5.0, description="Volatilidad anual del subyacente")
    T: float = Field(..., gt=0, description="Tiempo al vencimiento en años")
    option_type: str = Field(default="call", description="Tipo de opción: 'call' o 'put'")
    market_price: Optional[float] = Field(
        default=None, gt=0,
        description="Precio de mercado para calcular volatilidad implícita (opcional)",
    )

    @field_validator("option_type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in ("call", "put"):
            raise ValueError("option_type debe ser 'call' o 'put'")
        return v


class StressRequest(BaseModel):
    """Parámetros para prueba de estrés del portafolio."""
    tickers: List[str] = Field(..., min_length=1, max_length=10)
    weights: List[float] = Field(..., min_length=1)
    capital: float = Field(default=100_000.0, gt=0)
    years: int = Field(default=3, ge=1, le=10)

    @field_validator("tickers")
    @classmethod
    def upper_tickers(cls, v: List[str]) -> List[str]:
        return [t.upper().strip() for t in v]

    @model_validator(mode="after")
    def check_weights(self) -> "StressRequest":
        if len(self.tickers) != len(self.weights):
            raise ValueError("tickers y weights deben tener la misma longitud")
        if any(w < 0 for w in self.weights):
            raise ValueError("Todos los pesos deben ser >= 0")
        if abs(sum(self.weights) - 1.0) > 1e-4:
            raise ValueError(f"Los pesos deben sumar 1.0; suma: {sum(self.weights):.6f}")
        return self


class PredictRequest(BaseModel):
    """Parámetros para predicción ML."""
    ticker: str = Field(..., description="Ticker del activo a predecir", examples=["AAPL"])
    years: int = Field(default=3, ge=1, le=10)

    @field_validator("ticker")
    @classmethod
    def upper_ticker(cls, v: str) -> str:
        return v.upper().strip()


class PortfolioCreateRequest(BaseModel):
    """Datos para crear o actualizar un portafolio guardado."""
    nombre: str = Field(..., min_length=1, max_length=100)
    descripcion: Optional[str] = Field(default=None, max_length=500)
    tickers: List[str] = Field(..., min_length=1, max_length=10)
    pesos: List[float] = Field(..., min_length=1)
    capital: float = Field(default=100_000.0, gt=0)

    @field_validator("tickers")
    @classmethod
    def upper_tickers(cls, v: List[str]) -> List[str]:
        return [t.upper().strip() for t in v]

    @model_validator(mode="after")
    def check_weights(self) -> "PortfolioCreateRequest":
        if len(self.tickers) != len(self.pesos):
            raise ValueError("tickers y pesos deben tener la misma longitud")
        if abs(sum(self.pesos) - 1.0) > 1e-4:
            raise ValueError(f"Los pesos deben sumar 1.0; suma: {sum(self.pesos):.6f}")
        return self


# ═══════════════════════════════════════════════════════════════════════════════
#  RESPONSE MODELS — nuevos módulos
# ═══════════════════════════════════════════════════════════════════════════════

class EWMAPunto(BaseModel):
    fecha: str
    volatilidad_diaria: float


class EWMAResponse(BaseModel):
    ticker: str
    lam: float = Field(description="Factor de decaimiento λ utilizado")
    volatilidad_1d: float = Field(description="Volatilidad EWMA a 1 día (pronóstico)")
    volatilidad_anual: float = Field(description="Volatilidad EWMA anualizada")
    n_observaciones: int
    serie: List[EWMAPunto]


class CurvaPunto(BaseModel):
    plazo: float = Field(description="Plazo en años")
    tasa_ns: float = Field(description="Tasa Nelson-Siegel ajustada")
    tasa_observada: Optional[float] = Field(default=None, description="Tasa observada de mercado")


class CurvaResponse(BaseModel):
    parametros: Dict[str, float] = Field(description="Parámetros β₀, β₁, β₂, τ del modelo Nelson-Siegel")
    tasas_observadas: Dict[str, float]
    curva: List[CurvaPunto]
    rmse: float = Field(description="Error cuadrático medio del ajuste Nelson-Siegel")


class FlujosBono(BaseModel):
    periodo: int
    año: float
    flujo: float
    pv_flujo: float


class BonoResponse(BaseModel):
    precio: float
    duracion_macaulay: float
    duracion_modificada: float
    convexidad: float
    dv01: float = Field(description="Dólar value of 1 bp — sensibilidad ante 1 punto base")
    flujos: List[FlujosBono]
    parametros: Dict[str, Any]


class OptionResponse(BaseModel):
    precio: float
    delta: float = Field(description="Δ — sensibilidad al precio del subyacente")
    gamma: float = Field(description="Γ — tasa de cambio del delta")
    vega: float = Field(description="ν — sensibilidad a la volatilidad (por 1 %)")
    theta: float = Field(description="Θ — decaimiento temporal (por día)")
    rho: float = Field(description="ρ — sensibilidad a la tasa de interés (por 1 %)")
    d1: float
    d2: float
    tipo: str
    volatilidad_implicita: Optional[float] = Field(
        default=None, description="Volatilidad implícita si se proporcionó market_price"
    )
    parametros: Dict[str, Any]


class EscenarioStress(BaseModel):
    escenario: str
    descripcion: str
    shock_equity_pct: float
    multiplicador_volatilidad: float
    pnl_pct: float
    pnl_usd: float
    var_estresado_pct: float
    var_estresado_usd: float
    incremento_var_pct: float


class StressResponse(BaseModel):
    capital: float
    var_base_pct: float
    var_base_usd: float
    volatilidad_base_anual: float
    escenarios: List[EscenarioStress]


class PredictResponse(BaseModel):
    ticker: str
    prediccion_retorno_pct: float = Field(description="Retorno predicho para el próximo día (%)")
    features: Dict[str, float] = Field(description="Features utilizadas en la predicción")
    modelo: str


class PortfolioResponse(BaseModel):
    id: int
    nombre: str
    descripcion: Optional[str]
    tickers: List[str]
    pesos: List[float]
    capital: float
    created_at: str
    updated_at: str
