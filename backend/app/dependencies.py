"""
backend/app/dependencies.py
Dependencias inyectables con FastAPI Depends().
Cada función retorna una instancia del servicio correspondiente,
desacoplando la lógica de negocio de las rutas.
"""

from fastapi import Depends

from .config import Settings, get_settings
from .services import (
    BondService,
    CAPMCalculator,
    DataService,
    EWMAService,
    GARCHService,
    MacroService,
    MLPredictor,
    OptionPricer,
    PortfolioAnalyzer,
    RiskCalculator,
    SignalGenerator,
    StressTester,
    TechnicalIndicators,
    YieldCurveService,
)


# ── Servicio de datos ─────────────────────────────────────────────────────────

def get_data_service(settings: Settings = Depends(get_settings)) -> DataService:
    """Inyecta DataService configurado con el benchmark de settings."""
    return DataService(benchmark=settings.benchmark)


# ── Indicadores técnicos ──────────────────────────────────────────────────────

def get_technical_indicators() -> TechnicalIndicators:
    """Inyecta la clase de indicadores técnicos (sin estado)."""
    return TechnicalIndicators()


# ── Calculadora de riesgo ─────────────────────────────────────────────────────

def get_risk_calculator() -> RiskCalculator:
    """Inyecta RiskCalculator (sin estado)."""
    return RiskCalculator()


# ── Calculadora CAPM ──────────────────────────────────────────────────────────

def get_capm_calculator() -> CAPMCalculator:
    """Inyecta CAPMCalculator (sin estado)."""
    return CAPMCalculator()


# ── Analizador de portafolio ──────────────────────────────────────────────────

def get_portfolio_analyzer() -> PortfolioAnalyzer:
    """Inyecta PortfolioAnalyzer (sin estado)."""
    return PortfolioAnalyzer()


# ── Generador de señales ──────────────────────────────────────────────────────

def get_signal_generator(
    ti: TechnicalIndicators = Depends(get_technical_indicators),
) -> SignalGenerator:
    """Inyecta SignalGenerator con su dependencia de TechnicalIndicators."""
    return SignalGenerator(ti)


# ── Servicio macro ────────────────────────────────────────────────────────────

def get_macro_service() -> MacroService:
    """Inyecta MacroService (sin estado)."""
    return MacroService()


# ── Servicio GARCH ────────────────────────────────────────────────────────────

def get_garch_service() -> GARCHService:
    """Inyecta GARCHService (sin estado)."""
    return GARCHService()


# ── Servicio EWMA ─────────────────────────────────────────────────────────────

def get_ewma_service() -> EWMAService:
    """Inyecta EWMAService (sin estado)."""
    return EWMAService()


# ── Servicio Curva de Rendimiento ─────────────────────────────────────────────

def get_yield_curve_service() -> YieldCurveService:
    """Inyecta YieldCurveService (sin estado)."""
    return YieldCurveService()


# ── Servicio de Bonos ─────────────────────────────────────────────────────────

def get_bond_service() -> BondService:
    """Inyecta BondService (sin estado)."""
    return BondService()


# ── Pricer de Opciones ────────────────────────────────────────────────────────

def get_option_pricer() -> OptionPricer:
    """Inyecta OptionPricer (sin estado)."""
    return OptionPricer()


# ── Stress Tester ─────────────────────────────────────────────────────────────

def get_stress_tester() -> StressTester:
    """Inyecta StressTester (sin estado)."""
    return StressTester()


# ── Predictor ML (Singleton) ──────────────────────────────────────────────────

def get_ml_predictor() -> MLPredictor:
    """Inyecta MLPredictor — patrón Singleton: se carga el modelo una sola vez."""
    return MLPredictor(model_path="app/ml/model.joblib")
