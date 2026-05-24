"""
backend/tests/test_api.py
Suite de pruebas pytest para la API RiskLab.
Usa TestClient (sin red real) + base de datos SQLite en memoria para aislamiento.

Ejecutar:
    cd backend
    pytest tests/ -v
"""

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.database import Base, get_db

# ── Base de datos en memoria para pruebas ─────────────────────────────────────
# StaticPool garantiza que todas las sesiones comparten la misma conexión
# (necesario para SQLite in-memory que no persiste entre conexiones distintas)

engine_test = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine_test)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True, scope="module")
def setup_db():
    """Crea las tablas antes de los tests (StaticPool garantiza persistencia in-memory)."""
    from app import db_models  # noqa: F401 — registra los modelos con Base
    Base.metadata.create_all(bind=engine_test)
    app.dependency_overrides[get_db] = override_get_db
    yield
    Base.metadata.drop_all(bind=engine_test)
    app.dependency_overrides.clear()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ════════════════════════════════════════════════════════════════════════════════
#  Tests de infraestructura
# ════════════════════════════════════════════════════════════════════════════════

class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "timestamp" in data
        assert "version" in data

    def test_root_redirects_to_docs(self, client):
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code in (301, 302, 307, 308)
        assert "/docs" in resp.headers.get("location", "")


# ════════════════════════════════════════════════════════════════════════════════
#  Tests de cálculo financiero (sin red — mocks internos)
# ════════════════════════════════════════════════════════════════════════════════

class TestBondDuration:
    """Tests del endpoint /bono/duracion — cálculo 100% interno, sin red."""

    def test_bono_basico(self, client):
        payload = {
            "face_value": 1000,
            "coupon_rate": 0.05,
            "ytm": 0.06,
            "periods": 10,
            "frequency": 2,
        }
        resp = client.post("/bono/duracion", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["precio"] > 0
        assert data["duracion_macaulay"] > 0
        assert data["duracion_modificada"] > 0
        assert data["convexidad"] > 0
        assert data["dv01"] > 0
        assert len(data["flujos"]) == 10

    def test_bono_par_cuando_coupon_eq_ytm(self, client):
        """Un bono cotiza a la par cuando el cupón es igual al YTM."""
        payload = {
            "face_value": 1000,
            "coupon_rate": 0.05,
            "ytm": 0.05,
            "periods": 10,
            "frequency": 2,
        }
        resp = client.post("/bono/duracion", json=payload)
        assert resp.status_code == 200
        precio = resp.json()["precio"]
        assert abs(precio - 1000.0) < 0.01  # debe ser ~1000

    def test_duracion_modificada_menor_que_macaulay(self, client):
        payload = {
            "face_value": 1000,
            "coupon_rate": 0.06,
            "ytm": 0.07,
            "periods": 20,
            "frequency": 2,
        }
        resp = client.post("/bono/duracion", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["duracion_modificada"] < data["duracion_macaulay"]

    def test_bono_invalido_cupon_negativo(self, client):
        payload = {
            "face_value": 1000,
            "coupon_rate": -0.01,  # inválido
            "ytm": 0.05,
            "periods": 10,
            "frequency": 2,
        }
        resp = client.post("/bono/duracion", json=payload)
        assert resp.status_code == 422


class TestOptionPricing:
    """Tests del endpoint /opcion/precio — Black-Scholes interno."""

    def test_call_at_the_money(self, client):
        payload = {"S": 100, "K": 100, "r": 0.05, "sigma": 0.2, "T": 1.0, "option_type": "call"}
        resp = client.post("/opcion/precio", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["precio"] > 0
        assert 0 < data["delta"] < 1     # call delta en [0,1]
        assert data["gamma"] > 0
        assert data["vega"] > 0
        assert data["theta"] < 0         # theta siempre negativo

    def test_put_call_parity(self, client):
        """Verifica paridad put-call: C - P = S - K·e^{-rT}"""
        params = {"S": 100, "K": 100, "r": 0.05, "sigma": 0.2, "T": 1.0}
        call_resp = client.post("/opcion/precio", json={**params, "option_type": "call"})
        put_resp = client.post("/opcion/precio", json={**params, "option_type": "put"})
        assert call_resp.status_code == 200
        assert put_resp.status_code == 200

        import math
        C = call_resp.json()["precio"]
        P = put_resp.json()["precio"]
        S, K, r, T = params["S"], params["K"], params["r"], params["T"]
        lhs = C - P
        rhs = S - K * math.exp(-r * T)
        assert abs(lhs - rhs) < 0.01  # tolerancia de 1 centavo

    def test_opcion_tipo_invalido(self, client):
        payload = {"S": 100, "K": 100, "r": 0.05, "sigma": 0.2, "T": 1.0, "option_type": "swap"}
        resp = client.post("/opcion/precio", json=payload)
        assert resp.status_code == 422


class TestPortfolios:
    """Tests CRUD de /portafolios con base de datos en memoria."""

    def test_crear_portafolio(self, client):
        payload = {
            "nombre": "Portafolio Test",
            "descripcion": "Prueba unitaria",
            "tickers": ["AAPL", "JPM"],
            "pesos": [0.6, 0.4],
            "capital": 50000,
        }
        resp = client.post("/portafolios", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["nombre"] == "Portafolio Test"
        assert data["tickers"] == ["AAPL", "JPM"]
        assert abs(sum(data["pesos"]) - 1.0) < 1e-6
        assert "id" in data

    def test_listar_portafolios(self, client):
        resp = client.get("/portafolios")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        assert len(resp.json()) >= 1

    def test_eliminar_portafolio(self, client):
        # Crear primero
        payload = {
            "nombre": "Para Eliminar",
            "tickers": ["AAPL"],
            "pesos": [1.0],
            "capital": 10000,
        }
        create_resp = client.post("/portafolios", json=payload)
        pid = create_resp.json()["id"]

        # Eliminar
        del_resp = client.delete(f"/portafolios/{pid}")
        assert del_resp.status_code == 204

        # Verificar que ya no existe
        list_resp = client.get("/portafolios")
        ids = [p["id"] for p in list_resp.json()]
        assert pid not in ids

    def test_eliminar_portafolio_inexistente(self, client):
        resp = client.delete("/portafolios/99999")
        assert resp.status_code == 404

    def test_portafolio_pesos_invalidos(self, client):
        payload = {
            "nombre": "Inválido",
            "tickers": ["AAPL", "JPM"],
            "pesos": [0.6, 0.6],  # suma != 1
            "capital": 10000,
        }
        resp = client.post("/portafolios", json=payload)
        assert resp.status_code == 422


class TestEWMAService:
    """Tests unitarios del servicio EWMA sin llamadas a red."""

    def test_ewma_calculo_basico(self):
        import numpy as np
        import pandas as pd
        from app.services import EWMAService

        rng = np.random.default_rng(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        returns = pd.Series(rng.normal(0, 0.01, 100), index=dates)

        svc = EWMAService()
        result = svc.compute(returns, lam=0.94)

        assert result["lambda"] == 0.94
        assert result["volatilidad_1d"] > 0
        assert result["volatilidad_anual"] > 0
        assert result["n_observaciones"] == 100
        assert len(result["serie"]) == 100

    def test_ewma_lambda_boundary(self):
        import numpy as np
        import pandas as pd
        from app.services import EWMAService

        rng = np.random.default_rng(0)
        dates = pd.date_range("2021-01-01", periods=50, freq="B")
        returns = pd.Series(rng.normal(0, 0.015, 50), index=dates)
        svc = EWMAService()

        # λ cercano a 1 → muy poca actualización (memoria larga)
        r_high = svc.compute(returns, lam=0.99)
        # λ bajo → actualización rápida (memoria corta)
        r_low = svc.compute(returns, lam=0.5)

        assert r_high["volatilidad_1d"] != r_low["volatilidad_1d"]


class TestNelsonSiegel:
    """Tests del ajuste Nelson-Siegel."""

    def test_nelson_siegel_formula(self):
        from app.services import YieldCurveService
        svc = YieldCurveService()
        # β₀=0.05, β₁=-0.02, β₂=0.01, τ=1.5
        y_0 = svc.nelson_siegel(0.01, 0.05, -0.02, 0.01, 1.5)
        y_10 = svc.nelson_siegel(10, 0.05, -0.02, 0.01, 1.5)
        assert 0 < y_0 < 0.1
        assert 0 < y_10 < 0.1

    def test_nelson_siegel_fit(self):
        from app.services import YieldCurveService
        svc = YieldCurveService()
        rates = {0.25: 0.053, 1.0: 0.052, 2.0: 0.049, 5.0: 0.047, 10.0: 0.046, 30.0: 0.048}
        result = svc.fit(rates)
        assert "parametros" in result
        assert "curva" in result
        assert result["rmse"] < 0.01  # ajuste razonablemente bueno
        assert len(result["curva"]) == 11
