"""
Módulo 9 — Renta Fija
Curva de rendimiento Nelson-Siegel y análisis de bonos (Duración, Convexidad).
"""

import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

BACKEND = "http://localhost:8000"


def run():
    st.title("📈 Módulo 9 — Renta Fija")

    tab1, tab2 = st.tabs(["Curva de Rendimiento", "Análisis de Bonos"])

    # ── Tab 1: Curva Nelson-Siegel ─────────────────────────────────────────────
    with tab1:
        st.subheader("Curva de Rendimiento Nelson-Siegel")
        st.markdown(
            "Tasas del Tesoro de EE.UU. ajustadas con el modelo Nelson-Siegel: "
            "**y(T) = β₀ + β₁·f₁(T,τ) + β₂·f₂(T,τ)**"
        )

        if st.button("Obtener Curva de Rendimiento", type="primary"):
            with st.spinner("Obteniendo tasas y ajustando Nelson-Siegel..."):
                try:
                    resp = requests.get(f"{BACKEND}/curva-rendimiento", timeout=30)
                    resp.raise_for_status()
                    data = resp.json()

                    # Parámetros del modelo
                    params = data["parametros"]
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("β₀ (Nivel largo plazo)", f"{params['beta_0']:.4f}")
                    col2.metric("β₁ (Pendiente)", f"{params['beta_1']:.4f}")
                    col3.metric("β₂ (Curvatura)", f"{params['beta_2']:.4f}")
                    col4.metric("τ (Factor de escala)", f"{params['tau']:.4f}")
                    st.metric("RMSE del ajuste", f"{data['rmse']:.6f}")

                    # Gráfico de la curva
                    curva = data["curva"]
                    plazos = [p["plazo"] for p in curva]
                    tasas_ns = [p["tasa_ns"] * 100 for p in curva]
                    obs = {p["plazo"]: p["tasa_observada"] for p in curva if p["tasa_observada"] is not None}

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=plazos, y=tasas_ns, mode="lines",
                        name="Nelson-Siegel", line=dict(color="#6366F1", width=3)
                    ))
                    if obs:
                        fig.add_trace(go.Scatter(
                            x=list(obs.keys()), y=[v * 100 for v in obs.values()],
                            mode="markers", name="Tasas Observadas (FRED)",
                            marker=dict(size=10, color="#F59E0B", symbol="circle")
                        ))
                    fig.update_layout(
                        title="Curva de Rendimiento del Tesoro — Nelson-Siegel",
                        xaxis_title="Plazo (años)",
                        yaxis_title="Tasa (%)",
                        template="plotly_dark",
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Tasas observadas
                    st.subheader("Tasas Observadas")
                    tasas_df = {
                        "Plazo": list(data["tasas_observadas"].keys()),
                        "Tasa (%)": [v * 100 for v in data["tasas_observadas"].values()],
                    }
                    st.dataframe(tasas_df, use_container_width=True)

                except Exception as e:
                    st.error(f"Error: {e}")

    # ── Tab 2: Análisis de Bonos ───────────────────────────────────────────────
    with tab2:
        st.subheader("Análisis de Bono de Cupón Fijo")
        st.markdown(
            "Calcula el **precio**, **duración de Macaulay**, **duración modificada**, "
            "**convexidad** y **DV01** de un bono."
        )

        col1, col2 = st.columns(2)
        with col1:
            face_value = st.number_input("Valor nominal (USD)", value=1000.0, min_value=1.0)
            coupon_rate = st.number_input("Tasa de cupón anual (%)", value=5.0, min_value=0.0, max_value=100.0) / 100
            ytm = st.number_input("Yield to Maturity anual (%)", value=6.0, min_value=0.01, max_value=100.0) / 100
        with col2:
            periods = st.number_input("Número de períodos", value=10, min_value=1, max_value=120)
            frequency = st.selectbox("Frecuencia de pago", options=[1, 2, 4], index=1,
                                      format_func=lambda x: {1: "Anual", 2: "Semestral", 4: "Trimestral"}[x])

        if st.button("Calcular Duración y Convexidad", type="primary"):
            with st.spinner("Calculando..."):
                payload = {
                    "face_value": face_value,
                    "coupon_rate": coupon_rate,
                    "ytm": ytm,
                    "periods": periods,
                    "frequency": frequency,
                }
                try:
                    resp = requests.post(f"{BACKEND}/bono/duracion", json=payload, timeout=15)
                    resp.raise_for_status()
                    data = resp.json()

                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Precio", f"${data['precio']:,.2f}")
                    col2.metric("Duración Macaulay", f"{data['duracion_macaulay']:.4f} años")
                    col3.metric("Duración Modificada", f"{data['duracion_modificada']:.4f}")
                    col4.metric("Convexidad", f"{data['convexidad']:.4f}")
                    col5.metric("DV01", f"${data['dv01']:.4f}")

                    # Interpretación
                    dm = data["duracion_modificada"]
                    st.info(
                        f"Con una duración modificada de **{dm:.4f}**, un incremento de 100 pb "
                        f"en el YTM causaría una pérdida aproximada de **{dm:.2f}%** en el precio del bono."
                    )

                    # Flujos de caja
                    st.subheader("Flujos de Caja")
                    flujos = data["flujos"]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=[f["año"] for f in flujos],
                        y=[f["flujo"] for f in flujos],
                        name="Flujo de caja",
                        marker_color="#6366F1",
                    ))
                    fig.add_trace(go.Scatter(
                        x=[f["año"] for f in flujos],
                        y=[f["pv_flujo"] for f in flujos],
                        mode="lines+markers",
                        name="Valor presente",
                        line=dict(color="#F59E0B"),
                    ))
                    fig.update_layout(
                        title="Flujos de Caja y Valor Presente",
                        xaxis_title="Año",
                        yaxis_title="USD",
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error: {e}")


if __name__ == "__main__":
    run()
