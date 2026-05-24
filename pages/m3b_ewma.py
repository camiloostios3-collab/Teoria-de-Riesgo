"""
Módulo 3b — Volatilidad EWMA
Volatilidad condicional con EWMA (Exponentially Weighted Moving Average).
"""

import requests
import streamlit as st
import plotly.graph_objects as go

BACKEND = "http://localhost:8000"


def run():
    st.title("📊 Módulo 3b — Volatilidad EWMA")
    st.markdown(
        "Calcula la **volatilidad EWMA** con factor de decaimiento λ configurable. "
        "Fórmula recursiva: **σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}**"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.text_input("Ticker", value="AAPL").upper()
    with col2:
        years = st.slider("Años de historia", 1, 10, 3)
    with col3:
        lam = st.slider(
            "Factor λ (RiskMetrics = 0.94)",
            min_value=0.50, max_value=0.99, value=0.94, step=0.01,
        )

    st.info(
        f"λ = {lam:.2f} — {'Mayor memoria (suavizado)' if lam >= 0.90 else 'Menor memoria (reactivo)'}. "
        "RiskMetrics usa λ=0.94 para datos diarios."
    )

    if st.button("Calcular EWMA", type="primary"):
        with st.spinner(f"Calculando volatilidad EWMA para {ticker}..."):
            payload = {"ticker": ticker, "years": years, "lam": lam}
            try:
                resp = requests.post(f"{BACKEND}/volatilidad", json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()

                col1, col2, col3 = st.columns(3)
                col1.metric("λ utilizado", f"{data['lam']:.2f}")
                col2.metric("Volatilidad EWMA 1D", f"{data['volatilidad_1d']*100:.4f}%")
                col3.metric("Volatilidad EWMA Anual", f"{data['volatilidad_anual']*100:.2f}%")

                # Serie temporal de volatilidad
                serie = data["serie"]
                fechas = [p["fecha"] for p in serie]
                vols = [p["volatilidad_diaria"] * 100 for p in serie]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=fechas, y=vols,
                    mode="lines",
                    name=f"Volatilidad EWMA (λ={lam})",
                    line=dict(color="#6366F1", width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(99,102,241,0.15)",
                ))
                fig.update_layout(
                    title=f"Volatilidad EWMA — {ticker} | λ={lam}",
                    xaxis_title="Fecha",
                    yaxis_title="Volatilidad diaria (%)",
                    template="plotly_dark",
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Comparación λ = 0.94 vs. λ = 0.75
                st.subheader("Comparación de λ")
                resp2 = requests.post(f"{BACKEND}/volatilidad",
                                       json={"ticker": ticker, "years": years, "lam": 0.75},
                                       timeout=60)
                data2 = resp2.json()
                vols2 = [p["volatilidad_diaria"] * 100 for p in data2["serie"]]

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=fechas, y=vols,
                                          mode="lines", name=f"λ={lam}",
                                          line=dict(color="#6366F1")))
                fig2.add_trace(go.Scatter(x=fechas, y=vols2,
                                          mode="lines", name="λ=0.75",
                                          line=dict(color="#F59E0B")))
                fig2.update_layout(
                    title=f"EWMA: λ={lam} vs λ=0.75",
                    xaxis_title="Fecha",
                    yaxis_title="Volatilidad diaria (%)",
                    template="plotly_dark",
                )
                st.plotly_chart(fig2, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")


if __name__ == "__main__":
    run()
