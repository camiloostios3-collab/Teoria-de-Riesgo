"""
Módulo 11 — Pruebas de Estrés y Machine Learning
Stress testing con 3 escenarios históricos + predicción ML de retornos.
"""

import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

BACKEND = "http://localhost:8000"

DEFAULT_TICKERS = ["AAPL", "JPM", "XOM", "JNJ", "AMZN"]
DEFAULT_WEIGHTS = [0.25, 0.25, 0.20, 0.15, 0.15]


def run():
    st.title("🔴 Módulo 11 — Pruebas de Estrés & Machine Learning")

    tab1, tab2 = st.tabs(["Pruebas de Estrés", "Predicción ML"])

    # ── Tab 1: Stress Testing ──────────────────────────────────────────────────
    with tab1:
        st.subheader("Stress Testing del Portafolio")
        st.markdown(
            "Aplica **3 escenarios de estrés históricos** al portafolio y calcula "
            "el impacto en P&L y el VaR estresado."
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            tickers_str = st.text_input(
                "Tickers (separados por coma)",
                value=", ".join(DEFAULT_TICKERS),
            )
            tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
        with col2:
            capital = st.number_input("Capital (USD)", value=100_000, min_value=1_000)

        # Pesos
        st.write("Pesos del portafolio:")
        cols = st.columns(len(tickers))
        pesos = []
        for i, (col, t) in enumerate(zip(cols, tickers)):
            w = col.number_input(
                t, value=DEFAULT_WEIGHTS[i] if i < len(DEFAULT_WEIGHTS) else round(1 / len(tickers), 2),
                min_value=0.0, max_value=1.0, step=0.05, key=f"stress_w_{i}"
            )
            pesos.append(w)

        suma = sum(pesos)
        if abs(suma - 1.0) > 1e-3:
            st.warning(f"Los pesos suman {suma:.4f} — deben sumar 1.0")

        if st.button("Ejecutar Stress Testing", type="primary", disabled=abs(suma - 1.0) > 1e-3):
            payload = {"tickers": tickers, "weights": pesos, "capital": capital, "years": 3}
            with st.spinner("Aplicando escenarios de estrés..."):
                try:
                    resp = requests.post(f"{BACKEND}/stress", json=payload, timeout=60)
                    resp.raise_for_status()
                    data = resp.json()

                    # Métricas base
                    st.subheader("Situación Base del Portafolio")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Capital", f"${data['capital']:,.0f}")
                    col2.metric("VaR Base (95%)", f"{data['var_base_pct']:.2f}%",
                                delta=f"-${data['var_base_usd']:,.0f}")
                    col3.metric("Volatilidad Anual", f"{data['volatilidad_base_anual']:.2f}%")

                    # Resultados por escenario
                    st.subheader("Resultados por Escenario de Estrés")
                    for esc in data["escenarios"]:
                        with st.expander(f"🔴 {esc['escenario'].replace('_', ' ').title()} — P&L: {esc['pnl_pct']:+.1f}%"):
                            st.markdown(f"**{esc['descripcion']}**")
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Shock Equity", f"{esc['shock_equity_pct']:+.1f}%")
                            c2.metric("P&L estimado", f"{esc['pnl_pct']:+.1f}%",
                                      delta=f"${esc['pnl_usd']:+,.0f}")
                            c3.metric("VaR Estresado", f"{esc['var_estresado_pct']:.2f}%")
                            c4.metric("Incremento VaR", f"{esc['incremento_var_pct']:+.1f}%")

                    # Comparación gráfica
                    st.subheader("Comparación de VaR: Base vs. Estresado")
                    escenarios = [e["escenario"].replace("_", " ").title() for e in data["escenarios"]]
                    var_base = [data["var_base_usd"]] * len(escenarios)
                    var_stress = [e["var_estresado_usd"] for e in data["escenarios"]]
                    pnl = [e["pnl_usd"] for e in data["escenarios"]]

                    fig = go.Figure()
                    fig.add_trace(go.Bar(name="VaR Base", x=escenarios, y=var_base,
                                         marker_color="#6366F1"))
                    fig.add_trace(go.Bar(name="VaR Estresado", x=escenarios, y=var_stress,
                                         marker_color="#EF4444"))
                    fig.add_trace(go.Bar(name="P&L (pérdida)", x=escenarios,
                                         y=[abs(p) for p in pnl], marker_color="#F59E0B"))
                    fig.update_layout(
                        barmode="group",
                        title="VaR y Pérdida Estimada por Escenario de Estrés",
                        yaxis_title="USD",
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error: {e}")

    # ── Tab 2: Predicción ML ───────────────────────────────────────────────────
    with tab2:
        st.subheader("Predicción ML de Retornos")
        st.markdown(
            "Usa un modelo **Random Forest** pre-entrenado para predecir el retorno "
            "del siguiente día hábil. Cada predicción queda registrada en la base de datos."
        )

        col1, col2 = st.columns(2)
        with col1:
            ticker_ml = st.selectbox(
                "Activo a predecir",
                options=["AAPL", "JPM", "XOM", "JNJ", "AMZN", "MSFT", "NVDA", "GOOGL"],
            )
        with col2:
            years_ml = st.slider("Años de historia para features", 1, 5, 3)

        if st.button("Generar Predicción", type="primary"):
            with st.spinner(f"Generando predicción para {ticker_ml}..."):
                try:
                    resp = requests.post(f"{BACKEND}/predict",
                                         json={"ticker": ticker_ml, "years": years_ml},
                                         timeout=60)
                    resp.raise_for_status()
                    data = resp.json()

                    pred = data["prediccion_retorno_pct"]
                    color = "#10B981" if pred >= 0 else "#EF4444"
                    emoji = "📈" if pred >= 0 else "📉"

                    st.markdown(
                        f"<h2 style='color:{color};text-align:center;'>{emoji} "
                        f"Retorno predicho: {pred:+.4f}%</h2>",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Modelo: {data['modelo']}")

                    # Features utilizadas
                    st.subheader("Features del Modelo")
                    features = data["features"]
                    feat_names = {
                        "media_5d": "Media 5 días",
                        "media_20d": "Media 20 días",
                        "vol_5d": "Volatilidad 5 días",
                        "vol_20d": "Volatilidad 20 días",
                        "momentum_5d": "Momentum 5 días",
                        "rsi_proxy": "RSI (proxy)",
                    }

                    fig = go.Figure(go.Bar(
                        x=[feat_names.get(k, k) for k in features.keys()],
                        y=list(features.values()),
                        marker_color=["#10B981" if v >= 0 else "#EF4444" for v in features.values()],
                    ))
                    fig.update_layout(
                        title="Features utilizadas en la predicción",
                        yaxis_title="Valor",
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.success("Predicción registrada en la base de datos SQLite (tabla prediction_logs).")

                except Exception as e:
                    st.error(f"Error: {e}")


if __name__ == "__main__":
    run()
