"""
Módulo 10 — Derivados
Black-Scholes para opciones europeas, 5 Greeks y volatilidad implícita.
"""

import requests
import streamlit as st
import plotly.graph_objects as go
import numpy as np

BACKEND = "http://localhost:8000"


def run():
    st.title("⚡ Módulo 10 — Derivados: Opciones")
    st.markdown(
        "Valuación de opciones europeas con **Black-Scholes**, "
        "cálculo de los **5 Greeks** y **volatilidad implícita** vía Newton-Raphson."
    )

    col1, col2 = st.columns(2)
    with col1:
        S = st.number_input("S — Precio del subyacente (USD)", value=100.0, min_value=0.01)
        K = st.number_input("K — Precio de ejercicio (strike, USD)", value=100.0, min_value=0.01)
        sigma = st.number_input("σ — Volatilidad anual (%)", value=20.0, min_value=0.1, max_value=500.0) / 100
        option_type = st.radio("Tipo de opción", options=["call", "put"], horizontal=True)
    with col2:
        T = st.number_input("T — Tiempo al vencimiento (años)", value=1.0, min_value=0.01, max_value=30.0)
        r = st.number_input("r — Tasa libre de riesgo (%)", value=5.0, min_value=0.0, max_value=50.0) / 100
        market_price = st.number_input(
            "Precio de mercado (opcional, para vol. implícita)", value=0.0, min_value=0.0
        )
        market_price = market_price if market_price > 0 else None

    if st.button("Calcular Precio y Greeks", type="primary"):
        payload = {
            "S": S, "K": K, "r": r, "sigma": sigma, "T": T,
            "option_type": option_type,
            "market_price": market_price,
        }
        with st.spinner("Calculando Black-Scholes..."):
            try:
                resp = requests.post(f"{BACKEND}/opcion/precio", json=payload, timeout=15)
                resp.raise_for_status()
                data = resp.json()

                # Métricas principales
                st.subheader("Precio Black-Scholes")
                col1, col2 = st.columns(2)
                col1.metric("Precio de la opción", f"${data['precio']:.4f}")
                if data.get("volatilidad_implicita"):
                    col2.metric("Volatilidad Implícita", f"{data['volatilidad_implicita']*100:.2f}%")

                # 5 Greeks
                st.subheader("Los 5 Greeks")
                g1, g2, g3, g4, g5 = st.columns(5)
                g1.metric("Δ Delta", f"{data['delta']:.4f}",
                          help="Sensibilidad al precio del subyacente")
                g2.metric("Γ Gamma", f"{data['gamma']:.6f}",
                          help="Tasa de cambio del Delta")
                g3.metric("ν Vega", f"{data['vega']:.4f}",
                          help="Sensibilidad a la volatilidad (por 1%)")
                g4.metric("Θ Theta", f"{data['theta']:.4f}",
                          help="Decaimiento temporal (por día)")
                g5.metric("ρ Rho", f"{data['rho']:.4f}",
                          help="Sensibilidad a la tasa libre de riesgo (por 1%)")

                st.caption(f"d₁ = {data['d1']:.4f} | d₂ = {data['d2']:.4f}")

                # Sensibilidad del precio al subyacente
                st.subheader("Sensibilidad del Precio al Subyacente")
                S_range = np.linspace(S * 0.5, S * 1.5, 100)
                prices = []
                for s in S_range:
                    p2 = requests.post(f"{BACKEND}/opcion/precio", json={
                        "S": float(s), "K": K, "r": r, "sigma": sigma, "T": T, "option_type": option_type
                    }, timeout=15).json()
                    prices.append(p2["precio"])

                intrinsic = [max(s - K, 0) if option_type == "call" else max(K - s, 0) for s in S_range]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=S_range.tolist(), y=prices,
                                         mode="lines", name="Precio Black-Scholes",
                                         line=dict(color="#6366F1", width=3)))
                fig.add_trace(go.Scatter(x=S_range.tolist(), y=intrinsic,
                                         mode="lines", name="Valor intrínseco",
                                         line=dict(color="#F59E0B", dash="dash")))
                fig.add_vline(x=S, line_dash="dot", line_color="white",
                              annotation_text=f"S={S}")
                fig.add_vline(x=K, line_dash="dot", line_color="red",
                              annotation_text=f"K={K}")
                fig.update_layout(
                    title=f"Precio de la {option_type.upper()} vs. Precio del Subyacente",
                    xaxis_title="Precio del subyacente (USD)",
                    yaxis_title="Precio de la opción (USD)",
                    template="plotly_dark",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Superficie de volatilidad implícita (smile)
                st.subheader("Smile de Volatilidad (sensibilidad a σ)")
                sigmas = np.linspace(0.05, 0.80, 50)
                prices_vs = []
                for sig in sigmas:
                    p3 = requests.post(f"{BACKEND}/opcion/precio", json={
                        "S": S, "K": K, "r": r, "sigma": float(sig), "T": T, "option_type": option_type
                    }, timeout=15).json()
                    prices_vs.append(p3["precio"])

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=(sigmas * 100).tolist(), y=prices_vs,
                                          mode="lines", name="Precio",
                                          line=dict(color="#10B981", width=2)))
                fig2.add_vline(x=sigma * 100, line_dash="dot", line_color="white",
                               annotation_text=f"σ={sigma*100:.0f}%")
                fig2.update_layout(
                    title="Precio de la Opción vs. Volatilidad (σ)",
                    xaxis_title="Volatilidad σ (%)",
                    yaxis_title="Precio (USD)",
                    template="plotly_dark",
                )
                st.plotly_chart(fig2, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")


if __name__ == "__main__":
    run()
