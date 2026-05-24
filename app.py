"""
app.py — RiskLab USTA
Streamlit | Python 3.12.1 | Modo claro refinado
"""

import streamlit as st

st.set_page_config(
    page_title="RiskLab · USTA",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.styles import GLOBAL_CSS
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
st.markdown("""
<style>
[data-testid="stSidebar"] .stRadio > label { display: none !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="padding:1.8rem 1.2rem 1.4rem;border-bottom:1px solid #D8DDE8;margin-bottom:0.8rem;">
        <div style="font-family:'Playfair Display',serif;font-size:1.35rem;
                    font-weight:700;color:#1A2035;letter-spacing:0.01em;line-height:1.1;">
            Risk<span style="color:#8B6914;font-style:italic;">Lab</span>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.52rem;
                    color:#8896A8;letter-spacing:0.22em;text-transform:uppercase;margin-top:5px;">
            USTA · Teoría del Riesgo
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.5rem;color:#8896A8;
                letter-spacing:0.3em;text-transform:uppercase;padding:0.2rem 1.2rem 0.5rem;">
        Módulos
    </div>
    """, unsafe_allow_html=True)

    modulo = st.radio(
        label="navegacion",
        options=[
            "Vista General",
            "Análisis Técnico",
            "Rendimientos",
            "ARCH / GARCH",
            "Volatilidad EWMA",
            "CAPM & Beta",
            "VaR & CVaR",
            "Markowitz",
            "Señales & Alertas",
            "Macro & Benchmark",
            "Renta Fija",
            "Derivados",
            "Estrés & ML",
        ],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div style="margin-top:2rem;padding:1rem 1.2rem;border-top:1px solid #D8DDE8;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.54rem;
                    color:#8896A8;line-height:2.0;">
            <span style="color:#1A6B4A;font-size:0.48rem;">●</span>
            <span style="color:#1A6B4A;"> API activa</span><br>
            AAPL · JPM · XOM · JNJ · AMZN<br>
            Yahoo Finance · Tiempo real<br>
            <span style="color:#C4CBD8;">S&P 500 · Benchmark</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if modulo == "Vista General":
    from pages.overview import show
    show()
elif modulo == "Análisis Técnico":
    from pages.m1_technical import show
    show()
elif modulo == "Rendimientos":
    from pages.m2_returns import show
    show()
elif modulo == "ARCH / GARCH":
    from pages.m3_garch import show
    show()
elif modulo == "CAPM & Beta":
    from pages.m4_capm import show
    show()
elif modulo == "VaR & CVaR":
    from pages.m5_var import show
    show()
elif modulo == "Markowitz":
    from pages.m6_markowitz import show
    show()
elif modulo == "Señales & Alertas":
    from pages.m7_signals import show
    show()
elif modulo == "Macro & Benchmark":
    from pages.m8_macro import show
    show()
elif modulo == "Volatilidad EWMA":
    from pages.m3b_ewma import run
    run()
elif modulo == "Renta Fija":
    from pages.m9_fixed_income import run
    run()
elif modulo == "Derivados":
    from pages.m10_options import run
    run()
elif modulo == "Estrés & ML":
    from pages.m11_stress import run
    run()