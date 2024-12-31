import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import altair as alt

def d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def black_scholes(S, K, T, r, sigma):
    D1 = d1(S, K, T, r, sigma)
    D2 = d2(S, K, T, r, sigma)
    call_price = S * norm.cdf(D1) - K * np.exp(-r * T) * norm.cdf(D2)
    put_price = K * np.exp(-r * T) * norm.cdf(-D2) - S * norm.cdf(-D1)
    return call_price, put_price

def calculate_greeks(S, K, T, r, sigma):
    D1 = d1(S, K, T, r, sigma)
    D2 = d2(S, K, T, r, sigma)
    
    delta = norm.cdf(D1)
    theta = -((S * norm.pdf(D1) * sigma) / (2 * np.sqrt(T))) - r * K * np.exp(-r * T) * norm.cdf(D2)
    vega = S * norm.pdf(D1) * np.sqrt(T)
    rho = K * T * np.exp(-r * T) * norm.cdf(D2)
    
    return delta, theta, vega, rho

st.set_page_config(page_title="Black-Scholes Pricing Model", layout="wide")

st.title("📈 Black-Scholes Pricing Model")

with st.sidebar:
    st.header("Input Parameters")
    S = st.slider("Stock Price (S)", min_value=10.0, max_value=500.0, value=100.0, step=1.0)
    K = st.slider("Strike Price (K)", min_value=10.0, max_value=500.0, value=100.0, step=1.0)
    T_days = st.slider("Time to Maturity (T) in days", min_value=1, max_value=1825, value=365, step=1)
    r = st.slider("Risk-Free Rate (r)", min_value=0.0, max_value=0.2, value=0.05, step=0.001)
    sigma = st.slider("Volatility (σ)", min_value=0.01, max_value=2.0, value=0.2, step=0.01)

T_years = T_days / 365

call_price, put_price = black_scholes(S, K, T_years, r, sigma)

st.markdown("---")
st.subheader("Key Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Stock Price", f"${S:.2f}")

with col2:
    st.metric("Call Option Price", f"${call_price:.2f}")

with col3:
    st.metric("Put Option Price", f"${put_price:.2f}")

K_values = np.linspace(0.5 * S, 1.5 * S, 100)
call_prices = [black_scholes(S, K, T_years, r, sigma)[0] for K in K_values]
put_prices = [black_scholes(S, K, T_years, r, sigma)[1] for K in K_values]

y_min = min(call_prices + put_prices) * 0.9
y_max = max(call_prices + put_prices) * 1.1

data = pd.DataFrame({
    "Strike Price (K)": np.tile(K_values, 2),
    "Option Price": call_prices + put_prices,
    "Option Type": ["Call"] * len(K_values) + ["Put"] * len(K_values)
})

chart = alt.Chart(data).mark_line().encode(
    x=alt.X("Strike Price (K):Q", title="Strike Price (K)", scale=alt.Scale(domain=[min(K_values), max(K_values)])),
    y=alt.Y("Option Price:Q", title="Option Price", scale=alt.Scale(domain=[y_min, y_max])),
    color="Option Type:N",
    tooltip=["Strike Price (K)", "Option Price", "Option Type"]
).properties(
    width=800,
    height=400,
    title="Option Prices vs Strike Price"
)

st.altair_chart(chart, use_container_width=True)

delta, theta, vega, rho = calculate_greeks(S, K, T_years, r, sigma)

st.subheader("Greeks")
greeks = pd.DataFrame({
    "Metric": ["Delta (Δ)", "Theta (Θ)", "Vega (ν)", "Rho (ρ)"],
    "Value": [delta, theta, vega, rho]
})

greeks = greeks.style.hide(axis="index")
st.table(greeks.data)
