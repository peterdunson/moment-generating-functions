import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mgf import *

st.title("Moment Generating Function — Moment Geometry")

# Fixed domain
t = np.linspace(-0.5, 0.5, 400)

dist = st.selectbox(
    "Distribution",
    ["Normal", "Exponential", "Gamma", "Poisson", "Bernoulli", "Binomial"]
)

# Initialize
M = np.zeros_like(t)
mean = var = m3 = m4 = 0.0

# ---- Distributions ----
if dist == "Normal":
    mu = st.slider("μ", -2.0, 2.0, 0.0)
    sigma = st.slider("σ", 0.5, 3.0, 1.0)
    M = mgf_normal(t, mu, sigma)
    mean = mu
    var = sigma**2
    m3 = mu**3 + 3*mu*var
    m4 = 3*var**2 + 6*mu**2*var + mu**4

elif dist == "Exponential":
    lam = st.slider("λ", 0.5, 3.0, 1.0)
    M = mgf_exponential(t, lam)
    M = np.where(t < lam - 0.05, M, np.nan)
    mean = 1/lam
    var = 1/lam**2
    m3 = 6/lam**3
    m4 = 24/lam**4

elif dist == "Gamma":
    alpha = st.slider("α", 0.5, 5.0, 2.0)
    beta = st.slider("β", 0.5, 5.0, 1.0)
    M = mgf_gamma(t, alpha, beta)
    M = np.where(t < beta - 0.05, M, np.nan)
    mean = alpha/beta
    var = alpha/(beta**2)
    m3 = 2*alpha/(beta**3)
    m4 = 3*alpha*(alpha+2)/(beta**4)

elif dist == "Poisson":
    lam = st.slider("λ", 0.5, 5.0, 2.0)
    M = mgf_poisson(t, lam)
    mean = var = lam
    m3 = lam
    m4 = lam + 3*lam**2

elif dist == "Bernoulli":
    p = st.slider("p", 0.0, 1.0, 0.5)
    M = mgf_bernoulli(t, p)
    mean = p
    var = p*(1-p)
    m3 = p
    m4 = p

elif dist == "Binomial":
    n = st.slider("n", 1, 20, 5)
    p = st.slider("p", 0.0, 1.0, 0.5)
    M = mgf_binomial(t, n, p)
    mean = n*p
    var = n*p*(1-p)
    m3 = n*p
    m4 = n*p*(1 + 3*(n-1)*p)

# ---- Main Plot ----
fig, ax = plt.subplots()
ax.plot(t, M, label="MGF")
ax.axvline(0, linestyle="--", color="gray")

# Fixed axes
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(0, 4)

# ---- Moment visuals ----
if st.checkbox("Show tangent at t = 0 (mean)"):
    tangent = 1 + mean*t
    ax.plot(t, tangent, "--", label="Tangent (1st moment)")

if st.checkbox("Show 2nd-order Taylor (variance)"):
    T2 = 1 + mean*t + 0.5*var*t**2
    ax.plot(t, T2, "--", label="2nd order")

if st.checkbox("Show 3rd-order Taylor"):
    T3 = 1 + mean*t + 0.5*var*t**2 + (1/6)*m3*t**3
    ax.plot(t, T3, "--", label="3rd order")

if st.checkbox("Show 4th-order Taylor"):
    T4 = (
        1 + mean*t
        + 0.5*var*t**2
        + (1/6)*m3*t**3
        + (1/24)*m4*t**4
    )
    ax.plot(t, T4, "--", label="4th order")

ax.set_xlabel("t")
ax.set_ylabel("M_X(t)")
ax.legend()
st.pyplot(fig)

# ---- Derivative plots ----
if st.checkbox("Show derivatives near t = 0"):
    dt = t[1] - t[0]
    M1 = np.gradient(M, dt)
    M2 = np.gradient(M1, dt)
    M3 = np.gradient(M2, dt)

    fig2, axs = plt.subplots(3, 1, sharex=True, figsize=(5, 6))

    axs[0].plot(t, M1)
    axs[0].axvline(0)
    axs[0].set_title("M'(t)")

    axs[1].plot(t, M2)
    axs[1].axvline(0)
    axs[1].set_title("M''(t)")

    axs[2].plot(t, M3)
    axs[2].axvline(0)
    axs[2].set_title("M'''(t)")

    for axd in axs:
        axd.set_xlim(-0.2, 0.2)

    st.pyplot(fig2)

# ---- Numeric moments ----
st.markdown("### Moments from derivatives at 0")
st.markdown(f"**E[X] = {mean:.4f}**")
st.markdown(f"**Var(X) = {var:.4f}**")
st.markdown(f"**E[X³] = {m3:.4f}**")
st.markdown(f"**E[X⁴] = {m4:.4f}**")
