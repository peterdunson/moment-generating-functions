import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mgf import *

st.title("Moment Generating Function Visualizer")

# Fixed t-domain
t = np.linspace(-0.5, 0.5, 400)

dist = st.selectbox(
    "Distribution",
    ["Normal", "Exponential", "Gamma", "Poisson", "Bernoulli", "Binomial"]
)

# Defaults
M = np.zeros_like(t)
mean = var = 0.0

if dist == "Normal":
    mu = st.slider("μ", -2.0, 2.0, 0.0)
    sigma = st.slider("σ", 0.5, 3.0, 1.0)
    M = mgf_normal(t, mu, sigma)
    mean, var = mu, sigma**2

elif dist == "Exponential":
    lam = st.slider("λ", 0.5, 3.0, 1.0)
    M = mgf_exponential(t, lam)
    M = np.where(t < lam - 0.05, M, np.nan)
    mean, var = 1/lam, 1/lam**2

elif dist == "Gamma":
    alpha = st.slider("α (shape)", 0.5, 5.0, 2.0)
    beta = st.slider("β (rate)", 0.5, 5.0, 1.0)
    M = mgf_gamma(t, alpha, beta)
    M = np.where(t < beta - 0.05, M, np.nan)
    mean, var = alpha/beta, alpha/(beta**2)

elif dist == "Poisson":
    lam = st.slider("λ", 0.5, 5.0, 2.0)
    M = mgf_poisson(t, lam)
    mean, var = lam, lam

elif dist == "Bernoulli":
    p = st.slider("p", 0.0, 1.0, 0.5)
    M = mgf_bernoulli(t, p)
    mean, var = p, p*(1-p)

elif dist == "Binomial":
    n = st.slider("n", 1, 20, 5)
    p = st.slider("p", 0.0, 1.0, 0.5)
    M = mgf_binomial(t, n, p)
    mean, var = n*p, n*p*(1-p)

# Plot
fig, ax = plt.subplots()
ax.plot(t, M, label="MGF")
ax.axvline(0, linestyle="--", color="gray")

# Fixed scale
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(0, 4)

if st.checkbox("Show 2nd-order Taylor approximation at t=0"):
    T = 1 + mean*t + 0.5*var*t**2
    ax.plot(t, T, "--", label="Taylor (2nd order)")

ax.set_xlabel("t")
ax.set_ylabel("M_X(t)")
ax.legend()

st.pyplot(fig)

st.markdown(f"**E[X] = {mean:.3f}**")
st.markdown(f"**Var(X) = {var:.3f}**")

