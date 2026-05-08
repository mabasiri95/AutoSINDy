# -*- coding: utf-8 -*-
"""
systems.py  —  Dynamical System Data Generators
================================================
Each generator now accepts a `noise_seed` parameter (default 42).
Pass a different integer per trial so each trial sees a genuinely
different noise realisation while keeping everything else fixed.

In your sweep (via run_single.py):
    cfg["data_params"][sys]["noise_seed"] = args.seed   # 32, 33, 34 …
"""

import numpy as np
from scipy.integrate import solve_ivp
import pysindy as ps


# ── Helper ────────────────────────────────────────────────────────────────────
def _add_noise_and_differentiate(X_clean, t_eval, noise_level, noise_seed):
    """Shared noise + derivative logic for all generators."""
    rng = np.random.RandomState(noise_seed)
    noise = rng.randn(*X_clean.shape) * np.std(X_clean, axis=0) * noise_level
    X_noisy = X_clean + noise
    X_dot = ps.SmoothedFiniteDifference()._differentiate(X_noisy, t_eval)
    return X_noisy, X_dot


# ── Harmonic Oscillator (damped) ──────────────────────────────────────────────
def generate_harmonic_oscillator_data(
        k1=5.0, k2=1.0, x0=[1.0, 0.0], t_end=10, n_samples=5000,
        noise_level=0.01, noise_seed=42):
    """
    dx/dt = y
    dy/dt = -k*x - (k2)*y     (lightly damped)
    """
    def system(t, x): return [x[1], -k1 * x[0] - (k2) * x[1]]

    t_eval = np.linspace(0, t_end, n_samples)
    X_clean = solve_ivp(system, [0, t_end], x0,
                        t_eval=t_eval, dense_output=True).y.T
    X_noisy, X_dot = _add_noise_and_differentiate(
        X_clean, t_eval, noise_level, noise_seed)
    print("✓ Step 1: Generated synthetic Harmonic Oscillator data.")
    return X_noisy, X_dot, t_eval


# ── Van der Pol ───────────────────────────────────────────────────────────────
def generate_vanderpol_data(
        mu=2.0, x0=[0.5, 0.5], t_end=25, n_samples=5000,
        noise_level=0.01, noise_seed=42):
    """
    dx/dt = y
    dy/dt = mu*(1 - x²)*y - x
    """
    def system(t, x): return [x[1], mu * (1 - x[0]**2) * x[1] - x[0]]

    t_eval = np.linspace(0, t_end, n_samples)
    X_clean = solve_ivp(system, [0, t_end], x0,
                        t_eval=t_eval, dense_output=True).y.T
    X_noisy, X_dot = _add_noise_and_differentiate(
        X_clean, t_eval, noise_level, noise_seed)
    print("✓ Step 1: Generated synthetic Van der Pol data.")
    return X_noisy, X_dot, t_eval


# ── Damped Pendulum ───────────────────────────────────────────────────────────
def generate_damped_pendulum_data(
        b=0.25, c=5.0, x0=[np.pi - 0.1, 0], t_end=10, n_samples=5000,
        noise_level=0.01, noise_seed=42):
    """
    dx/dt = y
    dy/dt = -b*y - c*sin(x)
    """
    def system(t, x): return [x[1], -b * x[1] - c * np.sin(x[0])]

    t_eval = np.linspace(0, t_end, n_samples)
    X_clean = solve_ivp(system, [0, t_end], x0,
                        t_eval=t_eval, dense_output=True).y.T
    X_noisy, X_dot = _add_noise_and_differentiate(
        X_clean, t_eval, noise_level, noise_seed)
    print("✓ Step 1: Generated synthetic Damped Pendulum data.")
    return X_noisy, X_dot, t_eval


# ── Lorenz ────────────────────────────────────────────────────────────────────
def generate_lorenz_data(
        sigma=10.0, rho=28.0, beta=8./3., x0=[-8.0, 8.0, 27.0],
        t_end=50, n_samples=5000, noise_level=0.01, noise_seed=42):
    """
    x' = sigma*(y - x)
    y' = x*(rho - z) - y
    z' = x*y - beta*z
    """
    def system(t, x):
        return [sigma*(x[1]-x[0]), x[0]*(rho-x[2])-x[1], x[0]*x[1]-beta*x[2]]

    t_eval = np.linspace(0, t_end, n_samples)
    X_clean = solve_ivp(system, [0, t_end], x0,
                        t_eval=t_eval, dense_output=True).y.T
    X_noisy, X_dot = _add_noise_and_differentiate(
        X_clean, t_eval, noise_level, noise_seed)
    print("✓ Step 1: Generated synthetic Lorenz System data.")
    return X_noisy, X_dot, t_eval


# ── Complex Lorenz ────────────────────────────────────────────────────────────
def generate_complex_lorenz_data(
        sigma=10.0, rho=28.0, beta=8./3., gamma=1.5,
        x0=[-8.0, 8.0, 27.0], t_end=50, n_samples=5000,
        noise_level=0.01, noise_seed=42):
    """
    x' = sigma*(y - x)
    y' = x*(rho - z) - y
    z' = x*y - beta*z + gamma*y*sin(x + z)
    """
    def system(t, x):
        return [
            sigma*(x[1]-x[0]),
            x[0]*(rho-x[2])-x[1],
            x[0]*x[1] - beta*x[2] + gamma*x[1]*np.sin(x[0]+x[2])
        ]

    t_eval = np.linspace(0, t_end, n_samples)
    X_clean = solve_ivp(system, [0, t_end], x0,
                        t_eval=t_eval, dense_output=True).y.T
    X_noisy, X_dot = _add_noise_and_differentiate(
        X_clean, t_eval, noise_level, noise_seed)
    print("✓ Step 1: Generated synthetic Complex Lorenz data.")
    return X_noisy, X_dot, t_eval


# ── Duffing ───────────────────────────────────────────────────────────────────
def generate_duffing_data(
        delta=0.1, alpha=-1.0, beta=1.0, x0=[0.5, 0.5],
        t_end=50, n_samples=5000, noise_level=0.01, noise_seed=42):
    """
    x' = y
    y' = -delta*y - alpha*x - beta*x³
    """
    def system(t, x):
        return [x[1], -delta*x[1] - alpha*x[0] - beta*x[0]**3]

    t_eval = np.linspace(0, t_end, n_samples)
    X_clean = solve_ivp(system, [0, t_end], x0,
                        t_eval=t_eval, dense_output=True).y.T
    X_noisy, X_dot = _add_noise_and_differentiate(
        X_clean, t_eval, noise_level, noise_seed)
    print("✓ Step 1: Generated synthetic Duffing Oscillator data.")
    return X_noisy, X_dot, t_eval


# ── Michaelis-Menten ──────────────────────────────────────────────────────────
def generate_michaelis_menten_data(
        vmax=1.5, km=0.3, x0=[1.0], t_end=10, n_samples=5000,
        noise_level=0.01, noise_seed=42):
    """
    x' = -vmax*x / (km + x)
    """
    def system(t, x): return [-vmax*x[0] / (km + x[0])]

    t_eval = np.linspace(0, t_end, n_samples)
    X_clean = solve_ivp(system, [0, t_end], x0,
                        t_eval=t_eval, dense_output=True).y.T
    X_noisy, X_dot = _add_noise_and_differentiate(
        X_clean, t_eval, noise_level, noise_seed)
    print("✓ Step 1: Generated synthetic Michaelis-Menten data.")
    return X_noisy, X_dot, t_eval


# ── Modulated Oscillator ──────────────────────────────────────────────────────
def generate_modulated_oscillator_data(
        b=0.25, k=5.0, x0=[1.0, 0.0], t_end=10, n_samples=5000,
        noise_level=0.01, noise_seed=42):
    """
    x' = y
    y' = -b*y*cos(x) - k*x
    """
    def system(t, x): return [x[1], -b*x[1]*np.cos(x[0]) - k*x[0]]

    t_eval = np.linspace(0, t_end, n_samples)
    X_clean = solve_ivp(system, [0, t_end], x0,
                        t_eval=t_eval, dense_output=True).y.T
    X_noisy, X_dot = _add_noise_and_differentiate(
        X_clean, t_eval, noise_level, noise_seed)
    print("✓ Step 1: Generated synthetic Modulated Oscillator data.")
    return X_noisy, X_dot, t_eval


# ── Exponential System ────────────────────────────────────────────────────────
def generate_exponential_system_data(
        a=0.5, b=0.5, x0=[0.5, 0.5], t_end=10, n_samples=5000,
        noise_level=0.01, noise_seed=42):
    """
    dx/dt = -a*x - exp(-y)
    dy/dt = -b*y
    """
    def system(t, x): return [-a*x[0] - np.exp(-x[1]), -b*x[1]]

    t_eval = np.linspace(0, t_end, n_samples)
    X_clean = solve_ivp(system, [0, t_end], x0,
                        t_eval=t_eval, dense_output=True).y.T
    X_noisy, X_dot = _add_noise_and_differentiate(
        X_clean, t_eval, noise_level, noise_seed)
    print("✓ Step 1: Generated synthetic Exponential System data.")
    return X_noisy, X_dot, t_eval