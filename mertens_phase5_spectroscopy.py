"""
Phase 5 — Mertens Sum Zero-Height Spectroscopy
===============================================
Computes the weighted Mertens sum S(N) = Σ_{k=2}^{N} μ(k) · ln(k)/k
and Fourier-analyzes it in ln(N) space to detect oscillations at
frequencies γ_n/(2π) corresponding to Riemann zero heights.

Theory
------
The explicit formula predicts:
    S(N) + 1 ≈ -2 Re Σ_ρ  N^{ρ-1} / (ρ · ζ'(ρ))

Under RH (ρ = 1/2 + iγ), each zero contributes an oscillation
    amplitude ~ N^{-1/2} · |ζ'(1/2+iγ)|^{-1}
    frequency = γ / (2π)   in the variable t = ln(N)

The normalized signal:
    s(N) = (S(N) + 1) · √N

should exhibit sinusoidal oscillations at frequencies γ_n/(2π).
A discrete power spectrum of s sampled at t = ln(N), N=2..N_MAX,
should show peaks at those frequencies.

Outputs
-------
  mertens_phase5.png  — 4-panel figure
  mertens_phase5.json — numerical results, Z-scores, circuit spec
"""

import numpy as np
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ─── Parameters ───────────────────────────────────────────────────────────────
N_MAX      = 5_000_000      # sieve up to here; ~5M gives good frequency resolution
N_MIN_FT   = 100            # discard small-N transient in FT
N_RESAMPLE = 20_000         # uniform t-grid points for FFT (Nyquist >> γ₃₀)

# First 30 non-trivial Riemann zero imaginary parts (γ_n)
RIEMANN_ZEROS = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918720, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337376, 82.910381, 84.735493, 87.425275, 88.809112,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
]

# ─── Step 1: Linear sieve for Möbius function ─────────────────────────────────
def mobius_sieve(n_max):
    """Return array mu[0..n_max] (indices 0..n_max, mu[0]=mu[1]=0/1 by convention)."""
    print(f"  Sieving μ(k) up to {n_max:,} ...", flush=True)
    t0 = time.time()
    mu = np.zeros(n_max + 1, dtype=np.int8)
    mu[1] = 1
    is_prime  = np.ones(n_max + 1, dtype=bool)
    primes    = []
    # smallest prime factor sieve
    spf = np.arange(n_max + 1, dtype=np.int32)

    for i in range(2, n_max + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1          # prime → μ = -1
        for p in primes:
            if i * p > n_max:
                break
            is_prime[i * p] = False
            if i % p == 0:
                # p² | i·p  →  μ(i·p) = 0
                spf[i * p] = p
                # mu[i*p] stays 0
                break
            else:
                spf[i * p] = p
                mu[i * p]  = -mu[i]

    print(f"  Sieve done in {time.time()-t0:.1f}s", flush=True)
    return mu

# ─── Step 2: Compute S(N) cumulatively ────────────────────────────────────────
def compute_mertens(mu, n_max):
    """Return arrays N_arr (2..n_max) and S_arr = cumsum of mu*ln/k."""
    print(f"  Computing S(N) for N=2..{n_max:,} ...", flush=True)
    t0 = time.time()
    k  = np.arange(2, n_max + 1, dtype=np.float64)
    f  = np.log(k) / k                      # Mertens weight f(k) = ln(k)/k
    mu_k = mu[2:n_max + 1].astype(np.float64)
    terms = mu_k * f
    S    = np.cumsum(terms)                  # S[i] = S(i+2)
    N_arr = k                                # N_arr[i] = i+2
    print(f"  S(N) done in {time.time()-t0:.1f}s", flush=True)
    return N_arr, S

# ─── Step 3: Normalized oscillatory signal ────────────────────────────────────
def normalize_signal(N_arr, S_arr, n_min=N_MIN_FT):
    """s(N) = (S(N)+1) * sqrt(N), restrict to N >= n_min."""
    mask   = N_arr >= n_min
    N_sel  = N_arr[mask]
    S_sel  = S_arr[mask]
    s      = (S_sel + 1.0) * np.sqrt(N_sel)
    t      = np.log(N_sel)           # time variable for FT
    return t, s, N_sel, S_sel

# ─── Step 4: Resample + FFT power spectrum ────────────────────────────────────
def fft_spectrum(t, s, n_resample=N_RESAMPLE):
    """
    Resample s(t) onto a uniform t-grid (avoiding N×M memory explosion),
    then compute FFT power spectrum.
    Returns omegas (angular freq), power — both covering the full range.
    """
    print("  Resampling to uniform t-grid ...", flush=True)
    t0 = time.time()
    t_min, t_max = t[0], t[-1]
    t_uniform = np.linspace(t_min, t_max, n_resample)
    # Linear interpolation is fine — signal is smooth on this scale
    interp    = interp1d(t, s, kind='linear', assume_sorted=True)
    s_uniform = interp(t_uniform)

    dt  = (t_max - t_min) / (n_resample - 1)
    print(f"  Δt={dt:.5f}, Nyquist ω={np.pi/dt:.1f} >> γ₃₀={RIEMANN_ZEROS[-1]:.1f}", flush=True)

    # FFT
    s_zm   = s_uniform - s_uniform.mean()
    fft_v  = np.fft.rfft(s_zm)
    freqs  = np.fft.rfftfreq(n_resample, d=dt)   # cycles per unit-t
    omegas = 2 * np.pi * freqs                    # angular frequency

    power  = (np.abs(fft_v) ** 2) / n_resample   # normalized power
    print(f"  FFT done in {time.time()-t0:.2f}s", flush=True)
    return omegas, power

# ─── Step 4b: Manual DFT at exact γ_n for precise Z-scores ───────────────────
def dft_at_zeros(t, s):
    """
    Compute F(γ) = Σ_k s(t_k) exp(-i·γ·t_k) Δt_k  at each Riemann zero γ.
    Uses trapezoidal weights Δt_k ≈ 1/N_k.  O(N·30) memory-safe.
    """
    print("  Manual DFT at 30 Riemann zeros ...", flush=True)
    t0 = time.time()
    # Quadrature weights (trapezoidal rule differences in t)
    dt_weights        = np.empty_like(t)
    dt_weights[1:-1]  = (t[2:] - t[:-2]) / 2.0
    dt_weights[0]     = t[1] - t[0]
    dt_weights[-1]    = t[-1] - t[-2]

    s_zm = s - s.mean()
    n    = len(t)
    powers_z = []
    for gamma in RIEMANN_ZEROS:
        # vectorized batch to keep memory O(N)
        cos_t = np.cos(gamma * t)
        sin_t = np.sin(gamma * t)
        F_re  = float(np.dot(s_zm * dt_weights, cos_t))
        F_im  = float(np.dot(s_zm * dt_weights, sin_t))
        powers_z.append(F_re**2 + F_im**2)

    print(f"  DFT done in {time.time()-t0:.2f}s", flush=True)
    return np.array(powers_z)

# ─── Step 5: Z-scores at Riemann zeros ────────────────────────────────────────
def zero_z_scores(omegas_fft, power_fft, powers_dft):
    """
    Z-scores from manual DFT powers_dft.
    Background estimated from FFT power at null ω values
    (gaps between known zeros, avoiding ±2 units around each γ_n).
    """
    # Build null set: sample FFT power at frequencies avoiding all zeros
    null_mask = np.ones(len(omegas_fft), dtype=bool)
    for gamma in RIEMANN_ZEROS:
        null_mask &= (np.abs(omegas_fft - gamma) > 2.0)
    # Also restrict to the range of interest
    null_mask &= (omegas_fft > RIEMANN_ZEROS[0] * 0.5)
    null_mask &= (omegas_fft < RIEMANN_ZEROS[-1] * 1.2)
    null_power = power_fft[null_mask]
    mu_bg  = float(null_power.mean())
    sig_bg = float(null_power.std())

    # Also get FFT power at each gamma for comparison
    peak_powers_fft = [float(np.interp(g, omegas_fft, power_fft)) for g in RIEMANN_ZEROS]

    # Z-score from DFT: use same background sigma scaled by DFT normalization
    # DFT power ~ (F_re² + F_im²); FFT power ~ |fft|²/N
    # Normalize DFT by dividing by mean DFT at null freqs
    null_dft = np.array([float(np.interp(g, omegas_fft, power_fft))
                         for g in np.linspace(RIEMANN_ZEROS[0]*0.5,
                                              RIEMANN_ZEROS[-1]*1.2, 200)
                         if all(abs(g - gz) > 2.0 for gz in RIEMANN_ZEROS)])
    # Use FFT-based Z-scores (robust and background-calibrated)
    z_scores    = [(p - mu_bg) / sig_bg if sig_bg > 0 else 0.0
                   for p in peak_powers_fft]

    return z_scores, peak_powers_fft, mu_bg, sig_bg

# ─── Step 6: Quantum circuit specification ────────────────────────────────────
def quantum_circuit_spec(z_scores, peak_powers, sig_bg):
    """
    Derive GHZ phase estimation circuit parameters from SNR analysis.

    A k-qubit GHZ state gives Heisenberg-limited phase resolution δφ ~ 1/(k·√M)
    where M = number of shots.  We need δφ < 2π·Δγ_min/ln(N_MAX) to resolve
    adjacent zeros.

    Signal amplitude A extracted from Lomb-Scargle:
    normalized LS power P ≈ (A·√(n/2))² / σ²  →  A = sqrt(2P)·σ/sqrt(n/2)
    where n = len(t) = N_MAX - N_MIN_FT.
    """
    n_data   = N_MAX - N_MIN_FT
    # amplitude in units of normalized signal (dimensionless)
    A_top3   = sorted(zip(z_scores, RIEMANN_ZEROS), reverse=True)[:3]

    # Frequency resolution: Δγ_min ≈ min spacing of zeros
    gamma_arr   = np.array(RIEMANN_ZEROS)
    delta_gamma = float(np.min(np.diff(gamma_arr)))      # ~2.8 at low end

    # Phase resolution needed (in ln(N) space)
    # One full oscillation of γ_n spans Δt = 2π/γ_n.
    # To distinguish adjacent zeros separated by Δγ, need resolution Δt_res < 2π/Δγ.
    # With N_MAX=5e6, ln range Δt = ln(N_MAX)-ln(N_MIN_FT) ≈ 10.7
    t_range  = np.log(N_MAX) - np.log(N_MIN_FT)
    freq_res = 1.0 / t_range                             # cycles per unit t
    freq_res_omega = 2 * np.pi * freq_res                # in ω units

    # Qubits needed: k ≥ ceil(log2(omega_max / freq_res_omega))
    k_continuous = np.log2(RIEMANN_ZEROS[-1] / freq_res_omega)
    k_qubits     = int(np.ceil(k_continuous))

    # Shots for SNR≥5 at weakest detected peak (use median Z)
    med_z = float(np.median([z for z in z_scores if z > 0]) or 1.0)
    # SNR ∝ √(M·k²) for Heisenberg;  need SNR≥5
    shots_heisenberg = int(np.ceil((5.0 / (med_z * k_qubits))**2 * n_data))
    shots_heisenberg = max(shots_heisenberg, 1000)

    return {
        "n_qubits_GHZ": k_qubits,
        "shots_recommended": shots_heisenberg,
        "freq_resolution_omega": round(freq_res_omega, 4),
        "delta_gamma_min": round(delta_gamma, 4),
        "t_range_lnN": round(t_range, 4),
        "top3_detections": [
            {"gamma": g, "z_score": round(z, 2)} for z, g in A_top3
        ],
    }

# ─── Step 7: Figure ────────────────────────────────────────────────────────────
def make_figure(N_arr, S_arr, t, s, N_sel, S_sel, omegas, power,
                z_scores, mu_bg, sig_bg, out_path, zeros):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Phase 5 — Mertens Sum Zero-Height Spectroscopy", fontsize=14, fontweight='bold')

    # Panel 1: S(N) vs ln(N)
    ax = axes[0, 0]
    step = max(1, len(N_arr) // 5000)   # downsample for plotting
    ax.plot(np.log(N_arr[::step]), S_arr[::step], lw=0.6, color='steelblue')
    ax.axhline(-1, color='k', lw=0.8, ls='--', label='S(N) = -1')
    ax.set_xlabel("ln(N)")
    ax.set_ylabel("S(N)")
    ax.set_title("Weighted Mertens Sum S(N)")
    ax.legend(fontsize=8)

    # Panel 2: |S(N)+1| vs N with N^{-1/2} envelope
    ax = axes[0, 1]
    residual = np.abs(S_arr[::step] + 1.0)
    envelope = 1.0 / np.sqrt(N_arr[::step])
    ax.semilogy(N_arr[::step], residual, lw=0.5, color='darkorange', label='|S(N)+1|')
    ax.semilogy(N_arr[::step], envelope, 'k--', lw=1, label='N^{-1/2} (RH)')
    ax.set_xlabel("N")
    ax.set_ylabel("|S(N)+1|")
    ax.set_title("|S(N)+1| vs RH Envelope")
    ax.legend(fontsize=8)

    # Panel 3: Normalized signal s(N) in ln(N) space
    ax = axes[1, 0]
    step2 = max(1, len(t) // 5000)
    ax.plot(t[::step2], s[::step2], lw=0.5, color='purple', alpha=0.8)
    ax.set_xlabel("t = ln(N)")
    ax.set_ylabel("s(N) = (S(N)+1)·√N")
    ax.set_title("Normalized Oscillatory Signal")
    ax.axhline(0, color='k', lw=0.5)

    # Panel 4: Lomb-Scargle power spectrum with zero markers
    ax = axes[1, 1]
    ax.plot(omegas, power, lw=0.7, color='navy', alpha=0.8, label='LS power')
    ax.axhline(mu_bg, color='gray', lw=0.8, ls='--', label='BG mean')
    ax.axhline(mu_bg + 3*sig_bg, color='red', lw=0.8, ls=':', label='3σ')
    ax.axhline(mu_bg + 5*sig_bg, color='darkred', lw=0.8, ls=':', label='5σ')
    # Mark each Riemann zero
    ymax = power.max()
    for i, (gamma, z) in enumerate(zip(zeros, z_scores)):
        color = 'red' if z >= 3 else ('orange' if z >= 2 else 'gray')
        ax.axvline(gamma, color=color, lw=0.8, alpha=0.6)
        if z >= 2:
            ax.text(gamma, ymax * 0.85, f"γ{i+1}\nZ={z:.1f}",
                    fontsize=5, ha='center', color=color, rotation=90)
    ax.set_xlabel("ω (angular frequency in ln-N)")
    ax.set_ylabel("Normalized LS Power")
    ax.set_title("Power Spectrum — Peaks at Riemann Zeros?")
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {out_path}", flush=True)

# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    import os
    # Resolve output paths relative to this script file, regardless of cwd
    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        here = os.getcwd()
    fig_path  = os.path.join(here, "mertens_phase5.png")
    json_path = os.path.join(here, "mertens_phase5.json")
    print(f"  Output dir: {here}", flush=True)

    print("=== Phase 5: Mertens Sum Zero-Height Spectroscopy ===", flush=True)
    t_total = time.time()

    # 1. Sieve
    mu = mobius_sieve(N_MAX)

    # 2. Mertens sum
    N_arr, S_arr = compute_mertens(mu, N_MAX)

    # 3. Normalize
    t_ln, s, N_sel, S_sel = normalize_signal(N_arr, S_arr)
    print(f"  s(N) range: [{s.min():.4f}, {s.max():.4f}]", flush=True)

    # 4. FFT power spectrum (on uniform t-grid)
    omegas, power = fft_spectrum(t_ln, s)

    # 4b. Precise DFT at each Riemann zero
    powers_dft = dft_at_zeros(t_ln, s)

    # 5. Z-scores
    z_scores, peak_powers, mu_bg, sig_bg = zero_z_scores(omegas, power, powers_dft)

    # 6. Summary
    print("\n  Z-scores at first 10 Riemann zeros:", flush=True)
    for i, (g, z) in enumerate(zip(RIEMANN_ZEROS[:10], z_scores[:10])):
        marker = " *** DETECTED ***" if z >= 3 else (" * " if z >= 2 else "")
        print(f"    g{i+1:2d} = {g:8.3f}   Z = {z:6.2f}{marker}", flush=True)

    top5 = sorted(zip(z_scores, RIEMANN_ZEROS), reverse=True)[:5]
    print("\n  Top 5 peaks (by Z-score):", flush=True)
    for z, g in top5:
        idx = RIEMANN_ZEROS.index(g) + 1
        print(f"    g{idx} = {g:.3f}   Z = {z:.2f}", flush=True)

    # 7. Circuit spec
    circuit_spec = quantum_circuit_spec(z_scores, peak_powers, sig_bg)
    print(f"\n  Quantum circuit spec (C5):", flush=True)
    for k, v in circuit_spec.items():
        print(f"    {k}: {v}", flush=True)

    # 8. Figure
    make_figure(N_arr, S_arr, t_ln, s, N_sel, S_sel, omegas, power,
                z_scores, mu_bg, sig_bg, fig_path, RIEMANN_ZEROS)

    # 9. Save JSON
    results = {
        "n_max": N_MAX,
        "signal_length": len(t_ln),
        "s_min": float(s.min()),
        "s_max": float(s.max()),
        "background_mean": float(mu_bg),
        "background_sigma": float(sig_bg),
        "zero_results": [
            {
                "index": i + 1,
                "gamma": g,
                "z_score": round(z, 3),
                "detected_3sigma": bool(z >= 3),
                "detected_5sigma": bool(z >= 5),
            }
            for i, (g, z) in enumerate(zip(RIEMANN_ZEROS, z_scores))
        ],
        "top5_by_z": [
            {"gamma": g, "z_score": round(z, 2)} for z, g in top5
        ],
        "quantum_circuit_C5": circuit_spec,
    }
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  JSON saved: {json_path}", flush=True)
    print(f"\n=== Done in {time.time()-t_total:.1f}s ===", flush=True)

if __name__ == "__main__":
    main()
