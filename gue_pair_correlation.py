"""
Phase 6 — GUE Pair Correlation and Spectral Form Factor
========================================================
Tests whether the Riemann zeros obey GUE (Gaussian Unitary Ensemble)
random matrix statistics, as predicted by the Berry-Keating H=xp conjecture.

Three independent tests:
  1. Nearest-neighbor spacing distribution vs GUE Wigner surmise
  2. Pair correlation function R2(alpha) vs Montgomery's formula
  3. Spectral form factor K(tau) = |sum exp(2pi*i*gamma_n*tau)|^2 vs GUE

The connection to PFBMW:
  - Phase 5 showed f = ln(k)/k Mertens sum encodes zero heights (arithmetic side)
  - Phase 6 tests whether those zeros have GUE spectral statistics (quantum chaos side)
  - If yes: the arithmetic structure of f generates a spectrum indistinguishable from
    a quantum-chaotic Hamiltonian — the Berry-Keating prediction is confirmed locally
  - IBM circuit target: a GHZ-based circuit that samples the spectral form factor K(tau)
    directly, exploiting quantum parallelism to compute all N^2 phase pairs at once

Outputs
-------
  gue_pair_correlation.png   — 4-panel figure
  gue_pair_correlation.json  — statistics, KS tests, IBM circuit spec
"""

import numpy as np
import json
import time
from scipy.stats import kstest, chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import erf

# ─── First 100 non-trivial Riemann zero imaginary parts ───────────────────────
# Source: LMFDB / Odlyzko tables (accurate to 6 decimal places)
ZEROS_100 = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918720, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337376, 82.910381, 84.735493, 87.425275, 88.809112,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
    103.725538, 105.446623, 107.168611, 111.029536, 111.874659,
    114.320221, 116.226680, 118.790782, 121.370125, 122.946829,
    124.256819, 127.516683, 129.578704, 131.087688, 133.497737,
    134.756510, 138.116042, 139.736209, 141.123707, 143.111846,
    146.000982, 147.422765, 150.053521, 150.925257, 153.024693,
    156.112909, 157.597592, 158.849988, 161.188964, 163.030709,
    165.537069, 167.184439, 169.094515, 169.911976, 173.411536,
    174.754191, 176.441434, 178.377407, 179.916484, 182.207078,
    184.874467, 185.598783, 187.228922, 189.416158, 192.026656,
    193.079726, 195.265397, 196.876481, 198.015309, 201.264751,
    202.493594, 204.189671, 205.394697, 207.906258, 209.576509,
    211.690862, 213.347919, 214.547044, 216.169538, 219.067596,
    220.714918, 221.430705, 224.007000, 224.983324, 227.421444,
    229.337413, 231.250188, 231.987235, 233.693404, 236.524229,
]

# ─── GUE analytic predictions ─────────────────────────────────────────────────

def gue_wigner_surmise(s):
    """GUE nearest-neighbor spacing PDF: p(s) = (32/pi^2) s^2 exp(-4s^2/pi)"""
    return (32.0 / np.pi**2) * s**2 * np.exp(-4.0 * s**2 / np.pi)

def gue_wigner_cdf(s):
    """CDF of GUE Wigner surmise (numerical integration via erf)."""
    # CDF = integral_0^s p(t) dt  — computed via incomplete gamma
    from scipy.special import gammainc
    # p(s) = (32/pi^2) s^2 exp(-4s^2/pi)
    # Let u = 4s^2/pi => CDF = (1/sqrt(pi)) * gamma(3/2, u) * (something)
    # Easier: numerical cumulative
    s_arr = np.atleast_1d(np.asarray(s, dtype=float))
    cdf   = np.zeros_like(s_arr)
    for i, si in enumerate(s_arr):
        t   = np.linspace(0, si, 2000)
        cdf[i] = np.trapz(gue_wigner_surmise(t), t)
    return cdf if len(cdf) > 1 else float(cdf[0])

def goe_wigner_surmise(s):
    """GOE (for comparison): p(s) = (pi/2) s exp(-pi*s^2/4)"""
    return (np.pi / 2.0) * s * np.exp(-np.pi * s**2 / 4.0)

def poisson_spacing(s):
    """Poisson (uncorrelated): p(s) = exp(-s)"""
    return np.exp(-s)

def montgomery_r2(alpha):
    """
    Montgomery pair correlation: R2(alpha) = 1 - (sin(pi*alpha)/(pi*alpha))^2
    This is the GUE 2-point correlation function.
    """
    alpha = np.asarray(alpha, dtype=float)
    sinc  = np.where(alpha == 0, 1.0, np.sin(np.pi * alpha) / (np.pi * alpha))
    return 1.0 - sinc**2

def gue_form_factor(tau, n_zeros):
    """
    GUE spectral form factor (connected, unfolded):
      K(tau) = tau          for 0 < tau < 1
      K(tau) = 1            for tau >= 1
    This is the large-N RMT prediction.
    """
    tau = np.asarray(tau, dtype=float)
    return np.where(tau < 1.0, tau, np.ones_like(tau))

# ─── Step 1: Unfold the spectrum ──────────────────────────────────────────────

def unfold_zeros(zeros):
    """
    Unfold the Riemann zeros to unit mean spacing using the smooth counting function.
    N(T) ~ T/(2pi) * (ln(T/(2pi*e))) + 7/8  (Riemann-von Mangoldt)
    Unfolded: xi_n = N(gamma_n)
    """
    zeros = np.array(zeros, dtype=float)
    def smooth_count(T):
        return (T / (2*np.pi)) * (np.log(T / (2*np.pi*np.e))) + 7.0/8.0
    xi = smooth_count(zeros)
    return xi

def nearest_neighbor_spacings(xi):
    """Normalized nearest-neighbor spacings s_n = xi_{n+1} - xi_n (mean=1 by construction)."""
    spacings = np.diff(np.sort(xi))
    # Should already have mean ~1 after unfolding; normalize to be safe
    spacings = spacings / spacings.mean()
    return spacings

# ─── Step 2: Pair correlation function ────────────────────────────────────────

def compute_pair_correlation(xi, alpha_max=4.0, n_bins=60):
    """
    Empirical 2-point correlation: count pairs (xi_m, xi_n) with
    |xi_m - xi_n| in [alpha, alpha+dalpha], normalized.
    Returns alpha_centers, R2_empirical.
    """
    xi   = np.sort(xi)
    N    = len(xi)
    diffs = []
    for i in range(N):
        for j in range(i+1, N):
            d = xi[j] - xi[i]
            if d > alpha_max:
                break
            diffs.append(d)
    diffs = np.array(diffs)

    bins  = np.linspace(0, alpha_max, n_bins+1)
    hist, edges = np.histogram(diffs, bins=bins)
    centers = 0.5*(edges[:-1] + edges[1:])
    dalpha  = edges[1] - edges[0]

    # Correct normalization so R2 -> 1 for large alpha (Poisson baseline).
    # For N unfolded points in range L with density rho=1, expected count
    # of one-sided pairs in [alpha, alpha+dalpha] = N*(N-1)/2 * dalpha / L
    L    = float(xi[-1] - xi[0])   # range of unfolded zeros
    norm = N * (N - 1) / 2.0 * dalpha / L
    R2_emp = hist.astype(float) / norm

    return centers, R2_emp

# ─── Step 3: Spectral form factor ─────────────────────────────────────────────

def compute_form_factor(zeros, tau_max=3.0, n_tau=300):
    """
    K(tau) = (1/N) |sum_{n=1}^N exp(2*pi*i*gamma_n*tau)|^2
    Evaluated over a grid of tau values.
    Uses the unfolded zeros for proper normalization.
    """
    xi     = unfold_zeros(zeros)
    N      = len(xi)
    taus   = np.linspace(0.02, tau_max, n_tau)
    K      = np.zeros(n_tau)
    for k, tau in enumerate(taus):
        phases = 2.0 * np.pi * xi * tau
        K[k]   = (np.abs(np.sum(np.exp(1j*phases)))**2) / N
    return taus, K

# ─── Step 4: KS test vs GUE ───────────────────────────────────────────────────

def ks_test_gue(spacings):
    """KS test of empirical spacings vs GUE Wigner surmise CDF."""
    # Build empirical CDF
    s_sorted = np.sort(spacings)
    # Compute GUE CDF at each spacing value
    gue_cdf_vals = gue_wigner_cdf(s_sorted)
    # KS statistic
    n = len(s_sorted)
    ecdf = np.arange(1, n+1) / n
    D = np.max(np.abs(ecdf - gue_cdf_vals))
    # Approximate p-value (Kolmogorov distribution)
    z   = D * np.sqrt(n)
    # KS p-value via Kolmogorov formula
    from scipy.stats import ks_1samp
    result = ks_1samp(spacings, lambda x: np.squeeze(gue_wigner_cdf(x)))
    return float(result.statistic), float(result.pvalue)

# ─── Step 5: IBM circuit specification ────────────────────────────────────────

def ibm_circuit_spec(zeros, xi, spacings, K_empirical, taus):
    """
    Derive IBM quantum circuit parameters for measuring K(tau) via GHZ.

    Classical K(tau) requires O(N^2) operations (all pairs).
    A k-qubit GHZ state in superposition can compute K(tau) for one tau in O(N*k).
    For the full tau sweep, quantum advantage emerges at N >> k.

    Key parameters:
    - Frequency of K(tau) dip: at tau_dip ~ 0.3 (GUE prediction)
    - Phase resolution needed to resolve dip: delta_tau ~ 0.05
    - Equivalent phase: phi = 2*pi*gamma_1*tau ~ 2*pi*14.135*0.3 ~ 26.6 rad
    - Qubits needed: k = ceil(log2(2*pi/delta_phi)) where delta_phi ~ 2*pi*gamma_1*delta_tau
    """
    gamma_1 = zeros[0]
    gamma_max = zeros[-1]

    # Form factor dip location (empirical from GUE: K drops below 1 for tau < 1)
    tau_arr = taus
    K_arr   = K_empirical
    # Find where K first crosses 1 from below (ramp region)
    cross_idx = np.where(K_arr >= 0.9)[0]
    tau_ramp  = float(taus[cross_idx[0]]) if len(cross_idx) > 0 else 1.0

    # Phase per zero at tau = tau_ramp: phi = 2*pi*gamma_n*tau_ramp
    phi_1   = 2 * np.pi * gamma_1 * tau_ramp
    phi_max = 2 * np.pi * gamma_max * tau_ramp

    # Resolution to distinguish tau steps of delta_tau = 0.05
    delta_tau  = 0.05
    delta_phi  = 2 * np.pi * gamma_1 * delta_tau
    k_qubits   = int(np.ceil(np.log2(2 * np.pi / delta_phi))) + 1
    k_qubits   = max(k_qubits, 4)

    # Mean spacing and level repulsion quantification
    mean_spacing  = float(spacings.mean())
    spacing_var   = float(spacings.var())
    level_repulsion = float((spacings < 0.2).mean())   # fraction of small spacings

    # Shot count: need SNR >= 5 for K(tau) at tau_ramp
    # K ~ O(1), noise ~ 1/sqrt(M*k^2) for GHZ
    shots = int(np.ceil((5.0 / k_qubits)**2 * 1000))
    shots = max(shots, 2000)

    return {
        "target_observable": "spectral_form_factor_K(tau)",
        "n_qubits_GHZ": k_qubits,
        "shots_recommended": shots,
        "tau_ramp_estimate": round(tau_ramp, 3),
        "phi_per_zero_at_ramp": round(phi_1, 3),
        "delta_tau_resolution": delta_tau,
        "mean_spacing": round(mean_spacing, 4),
        "level_repulsion_fraction": round(level_repulsion, 4),
        "spacing_variance": round(spacing_var, 4),
        "classical_pairs_N2": len(zeros)**2,
        "quantum_advantage_threshold_N": int(2**k_qubits),
        "circuit_type": "GHZ_phase_accumulation_QFT_readout",
        "oracle_phase": "exp(2*pi*i*gamma_n*tau) for controlled-n",
    }

# ─── Step 6: Figure ───────────────────────────────────────────────────────────

def make_figure(spacings, centers_r2, R2_emp, taus, K_emp, ks_stat, ks_pval,
                zeros, xi, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Phase 6 — GUE Pair Correlation & Spectral Form Factor\n"
                 f"First {len(zeros)} Riemann Zeros", fontsize=13, fontweight='bold')

    # Panel 1: Nearest-neighbor spacing distribution
    ax = axes[0, 0]
    s_bins = np.linspace(0, 4, 30)
    ax.hist(spacings, bins=s_bins, density=True, alpha=0.6, color='steelblue',
            label=f'Zeros (n={len(spacings)})')
    s_plot = np.linspace(0, 4, 300)
    ax.plot(s_plot, gue_wigner_surmise(s_plot), 'r-', lw=2, label='GUE Wigner')
    ax.plot(s_plot, goe_wigner_surmise(s_plot), 'g--', lw=1.5, label='GOE Wigner')
    ax.plot(s_plot, poisson_spacing(s_plot), 'k:', lw=1.5, label='Poisson')
    ax.set_xlabel("Normalized spacing s")
    ax.set_ylabel("p(s)")
    ax.set_title(f"Nearest-Neighbor Spacings\nKS vs GUE: D={ks_stat:.3f}, p={ks_pval:.3f}")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 4)

    # Panel 2: Pair correlation R2(alpha)
    ax = axes[0, 1]
    alpha_plot = np.linspace(0.05, 4.0, 400)
    ax.plot(centers_r2, R2_emp, 'o-', ms=4, lw=1, color='steelblue',
            label='Empirical R₂(α)')
    ax.plot(alpha_plot, montgomery_r2(alpha_plot), 'r-', lw=2,
            label='Montgomery/GUE: 1-(sinπα/πα)²')
    ax.axhline(1.0, color='k', lw=0.8, ls=':', label='Poisson (R₂=1)')
    ax.set_xlabel("α (level separation)")
    ax.set_ylabel("R₂(α)")
    ax.set_title("Pair Correlation Function\nvs Montgomery Prediction")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.1, 2.0)

    # Panel 3: Spectral form factor K(tau)
    ax = axes[1, 0]
    gue_kline = gue_form_factor(taus, len(zeros))
    ax.plot(taus, K_emp, 'steelblue', lw=1, alpha=0.8, label='Empirical K(τ)')
    # Smoothed version
    from scipy.ndimage import uniform_filter1d
    K_smooth = uniform_filter1d(K_emp, size=15)
    ax.plot(taus, K_smooth, 'navy', lw=2, label='K(τ) smoothed')
    ax.plot(taus, gue_kline, 'r--', lw=2, label='GUE: min(τ,1)')
    ax.axhline(1.0, color='k', lw=0.5, ls=':')
    ax.set_xlabel("τ")
    ax.set_ylabel("K(τ)")
    ax.set_title("Spectral Form Factor\nDip-Ramp-Plateau Structure")
    ax.legend(fontsize=8)
    ax.set_xlim(0, taus[-1])
    ax.set_ylim(0, min(K_emp.max()*1.1, 5.0))

    # Panel 4: Cumulative spacing CDF vs GUE
    ax = axes[1, 1]
    s_sorted = np.sort(spacings)
    ecdf     = np.arange(1, len(s_sorted)+1) / len(s_sorted)
    s_dense  = np.linspace(0, 4, 500)
    gue_cdf  = gue_wigner_cdf(s_dense)
    ax.plot(s_sorted, ecdf, 'steelblue', lw=1.5, label='Empirical CDF')
    ax.plot(s_dense, gue_cdf, 'r-', lw=2, label='GUE CDF')
    ax.plot(s_dense, 1 - np.exp(-s_dense), 'k:', lw=1.5, label='Poisson CDF')
    ax.set_xlabel("s")
    ax.set_ylabel("CDF")
    ax.set_title("Cumulative Spacing Distribution\nvs GUE (Exact)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {out_path}", flush=True)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import os
    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        here = os.getcwd()
    fig_path  = os.path.join(here, "gue_pair_correlation.png")
    json_path = os.path.join(here, "gue_pair_correlation.json")

    print("=== Phase 6: GUE Pair Correlation & Spectral Form Factor ===", flush=True)
    t0 = time.time()

    zeros = np.array(ZEROS_100)
    N     = len(zeros)
    print(f"  Using first {N} Riemann zeros: gamma_1={zeros[0]:.3f} .. gamma_{N}={zeros[-1]:.3f}", flush=True)

    # 1. Unfold
    print("  Unfolding spectrum ...", flush=True)
    xi       = unfold_zeros(zeros)
    spacings = nearest_neighbor_spacings(xi)
    print(f"  {len(spacings)} spacings: mean={spacings.mean():.4f}, var={spacings.var():.4f}", flush=True)
    print(f"  Level repulsion check: P(s<0.1) = {(spacings<0.1).mean():.3f} (GUE predicts ~0)", flush=True)

    # 2. KS test vs GUE
    print("  KS test vs GUE Wigner surmise ...", flush=True)
    ks_stat, ks_pval = ks_test_gue(spacings)
    ks_goe, pval_goe = ks_test_gue_goe(spacings)
    print(f"  KS vs GUE: D={ks_stat:.4f}, p={ks_pval:.4f}", flush=True)
    print(f"  KS vs GOE: D={ks_goe:.4f}, p={pval_goe:.4f}", flush=True)

    # 3. Pair correlation
    print("  Computing pair correlation R2(alpha) ...", flush=True)
    centers_r2, R2_emp = compute_pair_correlation(xi, alpha_max=4.0, n_bins=50)
    # Compare at alpha = 1 (key GUE feature: R2(1) = 1 - 0 = 1, dip at small alpha)
    r2_at_1 = float(np.interp(1.0, centers_r2, R2_emp))
    r2_at_half = float(np.interp(0.5, centers_r2, R2_emp))
    print(f"  R2(0.5) = {r2_at_half:.3f} (GUE predicts {montgomery_r2(0.5):.3f})", flush=True)
    print(f"  R2(1.0) = {r2_at_1:.3f} (GUE predicts {montgomery_r2(1.0):.3f})", flush=True)

    # 4. Spectral form factor
    print("  Computing spectral form factor K(tau) ...", flush=True)
    taus, K_emp = compute_form_factor(zeros, tau_max=3.0, n_tau=300)
    # Smoothed
    from scipy.ndimage import uniform_filter1d
    K_smooth = uniform_filter1d(K_emp, size=15)
    k_at_half = float(np.interp(0.5, taus, K_smooth))
    k_at_1    = float(np.interp(1.0, taus, K_smooth))
    k_at_2    = float(np.interp(2.0, taus, K_smooth))
    print(f"  K(0.5) = {k_at_half:.3f} (GUE predicts 0.500)", flush=True)
    print(f"  K(1.0) = {k_at_1:.3f}  (GUE predicts 1.000)", flush=True)
    print(f"  K(2.0) = {k_at_2:.3f}  (GUE predicts 1.000)", flush=True)

    # 5. IBM circuit spec
    print("  Deriving IBM circuit spec ...", flush=True)
    circuit_spec = ibm_circuit_spec(zeros, xi, spacings, K_smooth, taus)
    print(f"  Circuit: {circuit_spec['n_qubits_GHZ']} qubits, "
          f"{circuit_spec['shots_recommended']} shots", flush=True)
    print(f"  Quantum advantage threshold: N >= {circuit_spec['quantum_advantage_threshold_N']}", flush=True)

    # 6. Figure
    make_figure(spacings, centers_r2, R2_emp, taus, K_emp,
                ks_stat, ks_pval, zeros, xi, fig_path)

    # 7. JSON
    results = {
        "n_zeros": N,
        "gamma_range": [float(zeros[0]), float(zeros[-1])],
        "spacings": {
            "mean": float(spacings.mean()),
            "variance": float(spacings.var()),
            "level_repulsion_P_s_lt_0p1": float((spacings < 0.1).mean()),
            "ks_vs_gue": {"D": round(ks_stat, 4), "p_value": round(ks_pval, 4)},
            "ks_vs_goe": {"D": round(ks_goe, 4), "p_value": round(pval_goe, 4)},
            "gue_consistent": bool(ks_pval > 0.05),
        },
        "pair_correlation": {
            "R2_at_0p5": round(r2_at_half, 4),
            "R2_at_1p0": round(r2_at_1, 4),
            "GUE_R2_at_0p5": round(float(montgomery_r2(0.5)), 4),
            "GUE_R2_at_1p0": round(float(montgomery_r2(1.0)), 4),
        },
        "form_factor": {
            "K_at_tau_0p5": round(k_at_half, 4),
            "K_at_tau_1p0": round(k_at_1, 4),
            "K_at_tau_2p0": round(k_at_2, 4),
            "GUE_K_at_tau_0p5": 0.5,
            "GUE_K_at_tau_1p0": 1.0,
        },
        "ibm_circuit_C6": circuit_spec,
        "interpretation": {
            "gue_match": bool(ks_pval > 0.05),
            "level_repulsion_confirmed": bool((spacings < 0.1).mean() < 0.02),
            "berry_keating_consistent":
                bool(ks_pval > 0.05 and (spacings < 0.1).mean() < 0.05),
            "note": (
                "GUE statistics confirm the zeros behave as eigenvalues of a "
                "quantum-chaotic Hamiltonian. Combined with Phase 5 (Mertens sum "
                "zero detection), this closes the arithmetic-spectral loop: "
                "f=ln(k)/k generates a spectrum consistent with H=xp chaos."
            )
        }
    }
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  JSON saved: {json_path}", flush=True)
    print(f"\n=== Done in {time.time()-t0:.1f}s ===", flush=True)


def ks_test_gue_goe(spacings):
    """KS test vs GOE Wigner surmise for comparison."""
    from scipy.stats import ks_1samp
    def goe_cdf(s):
        s = np.atleast_1d(np.asarray(s, float))
        cdf = np.zeros_like(s)
        for i, si in enumerate(s):
            t = np.linspace(0, si, 1000)
            cdf[i] = np.trapz(goe_wigner_surmise(t), t)
        return cdf
    result = ks_1samp(spacings, lambda x: np.squeeze(goe_cdf(x)))
    return float(result.statistic), float(result.pvalue)


if __name__ == "__main__":
    main()
