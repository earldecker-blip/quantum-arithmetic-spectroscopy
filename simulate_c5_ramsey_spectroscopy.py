"""
C5 — Mertens-Ramsey Zero Spectroscopy
======================================
GHZ-enhanced matched filter for Riemann zero detection.

Design
------
The Mertens signal s(N) = (S(N)+1)*sqrt(N) oscillates at gamma_n in ln(N).
To extract gamma_n with Heisenberg-limited precision, we use a k-qubit GHZ
circuit as a quantum reference oscillator:

  Circuit per (gamma_scan, N) pair:
    1. Prepare k-qubit GHZ: H, CNOT cascade
    2. Phase oracle: Rz(2 * k * gamma_scan * ln(N)) on each qubit
    3. Inverse GHZ: reverse CNOTs, H
    4. Measure qubit 0

  Expected output: <Z_0> = cos(2k * gamma_scan * ln(N)) + shot_noise

  Matched filter: C(gamma_scan) = sum_N s(N) * <Z_0(gamma_scan, N)>

  Peak at gamma_scan = gamma_n because:
    cos(2k*gamma_n*ln(N)) correlates maximally with s(N) at the zero frequency.

  Peak width: Deltag ~ pi / (k * T_range)
    k=1: pi/(1*5) = 0.63 rad
    k=5: pi/(5*5) = 0.13 rad   [5x narrower = Heisenberg advantage]

  Shot noise: each circuit contributes +/-1/sqrt(M) noise.
    After summing N_t terms: sigma_C ~ sqrt(N_t/M)
    SNR at gamma_1: SNR(k) = C_peak(k) / sigma_C ~ k * A_1 * sqrt(N_t*M)
    5x more signal from k-fold phase amplification: 5x SNR improvement.

IBM target
----------
  k=5 qubits, depth = 2k+2 = 12 gates (H, CX, Rz only)
  N_gamma = 100 scan points x N_t = 100 N values = 10,000 circuits
  Shots per circuit = 100; total = 1,000,000 shots
  Approx runtime on IBM: 15-30 minutes (batched)

  For quick IBM test: N_gamma=20, N_t=50, shots=200 => 200,000 shots (~5 min)

Outputs
-------
  c5_ramsey_spectroscopy.png    4-panel figure
  c5_ramsey_spectroscopy.json   results, Heisenberg gain, IBM spec
  c5_ibm_circuits_k5.qpy        serialized circuits for IBM (n_sample circuits)
"""

import numpy as np
import json, time, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    QISKIT_OK = True
except ImportError:
    QISKIT_OK = False

# ─── Known Riemann zeros ──────────────────────────────────────────────────────
ZEROS_30 = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918720, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337376, 82.910381, 84.735493, 87.425275, 88.809112,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
]

# ─── 1. Mertens signal ────────────────────────────────────────────────────────

def mobius_sieve(n_max):
    mu = np.zeros(n_max + 1, dtype=np.int8)
    mu[1] = 1
    is_prime = np.ones(n_max + 1, dtype=bool)
    primes = []
    for i in range(2, n_max + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > n_max: break
            is_prime[i * p] = False
            if i % p == 0: break
            else: mu[i * p] = -mu[i]
    return mu

def mertens_signal(n_max, n_min=100):
    mu   = mobius_sieve(n_max)
    k    = np.arange(2, n_max + 1, dtype=np.float64)
    S    = np.cumsum(mu[2:n_max + 1].astype(np.float64) * np.log(k) / k)
    N_full = k
    s_full = (S + 1.0) * np.sqrt(N_full)
    mask   = N_full >= n_min
    return np.log(N_full[mask]), s_full[mask]

# ─── 2. GHZ Ramsey circuit ────────────────────────────────────────────────────

def ghz_ramsey_qc(k, phase):
    """
    k-qubit GHZ circuit encoding a single phase value.
    phase = k * gamma_scan * ln(N)
    Output: <Z_0> = cos(2 * phase)
    """
    qc = QuantumCircuit(k, 1)
    qc.h(0)
    for i in range(k - 1): qc.cx(i, i + 1)
    for i in range(k):     qc.rz(2.0 * phase, i)
    for i in range(k - 2, -1, -1): qc.cx(i, i + 1)
    qc.h(0)
    qc.measure(0, 0)
    return qc

def analytic_response(k, gamma_scan, t_vals):
    """Exact cos(2k * gamma_scan * t) — analytic Aer equivalent."""
    return np.cos(2.0 * k * gamma_scan * t_vals)

def noisy_response(k, gamma_scan, t_vals, n_shots):
    """Analytic + binomial shot noise ~ 1/sqrt(shots)."""
    exact = analytic_response(k, gamma_scan, t_vals)
    noise = np.random.normal(0.0, 1.0 / np.sqrt(n_shots), len(t_vals))
    return np.clip(exact + noise, -1.0, 1.0)

# ─── 3. Matched filter C(gamma_scan) ─────────────────────────────────────────

def matched_filter(t_vals, s_vals, k, gamma_grid, n_shots, rng=None):
    """
    Power-spectrum matched filter:

      C(gamma) = |F_re(gamma)|^2 + |F_im(gamma)|^2

    where F_re = sum_N s(N)*cos(2k*gamma*ln N)*dt  (in-phase)
          F_im = sum_N s(N)*sin(2k*gamma*ln N)*dt  (quadrature)

    Using the complex (two-quadrature) response eliminates phase offset from
    t_start != 0 and gives peaks exactly at gamma_scan = gamma_n/k.
    The normalization is by signal energy so C_peak ~ 1 for a pure tone.

    For noisy shots, the IBM circuit measures cos only; shot noise is added
    to the in-phase component, quadrature estimated as analytic.
    """
    # Trapezoidal weights for non-equidistant grids
    w = np.empty(len(t_vals))
    w[1:-1] = (t_vals[2:] - t_vals[:-2]) / 2.0
    w[0]    = (t_vals[1]  - t_vals[0])   / 2.0
    w[-1]   = (t_vals[-1] - t_vals[-2])  / 2.0
    norm = np.sum(s_vals**2 * w)   # signal energy

    s_zm = s_vals - s_vals.mean()  # zero-mean to suppress DC

    C = np.zeros(len(gamma_grid))
    for gi, gamma in enumerate(gamma_grid):
        phase = 2.0 * k * gamma * t_vals
        cos_t = np.cos(phase)
        sin_t = np.sin(phase)
        if n_shots < 1e9:
            cos_t = cos_t + np.random.normal(0.0, 1.0 / np.sqrt(n_shots), len(t_vals))
        F_re = float(np.dot(s_zm * w, cos_t))
        F_im = float(np.dot(s_zm * w, sin_t))
        C[gi] = (F_re**2 + F_im**2) / norm**2

    return C

# ─── 4. Peak extraction ───────────────────────────────────────────────────────

def find_gamma_peaks(gamma_grid, C, zeros=ZEROS_30, tol=1.5):
    """Find peaks in power-spectrum C(gamma) and match to known zeros."""
    # Power spectrum is always >= 0; use median + std threshold
    height = np.median(C) + 1.5 * np.std(C)
    idx, _ = find_peaks(C, height=height, distance=3)

    matches = []
    for i in idx:
        g_est = float(gamma_grid[i])
        best  = np.argmin(np.abs(np.array(zeros) - g_est))
        delta = abs(zeros[best] - g_est)
        if delta < tol:
            matches.append({
                "gamma_estimated": round(g_est, 4),
                "gamma_known":     zeros[best],
                "error":           round(delta, 4),
                "zero_index":      best + 1,
                "C_value":         round(float(C[i]), 6),
            })
    return matches

def peak_width_hwhm(gamma_grid, C, gamma_target):
    """Half-width at half-max of the peak nearest gamma_target.
    Returns grid spacing if peak is flat or negative."""
    dg = float(gamma_grid[1] - gamma_grid[0])
    idx_peak = np.argmin(np.abs(gamma_grid - gamma_target))
    peak_val = C[idx_peak]
    if peak_val <= 0:
        return dg
    half_max = peak_val / 2.0
    # Search left
    left = idx_peak
    while left > 0 and C[left] > half_max:
        left -= 1
    # Search right
    right = idx_peak
    while right < len(C) - 1 and C[right] > half_max:
        right += 1
    hwhm = (gamma_grid[right] - gamma_grid[left]) / 2.0
    return float(hwhm) if hwhm > 0 else dg

# ─── 5. IBM circuits ──────────────────────────────────────────────────────────

def build_ibm_circuits(t_sample, k=5, gamma_ref=None, n_gamma=20):
    """
    Build IBM-ready circuits for a representative scan.
    For each (gamma_scan_i, N_j): one 12-gate circuit.
    Returns list of QuantumCircuits.
    """
    if not QISKIT_OK:
        return []
    if gamma_ref is None:
        gamma_ref = np.linspace(ZEROS_30[0]*0.7, ZEROS_30[4]*1.2, n_gamma)

    circuits = []
    for gamma in gamma_ref:
        for t in t_sample:
            phase = k * gamma * t
            qc = ghz_ramsey_qc(k, phase)
            qc.name = f"C5_k{k}_g{gamma:.3f}_t{t:.3f}"
            circuits.append(qc)
    return circuits

# ─── 6. Figure ────────────────────────────────────────────────────────────────

def make_figure(t_vals, s_vals,
                gamma_coarse, C_coarse, matches_coarse, zeros,
                gscan_by_k, C_fine_by_k, widths_by_k,
                T_range, out_path):
    """
    4-panel figure:
      TL: Mertens signal
      TR: k=1 coarse matched filter – zero detection overview
      BL: Fine scans for k=1,3,5 in γ_scan coordinates (peaks at γ₁/k)
          showing k× narrower peak = Heisenberg advantage
      BR: HWHM vs k with 1/k theory line
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("C5 — Mertens-Ramsey Zero Spectroscopy\n"
                 "Heisenberg Advantage: k-GHZ vs Classical", fontsize=13, fontweight='bold')
    colors = {1: 'steelblue', 3: 'darkorange', 5: 'crimson'}

    # ── Panel 1: Mertens signal ───────────────────────────────────────────────
    ax = axes[0, 0]
    step = max(1, len(t_vals) // 2000)
    ax.plot(t_vals[::step], s_vals[::step], 'k-', lw=0.6)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel("t = ln(N)")
    ax.set_ylabel("s(N) = (S(N)+1)·√N")
    ax.set_title("Mertens Signal (Phase Oracle Input)")
    ax.annotate(f"N_t = {len(t_vals)}\nT = {T_range:.2f}", xy=(0.03, 0.85),
                xycoords='axes fraction', fontsize=8)

    # ── Panel 2: k=1 coarse scan — zero detection ─────────────────────────────
    ax = axes[0, 1]
    ax.plot(gamma_coarse, C_coarse, 'steelblue', lw=1.0)
    ymax = C_coarse.max() * 1.05
    for i, g in enumerate(zeros[:10]):
        ax.axvline(g, color='black', lw=0.5, ls=':', alpha=0.6)
        ax.text(g, ymax * 0.88, f'γ{i+1}', fontsize=6, ha='center', rotation=90)
    ax.set_xlabel("γ_scan")
    ax.set_ylabel("C(γ_scan) = matched filter")
    ax.set_title(f"k=1 Overview: {len(matches_coarse)} Zeros Detected (γ ∈ [8,55])")
    ax.set_xlim(gamma_coarse[0], gamma_coarse[-1])

    # ── Panel 3: Fine scans for k=1,3,5 in γ_scan coordinates ────────────────
    # Peaks at γ₁/k; k=5 peak is 5× narrower than k=1 peak
    ax = axes[1, 0]
    for k in sorted(gscan_by_k.keys()):
        g = gscan_by_k[k]
        C = C_fine_by_k[k]
        # Normalise to peak = 1 for shape comparison
        Cpeak = C.max()
        if Cpeak > 0:
            Cn = C / Cpeak
        else:
            Cn = C
        w = widths_by_k.get(k, 0)
        label = f'k={k}  peak@{ZEROS_30[0]/k:.2f}  HWHM={w:.3f}'
        ax.plot(g, Cn, color=colors[k], lw=1.5, alpha=0.9, label=label)
        center = ZEROS_30[0] / k
        ax.axvline(center, color=colors[k], lw=0.6, ls='--', alpha=0.5)
    ax.set_xlabel("γ_scan  [peak at γ₁/k = " +
                  ", ".join(f"{ZEROS_30[0]/k:.2f}(k={k})" for k in sorted(gscan_by_k)) + "]")
    ax.set_ylabel("C(γ_scan) / C_peak  (normalised)")
    ax.set_title("Fine Scan: k× Narrower Peaks = Heisenberg Advantage\n"
                 "(each curve centred on its γ₁/k)")
    ax.legend(fontsize=8)

    # ── Panel 4: HWHM vs k ────────────────────────────────────────────────────
    ax = axes[1, 1]
    k_vals   = sorted(widths_by_k.keys())
    w_vals   = [widths_by_k[k] for k in k_vals]
    w1       = w_vals[0]
    w_theory = [np.pi / (k * T_range) for k in k_vals]
    ax.plot(k_vals, w_vals,   'o-', color='steelblue', ms=9, lw=2,  label='Measured HWHM')
    ax.plot(k_vals, w_theory, 'r--', lw=1.5,                        label='π/(k·T)  theory')
    ax.set_xlabel("Number of GHZ qubits k")
    ax.set_ylabel("Peak HWHM in γ_scan  (Δγ_scan)")
    ax.set_title("Heisenberg Scaling:  HWHM ∝ 1/k")
    ax.legend(fontsize=9)
    ax.set_xticks(k_vals)
    for k, w in zip(k_vals, w_vals):
        gain = w1 / w if w > 0 else float('nan')
        ax.annotate(f'{gain:.1f}×', (k, w), textcoords="offset points",
                    xytext=(6, 4), fontsize=9, color='crimson', fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {out_path}", flush=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        here = os.getcwd()
    fig_path  = os.path.join(here, "c5_ramsey_spectroscopy.png")
    json_path = os.path.join(here, "c5_ramsey_spectroscopy.json")

    print("=== C5: Mertens-Ramsey Zero Spectroscopy ===", flush=True)
    t_total = time.time()

    # ── Mertens signal ────────────────────────────────────────────────────────
    print("  Computing Mertens signal ...", flush=True)
    t_vals, s_vals = mertens_signal(n_max=500_000, n_min=100)
    T_range = t_vals[-1] - t_vals[0]
    print(f"  N_t={len(t_vals)}, t∈[{t_vals[0]:.2f},{t_vals[-1]:.2f}], T={T_range:.2f}", flush=True)
    print(f"  s range: [{s_vals.min():.3f}, {s_vals.max():.3f}]", flush=True)

    # Resample onto a UNIFORM t grid (ln N values are non-uniform: Δt ∝ 1/N)
    # A non-uniform grid would bias the matched-filter DFT and give spurious peaks.
    N_T = 2000
    t_s = np.linspace(t_vals[0], t_vals[-1], N_T)
    s_s = np.interp(t_s, t_vals, s_vals)
    print(f"  Resampled to {N_T} uniform t points, dt={t_s[1]-t_s[0]:.5f}", flush=True)

    K_VALS = [1, 3, 5]
    # Analytic (shot-noise free) for clean physics demonstration
    N_SHOTS = int(1e10)

    # ── Coarse scan k=1: zero detection overview ──────────────────────────────
    # k=1 reference cos(2γ_scan·t), peaks at γ_scan = γ_n
    gamma_coarse = np.linspace(8.0, 55.0, 500)
    print(f"  k=1 coarse scan ({len(gamma_coarse)} pts, γ∈[8,55]) ...", flush=True)
    t0 = time.time()
    C_coarse = matched_filter(t_s, s_s, k=1, gamma_grid=gamma_coarse, n_shots=N_SHOTS)
    print(f"    done in {time.time()-t0:.1f}s", flush=True)
    matches_coarse = find_gamma_peaks(gamma_coarse, C_coarse, zeros=ZEROS_30, tol=1.5)
    print(f"  Coarse: {len(matches_coarse)} zeros matched", flush=True)
    for m in matches_coarse[:6]:
        print(f"    γ_{m['zero_index']}={m['gamma_known']:.3f}  "
              f"est={m['gamma_estimated']:.3f}  err={m['error']:.4f}", flush=True)

    # ── Fine scan: Heisenberg advantage ───────────────────────────────────────
    # For k-qubit GHZ, matched filter cos(2k·γ_scan·t) peaks at γ_scan = γ₁/k
    # Peak HWHM in γ_scan space = π/(k·T_range)  → k× narrower for larger k
    # Use 300 pts over ±2/k around the expected peak at γ₁/k
    gscan_by_k  = {}
    C_fine_by_k = {}
    widths_by_k = {}

    print("\n  Fine scans around γ₁/k for each k:", flush=True)
    for k in K_VALS:
        center = ZEROS_30[0] / k          # expected peak position
        half   = max(3.0 / k, 0.5)        # scan half-width in γ_scan coords
        g = np.linspace(center - half, center + half, 500)
        print(f"  k={k}: γ_scan ∈ [{g[0]:.3f},{g[-1]:.3f}]  (peak expected at {center:.4f}) ...", flush=True)
        t0 = time.time()
        C = matched_filter(t_s, s_s, k, g, N_SHOTS)
        print(f"    done in {time.time()-t0:.1f}s", flush=True)
        gscan_by_k[k]  = g
        C_fine_by_k[k] = C
        w = peak_width_hwhm(g, C, center)
        widths_by_k[k] = w
        theory = np.pi / (k * T_range)
        gain   = widths_by_k[1] / w if w > 0 else float('nan')
        print(f"    HWHM={w:.4f}  theory={theory:.4f}  gain={gain:.2f}×", flush=True)

    # ── IBM circuits (representative sample) ─────────────────────────────────
    print("\n  Building IBM circuits (k=5) ...", flush=True)
    t_ibm  = t_s[::N_T // 50]  # 50 representative time points
    g_ibm  = np.linspace(ZEROS_30[0]/5 * 0.7, ZEROS_30[0]/5 * 1.3, 20)
    ibm_qcs = build_ibm_circuits(t_ibm, k=5, gamma_ref=g_ibm, n_gamma=20)
    print(f"  Built {len(ibm_qcs)} IBM circuits", flush=True)
    if QISKIT_OK and ibm_qcs:
        try:
            from qiskit.qpy import dump
            qpy_path = os.path.join(here, "c5_ibm_circuits_k5.qpy")
            with open(qpy_path, "wb") as fh:
                dump(ibm_qcs[:50], fh)
            print(f"  QPY saved: {qpy_path}", flush=True)
        except Exception as e:
            print(f"  QPY save failed: {e}", flush=True)

    # ── Figure ────────────────────────────────────────────────────────────────
    make_figure(t_s, s_s,
                gamma_coarse, C_coarse, matches_coarse, ZEROS_30,
                gscan_by_k, C_fine_by_k, widths_by_k,
                T_range, fig_path)

    # ── JSON ──────────────────────────────────────────────────────────────────
    heis_gain = {}
    w1 = widths_by_k.get(1, 1.0)
    for k in K_VALS:
        w = widths_by_k.get(k, 0)
        heis_gain[str(k)] = round(w1 / w, 3) if w > 0 else None

    results = {
        "experiment": "C5 Mertens-Ramsey Zero Spectroscopy",
        "n_t_points": N_T,
        "t_range": [round(float(t_s[0]), 3), round(float(t_s[-1]), 3)],
        "T_range": round(float(T_range), 3),
        "n_shots": "analytic",
        "k_values": K_VALS,
        "coarse_scan": {
            "gamma_range": [float(gamma_coarse[0]), float(gamma_coarse[-1])],
            "n_zeros_matched": len(matches_coarse),
            "matches": matches_coarse,
        },
        "fine_scan": {
            "description": "peak at gamma_scan = gamma_1/k; HWHM scales as 1/k",
            "peak_widths_hwhm_gamma_scan": {str(k): round(w, 5)
                                             for k, w in widths_by_k.items()},
            "theory_hwhm": {str(k): round(np.pi / (k * T_range), 5)
                            for k in K_VALS},
            "heisenberg_gain_vs_k1": heis_gain,
        },
        "ibm_spec": {
            "n_qubits": 5,
            "gate_depth": 12,
            "gate_set": ["h", "cx", "rz"],
            "n_circuits_quick_test": len(g_ibm) * len(t_ibm),
            "shots_per_circuit": 200,
            "total_shots_quick": len(g_ibm) * len(t_ibm) * 200,
            "estimated_ibm_runtime_min": 5,
            "gamma_scan_range_k5": [round(float(g_ibm[0]), 4),
                                    round(float(g_ibm[-1]), 4)],
            "qpy_file": "c5_ibm_circuits_k5.qpy",
        },
    }
    class _NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2, cls=_NpEncoder)
    print(f"  JSON saved: {json_path}", flush=True)
    print(f"\n=== Done in {time.time()-t_total:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()