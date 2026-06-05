"""
spectral_gap_zero_proxy.py -- Phase 4: Spectral Gap as Riemann Zero Proxy
==========================================================================
Tests the Berry-Keating prediction: if prime gap fluctuations encode Riemann
zero information, the residuals ε(p) = g(p) - g_smooth(p) should oscillate
at frequencies matching zero heights γ_n when viewed in log-prime space.

METHOD
------
1. Compute g(p) = Δf(p)/[ln(p)/p²] for all primes p ≤ P_MAX
   where Δf(p) = f(p) - f(p+1) (nearest composite gap, always p+1 for p≥5)

2. Fit 4-term smooth model:
   g_smooth(p) = 1 - 1/ln(p) - 1/p + 3/(2·p·ln(p))

3. Compute weighted residual signal:
   w(p) = p^{1/2} · ln(p) · ε(p)
   (The weight p^{1/2}·ln(p) is motivated by the explicit formula:
    δp_n ~ Σ_γ p^{1/2} cos(γ·ln(p)) / γ
    so if ε(p) ~ δp_n·(∂g/∂p), the weighted signal should have Fourier
    components at the zero heights γ_n)

4. Compute the power spectrum |F(γ)|² where:
   F(γ) = Σ_n w(p_n) · exp(-i·γ·ln(p_n))
   evaluated at γ = γ_n (Riemann zero heights) and at random γ for null.

5. Statistical test: compare |F(γ_n)|² distribution vs |F(γ_random)|²

KNOWN RIEMANN ZERO HEIGHTS (imaginary parts of first 30 non-trivial zeros)
---------------------------------------------------------------------------
γ_1  = 14.13472514...    γ_11 = 52.97032147...
γ_2  = 21.02203964...    γ_12 = 56.44624770...
γ_3  = 25.01085758...    γ_13 = 59.34704400...
γ_4  = 30.42487613...    γ_14 = 60.83177852...
γ_5  = 32.93506159...    γ_15 = 65.11254405...
γ_6  = 37.58617815...    γ_16 = 67.07981053...
γ_7  = 40.91871901...    γ_17 = 69.54640171...
γ_8  = 43.32707328...    γ_18 = 72.06715767...
γ_9  = 48.00515088...    γ_19 = 75.70469069...
γ_10 = 49.77383248...    γ_20 = 77.14484006...

OUTPUTS
-------
  spectral_gap_zero_proxy.png  -- 4-panel figure
  spectral_gap_zero_proxy.json -- numerical results
"""

import os, math, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal as sp_signal

script_dir = os.path.dirname(os.path.abspath(__file__))

# ── Parameters ────────────────────────────────────────────────────────────────

P_MAX = 500_000   # sieve primes up to this bound

# First 30 Riemann zero heights (imaginary parts) — high precision
RIEMANN_ZEROS = [
    14.134725141734693, 21.022039638771554, 25.010857580145688,
    30.424876125859513, 32.935061587739189, 37.586178158825671,
    40.918719012147495, 43.327073280914999, 48.005150881167159,
    49.773832477672302, 52.970321477714460, 56.446247697063246,
    59.347044002602352, 60.831778524609809, 65.112544048081607,
    67.079810529494173, 69.546401711173979, 72.067157674481907,
    75.704690699083289, 77.144840068874805, 79.337375020249367,
    82.910380854086030, 84.735492980517050, 87.425274613125229,
    88.809111208594001, 92.491899270558484, 94.651344040519896,
    95.870634228245309, 98.831194218193692, 101.31785100573139,
]

# ── Sieves and helpers ────────────────────────────────────────────────────────

def sieve(n):
    """Return sorted array of primes <= n."""
    is_p = bytearray([1]) * (n + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return np.array([i for i, v in enumerate(is_p) if v], dtype=np.int64)

def f(n):
    return math.log(n) / n

# ── 1. Compute g(p) ───────────────────────────────────────────────────────────

print(f"Sieving primes up to {P_MAX:,} ...")
primes = sieve(P_MAX)
# Drop p=2,3 (p+1 not always composite for very small p; also p=2 is degenerate)
primes = primes[primes >= 5]
print(f"  {len(primes):,} primes from 5 to {primes[-1]:,}")

print("Computing Δf(p) and g(p) ...")
lp   = np.log(primes.astype(float))
lp1  = np.log((primes + 1).astype(float))
fp   = lp / primes
fp1  = lp1 / (primes + 1)

delta_f = fp - fp1
asymp   = lp / primes**2

# g(p) = Δf(p) / [ln(p)/p²]
g = delta_f / asymp

# 4-term smooth model
g_smooth = 1.0 - 1.0/lp - 1.0/primes + 1.5/(primes * lp)

epsilon = g - g_smooth

print(f"  g(p) range: [{g.min():.6f}, {g.max():.6f}]")
print(f"  g_smooth range: [{g_smooth.min():.6f}, {g_smooth.max():.6f}]")
print(f"  ε(p) std: {epsilon.std():.6e}")
print()

# ── 2. Weighted residual signal ───────────────────────────────────────────────

# Weight motivated by explicit formula amplitude
w = primes**0.5 * lp * epsilon
print(f"  w(p) std: {w.std():.6e}")
print()

# ── 3. Fourier transform evaluated at zero heights ───────────────────────────
# F(γ) = Σ_n w(p_n) exp(-iγ ln(p_n))
# This is a non-uniform DFT at prescribed frequencies γ

t = lp.astype(np.float64)   # t_n = ln(p_n) — the "time" variable

print("Computing |F(γ)|² at Riemann zero heights ...")
zero_power = []
for gamma in RIEMANN_ZEROS:
    phase = np.exp(-1j * gamma * t)
    F = np.sum(w * phase)
    zero_power.append(abs(F)**2)
zero_power = np.array(zero_power)

# Null distribution: 1000 random γ in same range
rng = np.random.default_rng(42)
gamma_null = rng.uniform(RIEMANN_ZEROS[0], RIEMANN_ZEROS[-1], 1000)
null_power = []
for gamma in gamma_null:
    F = np.sum(w * np.exp(-1j * gamma * t))
    null_power.append(abs(F)**2)
null_power = np.array(null_power)

# Z-score of zero heights vs null distribution
null_mean = null_power.mean()
null_std  = null_power.std()
z_scores  = (zero_power - null_mean) / null_std

print(f"\nNull distribution: mean={null_mean:.3e}, std={null_std:.3e}")
print(f"\nRiemann zero |F(γ)|² and Z-scores:")
print(f"  {'γ_n':>10}  {'|F|²':>12}  {'Z':>7}")
print("  " + "-"*35)
for i, (gamma, pwr, z) in enumerate(zip(RIEMANN_ZEROS, zero_power, z_scores)):
    flag = " <-- peak" if z > 2.0 else ""
    print(f"  {gamma:>10.5f}  {pwr:>12.3e}  {z:>7.2f}{flag}")

# Summary stats
n_above_2sig = (z_scores > 2.0).sum()
n_above_3sig = (z_scores > 3.0).sum()
expected_2sig = 30 * 0.0228  # expected by chance
print(f"\nZeros with Z > 2σ: {n_above_2sig} (expected by chance: {expected_2sig:.1f})")
print(f"Zeros with Z > 3σ: {n_above_3sig}")
print()

# ── 4. Dense power spectrum for visual inspection ────────────────────────────

print("Computing dense power spectrum from γ=10 to 110 ...")
gamma_dense = np.linspace(10, 110, 10000)
power_dense = np.zeros(len(gamma_dense))

# Batch for speed: chunk over gamma
CHUNK = 500
for ci in range(0, len(gamma_dense), CHUNK):
    gc = gamma_dense[ci:ci+CHUNK]
    # phase matrix: (n_gamma, n_primes)
    phases = np.exp(-1j * np.outer(gc, t))
    F_batch = phases @ w
    power_dense[ci:ci+CHUNK] = np.abs(F_batch)**2

# Smooth with Gaussian for visual clarity
from scipy.ndimage import gaussian_filter1d
power_smooth = gaussian_filter1d(power_dense, sigma=15)
print("Done.")
print()

# ── 5. Figure ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    'Phase 4: Spectral Gap as Riemann Zero Proxy\n'
    r'$g(p) = \Delta f(p)\,/\,[\ln p/p^2]$ residuals vs zero heights $\gamma_n$',
    fontsize=12, fontweight='bold'
)
fig.patch.set_facecolor('white')

# ── Panel A: g(p) and smooth fit ──────────────────────────────────────────────
ax = axes[0, 0]
ax.set_facecolor('#f8f9fa')
# Subsample for display
stride = max(1, len(primes) // 5000)
ps = primes[::stride]
gs = g[::stride]
gs_s = g_smooth[::stride]

ax.scatter(np.log(ps), gs, s=0.5, c='#95a5a6', alpha=0.4, label='g(p)')
ax.plot(np.log(ps), gs_s, 'r-', lw=1.5, label=r'$g_{\rm smooth}(p)$', zorder=5)
ax.set_xlabel(r'$\ln(p)$', fontsize=11)
ax.set_ylabel(r'$g(p)$', fontsize=11)
ax.set_title(r'Correction ratio $g(p) = \Delta f(p)\,/\,[\ln p/p^2]$', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
ax.set_ylim(0.3, 1.05)

# ── Panel B: Residuals ε(p) ───────────────────────────────────────────────────
ax2 = axes[0, 1]
ax2.set_facecolor('#f8f9fa')
eps_s = epsilon[::stride]
ax2.scatter(np.log(ps), eps_s, s=0.5, c='#2980b9', alpha=0.4)
ax2.axhline(0, color='k', lw=0.8, ls='--')
ax2.set_xlabel(r'$\ln(p)$', fontsize=11)
ax2.set_ylabel(r'$\varepsilon(p) = g(p) - g_{\rm smooth}(p)$', fontsize=11)
ax2.set_title(r'Residuals after subtracting analytic $g_{\rm smooth}$', fontsize=10)
ax2.text(0.02, 0.97, f'std = {epsilon.std():.3e}', transform=ax2.transAxes,
         va='top', fontsize=9, color='#2980b9')
ax2.grid(True, alpha=0.2)

# ── Panel C: Dense power spectrum with zero heights marked ────────────────────
ax3 = axes[1, 0]
ax3.set_facecolor('#f8f9fa')
ax3.plot(gamma_dense, power_dense / null_mean, color='#bdc3c7', lw=0.5, alpha=0.7)
ax3.plot(gamma_dense, power_smooth / null_mean, color='#2980b9', lw=1.5,
         label='Power spectrum (smoothed)')
ax3.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.5, label='Null mean')
ax3.axhline(1 + 2*null_std/null_mean, color='#e74c3c', lw=0.8, ls=':',
            label='2σ threshold', alpha=0.7)

# Mark zero heights as vertical lines
for gamma in RIEMANN_ZEROS:
    ax3.axvline(gamma, color='#e74c3c', lw=0.6, alpha=0.45, zorder=3)

# Mark strongest zeros
for i, (gamma, z) in enumerate(zip(RIEMANN_ZEROS, z_scores)):
    if z > 2.0:
        ax3.annotate(f'γ={gamma:.1f}\nZ={z:.1f}', xy=(gamma, power_smooth[
            np.argmin(np.abs(gamma_dense - gamma))] / null_mean),
            xytext=(gamma + 1.5, ax3.get_ylim()[1] * 0.85 if i % 2 == 0 else ax3.get_ylim()[1] * 0.7),
            fontsize=7, color='#c0392b',
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=0.8))

ax3.set_xlabel(r'$\gamma$', fontsize=11)
ax3.set_ylabel(r'$|F(\gamma)|^2 / \langle|F|^2\rangle_{\rm null}$', fontsize=11)
ax3.set_title(r'Power spectrum of $w(p) = p^{1/2}\ln(p)\,\varepsilon(p)$'
              '\n(red lines: Riemann zero heights)', fontsize=10)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.2)

# ── Panel D: Z-score summary ──────────────────────────────────────────────────
ax4 = axes[1, 1]
ax4.set_facecolor('#f8f9fa')
colors_bar = ['#e74c3c' if z > 2 else '#3498db' if z > 1 else '#95a5a6'
              for z in z_scores]
x_pos = np.arange(len(RIEMANN_ZEROS))
bars = ax4.bar(x_pos, z_scores, color=colors_bar, alpha=0.8, edgecolor='white')
ax4.axhline(2.0, color='#e74c3c', lw=1.2, ls='--', label='2σ', alpha=0.7)
ax4.axhline(0.0, color='k', lw=0.8, alpha=0.5)
ax4.set_xticks(x_pos)
ax4.set_xticklabels([f'γ{i+1}\n{g:.0f}' for i, g in enumerate(RIEMANN_ZEROS)],
                     fontsize=6)
ax4.set_ylabel('Z-score vs null distribution', fontsize=11)
ax4.set_title(f'|F(γ_n)|² Z-scores for first 30 Riemann zeros\n'
              f'({n_above_2sig} > 2σ, expected {expected_2sig:.1f} by chance)', fontsize=10)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.2, axis='y')

# Add interpretation box
msg = ("Berry-Keating prediction:\n"
       "peaks at γ_n → zeros encoded\n"
       f"Observed: {n_above_2sig}/30 > 2σ\n"
       f"Expected by chance: {expected_2sig:.1f}/30")
ax4.text(0.98, 0.97, msg, transform=ax4.transAxes, va='top', ha='right',
         fontsize=8, color='#2c3e50',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#bdc3c7'))

plt.tight_layout()
out_fig = os.path.join(script_dir, 'spectral_gap_zero_proxy.png')
plt.savefig(out_fig, dpi=180, bbox_inches='tight')
plt.close()
print(f"Figure saved: spectral_gap_zero_proxy.png")

# ── 6. Save JSON report ───────────────────────────────────────────────────────

report = {
    'parameters': {
        'P_MAX': P_MAX,
        'n_primes': int(len(primes)),
        'primes_range': [int(primes[0]), int(primes[-1])],
    },
    'residuals': {
        'epsilon_std': float(epsilon.std()),
        'epsilon_mean': float(epsilon.mean()),
        'w_std': float(w.std()),
    },
    'null_distribution': {
        'n_samples': 1000,
        'mean': float(null_mean),
        'std': float(null_std),
    },
    'zero_analysis': [
        {
            'index': i + 1,
            'gamma': gamma,
            'power': float(pwr),
            'z_score': float(z),
        }
        for i, (gamma, pwr, z) in enumerate(zip(RIEMANN_ZEROS, zero_power, z_scores))
    ],
    'summary': {
        'n_above_2sigma': int(n_above_2sig),
        'n_above_3sigma': int(n_above_3sig),
        'expected_by_chance_2sigma': float(expected_2sig),
        'interpretation': (
            'WEAK SIGNAL' if n_above_2sig > 3 else
            'CONSISTENT WITH NULL' if n_above_2sig <= expected_2sig + 1 else
            'MARGINAL'
        ),
    },
}
out_json = os.path.join(script_dir, 'spectral_gap_zero_proxy.json')
with open(out_json, 'w') as fh:
    json.dump(report, fh, indent=2)
print(f"Report saved: spectral_gap_zero_proxy.json")
print()
print("="*60)
print("PHASE 4 SUMMARY")
print("="*60)
print(f"  Primes analyzed: {len(primes):,}")
print(f"  ε(p) std: {epsilon.std():.3e}")
print(f"  Zeros with Z > 2σ: {n_above_2sig}/30 (expected {expected_2sig:.1f})")
print(f"  Interpretation: {report['summary']['interpretation']}")
print("="*60)
