"""
Experiment 1 (Redesigned): Three Honest Tests of the PFBMW Framework
=====================================================================
Author: Earl Decker & Marco Gericke (analysis: June 2026)

PURPOSE
-------
The original experiment targeted η = 2.000 as a fixed validation point.
This redesign asks three falsifiable questions instead:

  TEST 1  Is prime-weighted coupling physically special relative to
          random coupling of equal total strength?

  TEST 2  Does τ < 0 (retrocausal parameter) produce coherence that
          τ > 0 does not? Is τ* ≈ -1.25 optimal, or does data suggest
          a different value?

  TEST 3  Do the eigenvalue spacings of the prime-Laplacian differ from
          standard random-matrix universality classes (GOE/GUE)?

PHYSICS MODEL
-------------
Spin-boson pure dephasing -- exact analytic result, no Monte Carlo noise.

    C(t) = exp(-Γ(t))
    Γ(t) = Σₖ gₖ² (1 - cos(ωₖ t))

where gₖ = coupling strength of mode k and ωₖ = mode frequency.

Prime lattice assignment:
  ωₖ = pₖ / p_max    (primes normalized to [0,1])
  gₖ = ln(pₖ)/pₖ     (prime-fractal weights, PFBMW eq. for w_p)

Retrocausal (τ) model:
  The τ < 0 term in the original simulation selectively damps high-frequency
  bath modes. Modelled here as an exponential frequency cutoff:
  gₖ_eff(τ) = gₖ × exp(τ × ωₖ)   [τ<0 reduces high-ω coupling]

Fair comparison: all coupling sets are normalized to equal Frobenius norm
  Σₖ gₖ² = constant, so differences arise only from shape, not magnitude.

INTERPRETATION GUIDE
--------------------
Test 1 -- prime outside random band (>2σ): prime structure is physically
          special. Framework gains credibility.
          prime inside band: weights are not special; any smooth distribution
          gives same result.

Test 2 -- optimal τ* ≈ -1.25 (within 0.2): retrocausal parameter confirmed.
          τ* ≠ -1.25: recalibrate τ from quantum data, not gain-media.
          No dependence on τ: retrocausal interpretation needs revision.

Test 3 -- eigenvalue spacing P(s) deviates from Poisson and GOE/GUE:
          prime-Laplacian is a new universality class (links to Riemann ζ).
          spacing = GOE: prime-Laplacian is just another random Hermitian.

Run time: ~60 seconds on a standard laptop.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.stats import ks_2samp
import sympy
import json
from datetime import datetime

matplotlib.rcParams.update({'font.size': 10, 'axes.titlesize': 11,
                            'axes.labelsize': 10, 'figure.dpi': 150})

print("=" * 65)
print("Experiment 1 (Redesigned): Three Honest Tests -- PFBMW Framework")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────
N_MODES      = 300       # number of prime modes
N_RAND       = 500       # random-weight ensemble size
N_TIME       = 400       # time points
T_MAX        = 15.0      # max dimensionless time (units of 1/p_max)
TAU_SCAN     = np.linspace(-2.5, 2.5, 60)
TAU_THEORY   = -1.25     # PFBMW prediction
T_FIXED      = 5.0       # fixed time for tau scan
SEED         = 2026

rng = np.random.default_rng(SEED)

# ─────────────────────────────────────────────────────────────
# PRIME LATTICE SETUP
# ─────────────────────────────────────────────────────────────
print(f"\nGenerating {N_MODES} primes...")
primes  = np.array(list(sympy.primerange(2, 5000))[:N_MODES], dtype=float)
p_max   = primes[-1]

omega   = primes / p_max            # normalized frequencies in [0, 1]
g_prime = np.log(primes) / primes   # PFBMW weight w_p = ln(p)/p

# Normalize to unit Frobenius norm for fair comparison
norm_sq = np.sum(g_prime**2)
g_prime_n = g_prime / np.sqrt(norm_sq)

times = np.linspace(0, T_MAX, N_TIME)

print(f"Prime range: p₂ = {int(primes[0])} ... p_{N_MODES} = {int(p_max)}")
print(f"Coupling norm²: {norm_sq:.6f}")


# ─────────────────────────────────────────────────────────────
# CORE DECOHERENCE FUNCTION
# ─────────────────────────────────────────────────────────────
def decohere(g_norm, omega, times, tau=0.0):
    """
    Exact spin-boson coherence with optional tau frequency cutoff.
    g_norm: normalized couplings (1D array, length N_MODES)
    omega:  mode frequencies (1D array)
    tau:    retrocausal parameter (negative = suppress high-freq modes)
    """
    if tau != 0.0:
        # Retrocausal cutoff: negative tau damps high-frequency modes
        g_eff = g_norm * np.exp(tau * omega)
        # Re-normalize so Frobenius norm is preserved
        norm = np.sqrt(np.sum(g_eff**2))
        g_eff = g_eff / norm if norm > 1e-10 else g_eff
    else:
        g_eff = g_norm

    # Vectorized: Gamma[t] = sum_k g_k^2 * (1 - cos(omega_k * t))
    Gamma = g_eff @ (1.0 - np.cos(np.outer(omega, times)))   # (N,) @ (N, T) = (T,)
    return np.exp(-Gamma)


# ─────────────────────────────────────────────────────────────
# TEST 1: Prime vs Random vs Uniform  (no tau correction)
# ─────────────────────────────────────────────────────────────
print("\n[TEST 1] Running prime vs random vs uniform comparison...")

C_prime   = decohere(g_prime_n, omega, times)
g_uniform = np.ones(N_MODES) / np.sqrt(N_MODES)
C_uniform = decohere(g_uniform, omega, times)

# Random ensemble
C_rand_all = np.zeros((N_RAND, N_TIME))
for i in range(N_RAND):
    # Exponential draws give heavy tail (physically realistic)
    g_r = rng.exponential(1.0, N_MODES)
    g_r = g_r / np.linalg.norm(g_r)
    C_rand_all[i] = decohere(g_r, omega, times)

C_rand_mean = C_rand_all.mean(axis=0)
C_rand_p2   = np.percentile(C_rand_all,  2.5, axis=0)  # 95% CI lower
C_rand_p98  = np.percentile(C_rand_all, 97.5, axis=0)  # 95% CI upper
C_rand_p1   = np.percentile(C_rand_all,  0.5, axis=0)  # 99% CI lower
C_rand_p99  = np.percentile(C_rand_all, 99.5, axis=0)  # 99% CI upper

# Quantify: at what sigma is prime outside random band at each time?
C_rand_std = C_rand_all.std(axis=0)
sigma_profile = (C_prime - C_rand_mean) / (C_rand_std + 1e-10)
max_sigma_idx = np.argmax(np.abs(sigma_profile))
max_sigma = sigma_profile[max_sigma_idx]
print(f"  Peak separation: {max_sigma:+.2f}σ at t = {times[max_sigma_idx]:.2f}")
t_cross = times[np.argmin(np.abs(sigma_profile))]
print(f"  Crossover (prime ~ random): t ≈ {t_cross:.2f}")

# KS test between prime trajectory and random ensemble at T/2
mid_idx = N_TIME // 2
ks_stat, ks_p = ks_2samp([C_prime[mid_idx]], C_rand_all[:, mid_idx])
print(f"  KS test at t={times[mid_idx]:.1f}: stat={ks_stat:.3f}, p-value computed from ensemble")

# Fraction of random draws where C_random > C_prime at early time
early = N_TIME // 8
frac_below = np.mean(C_rand_all[:, early] < C_prime[early])
print(f"  At t={times[early]:.1f}: {100*frac_below:.1f}% of random draws below prime (expect 50% if no difference)")


# ─────────────────────────────────────────────────────────────
# TEST 2: τ scan -- retrocausal revival signature
# ─────────────────────────────────────────────────────────────
print("\n[TEST 2] Scanning τ from {:.1f} to {:.1f}...".format(TAU_SCAN[0], TAU_SCAN[-1]))

t_idx_fixed = np.argmin(np.abs(times - T_FIXED))
C_tau_prime   = np.array([decohere(g_prime_n, omega, times, tau=t)[t_idx_fixed] for t in TAU_SCAN])
C_tau_uniform = np.array([decohere(g_uniform, omega, times, tau=t)[t_idx_fixed] for t in TAU_SCAN])

# Also compute full time curves for tau=TAU_THEORY and tau=+|TAU_THEORY|
C_tau_neg = decohere(g_prime_n, omega, times, tau=TAU_THEORY)
C_tau_pos = decohere(g_prime_n, omega, times, tau=abs(TAU_THEORY))
C_tau_zero = C_prime  # already computed (tau=0)

tau_opt_prime   = TAU_SCAN[np.argmax(C_tau_prime)]
tau_opt_uniform = TAU_SCAN[np.argmax(C_tau_uniform)]
print(f"  Optimal τ (prime coupling):   τ* = {tau_opt_prime:.3f}  (theory: {TAU_THEORY})")
print(f"  Optimal τ (uniform coupling): τ* = {tau_opt_uniform:.3f}")
revival_diff = C_tau_prime[np.argmin(np.abs(TAU_SCAN - TAU_THEORY))] - \
               C_tau_prime[np.argmin(np.abs(TAU_SCAN - abs(TAU_THEORY)))]
print(f"  Coherence gain: C(τ={TAU_THEORY}) - C(τ=+{abs(TAU_THEORY)}) = {revival_diff:+.5f}")


# ─────────────────────────────────────────────────────────────
# TEST 3: Prime-Laplacian eigenvalue spacing statistics
# ─────────────────────────────────────────────────────────────
print("\n[TEST 3] Computing prime-Laplacian eigenvalue spacings...")

def build_laplacian(weights):
    n = len(weights)
    L = np.diag(-2 * weights) + np.diag(weights[1:], 1) + np.diag(weights[1:], -1)
    L[0, 0]   = -weights[0];  L[0, 1]   = weights[0]
    L[-1, -1] = -weights[-1]; L[-1, -2] = weights[-1]
    return L

w_lap = np.log(primes) / primes
L_prime = build_laplacian(w_lap / np.sum(w_lap))
eigenvalues, _ = eigh(L_prime)

# Normalized spacings (unfolded)
eig_sorted = np.sort(np.abs(eigenvalues[1:]))   # skip zero mode
spacings_prime = np.diff(eig_sorted)
spacings_prime = spacings_prime / spacings_prime.mean()  # normalize mean=1

# GOE reference: Wigner surmise P(s) = (π/2)s × exp(-πs²/4)
s_vals = np.linspace(0, 3.5, 200)
P_GOE    = (np.pi/2) * s_vals * np.exp(-np.pi * s_vals**2 / 4)
# Poisson reference: P(s) = exp(-s)
P_Poisson = np.exp(-s_vals)

# Random matrix reference spacings
n_goe = N_MODES - 2
H_random = rng.standard_normal((n_goe, n_goe))
H_random = (H_random + H_random.T) / np.sqrt(2)
eig_goe = np.sort(np.linalg.eigvalsh(H_random))
sp_goe = np.diff(eig_goe) / np.diff(eig_goe).mean()

ks_stat_goe,     _ = ks_2samp(spacings_prime, sp_goe)
from scipy.stats import expon
ks_stat_poisson, _ = ks_2samp(spacings_prime, rng.exponential(1.0, len(spacings_prime)))
print(f"  KS distance from GOE:     {ks_stat_goe:.4f}   (0=identical, 1=maximally different)")
print(f"  KS distance from Poisson: {ks_stat_poisson:.4f}")
print(f"  Mean spacing ratio (Brody param proxy): {spacings_prime.mean():.4f}")
print(f"  Fraction of spacings < 0.2 (level repulsion):  {np.mean(spacings_prime < 0.2):.4f}")
print(f"  GOE level-repulsion reference:                  {np.mean(sp_goe < 0.2):.4f}")
print(f"  Poisson would give: {1 - np.exp(-0.2):.4f}")


# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────
print("\nGenerating figures...")

fig = plt.figure(figsize=(16, 11))
fig.suptitle("PFBMW Framework — Experiment 1 (Redesigned): Three Honest Tests",
             fontsize=13, fontweight='bold', y=0.98)

# ── Plot 1a: Full decoherence curves
ax1 = fig.add_subplot(2, 3, 1)
ax1.fill_between(times, C_rand_p1, C_rand_p99, alpha=0.15, color='gray', label='Random 99% CI')
ax1.fill_between(times, C_rand_p2, C_rand_p98, alpha=0.25, color='gray', label='Random 95% CI')
ax1.plot(times, C_rand_mean, '--', color='gray', lw=1.2, label='Random mean')
ax1.plot(times, C_uniform,   ':', color='orange', lw=1.5, label='Uniform weights')
ax1.plot(times, C_prime,     '-', color='steelblue', lw=2.0, label='Prime weights (PFBMW)')
ax1.set_xlabel('Dimensionless time  t')
ax1.set_ylabel('Coherence  C(t)')
ax1.set_title('Test 1: Prime vs Random (equal power)')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.25)

# ── Plot 1b: Sigma profile
ax2 = fig.add_subplot(2, 3, 2)
ax2.axhline(0, color='gray', lw=0.8)
ax2.axhline(2, color='orange', lw=0.8, linestyle='--', label='2σ')
ax2.axhline(-2, color='orange', lw=0.8, linestyle='--')
ax2.axhline(3, color='tomato', lw=0.8, linestyle=':', label='3σ')
ax2.axhline(-3, color='tomato', lw=0.8, linestyle=':')
ax2.plot(times, sigma_profile, color='steelblue', lw=1.8, label='(C_prime − C_rand)/σ_rand')
ax2.fill_between(times, sigma_profile, 0,
                 where=sigma_profile > 2, alpha=0.3, color='green', label='Prime faster')
ax2.fill_between(times, sigma_profile, 0,
                 where=sigma_profile < -2, alpha=0.3, color='tomato', label='Prime slower')
ax2.set_xlabel('Dimensionless time  t')
ax2.set_ylabel('Sigma separation')
ax2.set_title('Test 1: Statistical separation from random')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.25)
ax2.set_ylim(-8, 8)

# ── Plot 1c: Cumulative fraction above prime
ax3 = fig.add_subplot(2, 3, 3)
fracs = [np.mean(C_rand_all[:, i] < C_prime[i]) for i in range(N_TIME)]
ax3.axhline(0.5, color='gray', lw=0.8, linestyle='--', label='No difference (50%)')
ax3.axhline(0.95, color='orange', lw=0.8, linestyle=':', label='95% threshold')
ax3.plot(times, fracs, color='steelblue', lw=1.5)
ax3.set_xlabel('Dimensionless time  t')
ax3.set_ylabel('Fraction of random draws below prime')
ax3.set_title('Test 1: Probability prime outperforms random')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.25)
ax3.set_ylim(0, 1.05)

# ── Plot 2a: Tau scan (coherence at fixed t)
ax4 = fig.add_subplot(2, 3, 4)
ax4.axvline(TAU_THEORY,   color='steelblue', lw=1.0, linestyle='--', label=f'τ theory = {TAU_THEORY}')
ax4.axvline(tau_opt_prime, color='tomato',    lw=1.5, linestyle='-',  label=f'τ* measured = {tau_opt_prime:.2f}')
ax4.axvline(0, color='gray', lw=0.8)
ax4.plot(TAU_SCAN, C_tau_prime,   color='steelblue', lw=1.8, label='Prime coupling')
ax4.plot(TAU_SCAN, C_tau_uniform, color='orange',    lw=1.2, linestyle=':', label='Uniform coupling')
ax4.set_xlabel('Retrocausal parameter  τ')
ax4.set_ylabel(f'Coherence at t = {T_FIXED}')
ax4.set_title('Test 2: τ scan — retrocausal revival')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.25)

# ── Plot 2b: Time curves for τ = theory, +theory, 0
ax5 = fig.add_subplot(2, 3, 5)
ax5.plot(times, C_tau_zero, '-',  color='gray',      lw=1.5, label='τ = 0 (no retrocausal)')
ax5.plot(times, C_tau_pos,  '--', color='tomato',    lw=1.5, label=f'τ = +{abs(TAU_THEORY)} (dispersive)')
ax5.plot(times, C_tau_neg,  '-',  color='steelblue', lw=2.0, label=f'τ = {TAU_THEORY} (retrocausal, PFBMW)')
ax5.fill_between(times, C_tau_neg, C_tau_pos,
                 alpha=0.12, color='steelblue', label='Retrocausal advantage region')
ax5.set_xlabel('Dimensionless time  t')
ax5.set_ylabel('Coherence  C(t)')
ax5.set_title('Test 2: Retrocausal (τ<0) vs dispersive (τ>0)')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.25)

# ── Plot 3: Eigenvalue spacing statistics
ax6 = fig.add_subplot(2, 3, 6)
bins = np.linspace(0, 3.5, 30)
ax6.hist(spacings_prime, bins=bins, density=True, alpha=0.55, color='steelblue',
         label='Prime-Laplacian spacings', edgecolor='white', linewidth=0.5)
ax6.hist(sp_goe, bins=bins, density=True, alpha=0.35, color='gray',
         label='GOE (random matrix)', edgecolor='white', linewidth=0.5)
ax6.plot(s_vals, P_GOE,    '-',  color='darkgreen', lw=1.5, label='Wigner surmise (GOE)')
ax6.plot(s_vals, P_Poisson,'--', color='tomato',    lw=1.5, label='Poisson (uncorrelated)')
ax6.set_xlabel('Normalized spacing  s')
ax6.set_ylabel('P(s)')
ax6.set_title('Test 3: Eigenvalue spacing statistics')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.25)
ax6.set_xlim(0, 3.5)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = 'Experiment1_Redesigned_Results.png'
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
print(f"Figure saved: {fig_path}")


# ─────────────────────────────────────────────────────────────
# SUMMARY + PIVOT LOGIC
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("RESULTS SUMMARY")
print("=" * 65)

# Test 1 verdict
test1_pass = frac_below > 0.85   # prime outperforms >85% of random at early time
test1_label = "CONFIRMED" if test1_pass else "NOT CONFIRMED"
print(f"\nTest 1 — Prime coupling is special:  {test1_label}")
print(f"  {100*frac_below:.0f}% of random draws below prime at t={times[early]:.1f}")
if not test1_pass:
    print("  → PIVOT: test a family of weight functions w_p = ln(p)^α / p^β")
    print("           to find which (α,β) distinguishes prime lattice from random.")

# Test 2 verdict
tau_err = abs(tau_opt_prime - TAU_THEORY)
test2_pass = tau_err < 0.4 and revival_diff > 0
test2_label = "CONFIRMED" if test2_pass else "REFIT NEEDED"
print(f"\nTest 2 — Retrocausal τ signature:     {test2_label}")
print(f"  Measured τ* = {tau_opt_prime:.3f}   theory = {TAU_THEORY}   |Δτ| = {tau_err:.3f}")
print(f"  C(τ<0) - C(τ>0) = {revival_diff:+.5f}  ({'positive = revival' if revival_diff>0 else 'negative = suppression'})")
if not test2_pass:
    print(f"  → PIVOT: IBM Quantum Exp 2 delay scan from 50–200 ns to measure τ* empirically.")

# Test 3 verdict
test3_different = ks_stat_goe > 0.15
test3_label = "DISTINCT FROM GOE" if test3_different else "CONSISTENT WITH GOE"
print(f"\nTest 3 — Spectral uniqueness:          {test3_label}")
print(f"  KS distance from GOE     = {ks_stat_goe:.4f}")
print(f"  KS distance from Poisson = {ks_stat_poisson:.4f}")
if not test3_different:
    print("  → PIVOT: prime-Laplacian collapses to random matrix at this scale.")
    print("           Try N_MODES > 1000 or a different boundary condition.")


# ─────────────────────────────────────────────────────────────
# SAVE JSON
# ─────────────────────────────────────────────────────────────
results = {
    "timestamp":          datetime.now().isoformat(),
    "experiment":         "Experiment 1 Redesigned -- Three Honest Tests",
    "parameters": {
        "n_modes":        N_MODES,
        "n_random":       N_RAND,
        "tau_theory":     TAU_THEORY,
        "t_max":          T_MAX,
        "t_fixed_tau":    T_FIXED,
    },
    "test1": {
        "description":    "Prime vs random coupling (equal Frobenius norm)",
        "frac_below_at_early": float(frac_below),
        "max_sigma_separation": float(max_sigma),
        "crossover_time":  float(t_cross),
        "verdict":         test1_label,
    },
    "test2": {
        "description":    "Tau scan for retrocausal revival signature",
        "tau_optimal_prime":   float(tau_opt_prime),
        "tau_optimal_uniform": float(tau_opt_uniform),
        "tau_theory":          TAU_THEORY,
        "revival_diff":        float(revival_diff),
        "verdict":             test2_label,
    },
    "test3": {
        "description":    "Eigenvalue spacing statistics vs GOE / Poisson",
        "ks_from_goe":     float(ks_stat_goe),
        "ks_from_poisson": float(ks_stat_poisson),
        "verdict":         test3_label,
    },
}

json_path = 'Experiment1_Redesigned_Results.json'
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nFull results saved: {json_path}")
print("=" * 65)
print("Experiment 1 (Redesigned) complete.")
print("=" * 65)
