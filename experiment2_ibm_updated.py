"""
Experiment 2 (Updated): IBM Quantum Prime-Weighted GHZ Test
===========================================================
Author: Earl Decker & Marco Gericke -- PFBMW Framework, June 2026

UPDATES FROM EXPERIMENT 1 RESULTS
-----------------------------------
  Test 1 confirmed: prime weights ln(p)/p produce slower decoherence than
  ANY random coupling of equal power (100% of 500 random draws, 58,000σ
  separation). This validates the prime phase structure used here.

  Test 2 finding: τ* direction confirmed (C(τ<0) > C(τ>0) by +20.7%)
  but optimal value is τ* = -2.5, not -1.25 from theory. This experiment
  runs a delay scan (50–200 ns) on real hardware to measure τ* empirically.

  Test 3 confirmed: prime-Laplacian eigenvalue statistics are distinct from
  GOE (KS = 0.76) -- prime structure is a genuine spectral universality class.

THREE CIRCUITS RUN HERE
-----------------------
  A) Prime-weighted GHZ  vs  B) Random-phase GHZ
     Tests the core claim: prime weights preserve GHZ parity better.

  C) Delay scan: prime-weighted GHZ at delays [50, 80, 100, 125, 160, 200] ns
     Empirically determines τ* on actual quantum hardware.
     τ_ns = 125 ns corresponds to the theoretical τ = -1.25 (at 100 ns/unit).

IBM SETUP (ACTION REQUIRED)
----------------------------
  IBM deprecated the old "ibm_quantum" channel in early 2025.
  You need a NEW token from https://quantum.ibm.com/account

  Steps:
    1. Log in at https://quantum.ibm.com
    2. Click your profile (top right) → "Manage account"
    3. Copy the API token shown there
    4. Paste it below as IBM_TOKEN
    5. If you have a premium instance, update IBM_INSTANCE too

  Your old token (GaCtpwfjkyVpRcHRy8Ip7T2Bz4NV_...) is the legacy format
  and will not work with qiskit-ibm-runtime >= 0.20.

SET USE_SIMULATOR = True to validate circuits locally without IBM credentials.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from datetime import datetime
from qiskit import QuantumCircuit, transpile

# ─────────────────────────────────────────────────────────────
# CONFIGURATION  ← edit these
# ─────────────────────────────────────────────────────────────
IBM_TOKEN    = "96axnVJAp_PkXhi7mpX8t_CVj1NtzqmHjaApLQ5Pn96Q"
IBM_INSTANCE = "ibm-q/open/main"           # free plan default; update if premium
BACKEND_NAME = None                        # None = auto-pick least-busy

USE_SIMULATOR = False  # Running on real IBM Quantum hardware
SHOTS         = 4096
SEED          = 2026

# Delay scan values (ns) -- maps to τ via τ = -delay_ns / 100
DELAY_SCAN_NS = [50, 80, 100, 125, 160, 200]
# τ equivalents:  -0.5  -0.8  -1.0  -1.25 -1.6  -2.0

# Prime phases from Experiment 1 (confirmed as special)
PRIMES = [2, 3, 5, 7, 11, 13]
PRIME_PHASES = [np.log(p) / p * 2 * np.pi for p in PRIMES[:3]]

rng = np.random.default_rng(SEED)

# ─────────────────────────────────────────────────────────────
# BACKEND SETUP
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("Experiment 2 (Updated): IBM Quantum Prime-Weighted GHZ")
print("=" * 65)

if USE_SIMULATOR:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    backend = AerSimulator()
    print(f"Mode: LOCAL SIMULATOR ({backend.name})")
    print("      Set USE_SIMULATOR=False and add IBM_TOKEN for real hardware.\n")
else:
    from qiskit_ibm_runtime import QiskitRuntimeService
    print("Connecting to IBM Quantum Platform...")

    # Try saving account so the instance is auto-discovered
    try:
        QiskitRuntimeService.save_account(
            channel="ibm_quantum_platform",
            token=IBM_TOKEN,
            overwrite=True,
            set_as_default=True,
        )
        service = QiskitRuntimeService()
    except Exception:
        # Fallback: pass token directly
        service = QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token=IBM_TOKEN,
        )

    print(f"Connected. Discovering backends...")
    if BACKEND_NAME:
        backend = service.backend(BACKEND_NAME)
    else:
        all_backends = service.backends(min_num_qubits=3)
        operational = [b for b in all_backends
                       if b.status().operational and not getattr(b.configuration(), 'simulator', False)]
        if not operational:
            print("No real backends available -- falling back to simulator")
            from qiskit_aer import AerSimulator
            backend = AerSimulator()
        else:
            # Pick least busy
            backend = min(operational, key=lambda b: b.status().pending_jobs)
    print(f"Using backend: {backend.name}\n")


# ─────────────────────────────────────────────────────────────
# CIRCUIT BUILDERS
# ─────────────────────────────────────────────────────────────
def ghz_circuit(phases, delay_ns=0, label=""):
    """
    3-qubit GHZ state with RZ phase kicks and optional delay.
    Measured in the parity (H-rotated) basis.

    Phases encode the environment coupling:
      prime phases → fractal environment (PFBMW prediction)
      random phases → control condition

    Measurement in H basis gives:
      |000⟩ + |111⟩ even-parity outcomes signal GHZ coherence
    """
    qc = QuantumCircuit(3, 3, name=label)

    # GHZ preparation
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)

    # Prime-fractal (or random) phase environment
    for q, phi in enumerate(phases):
        qc.rz(phi, q)

    # Retrocausal delay (τ scan)
    if delay_ns > 0:
        for q in range(3):
            qc.delay(delay_ns, q, unit="ns")

    # Parity readout basis
    qc.h(0); qc.h(1); qc.h(2)
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def random_phases():
    return rng.uniform(0, 2 * np.pi, 3).tolist()


# ─────────────────────────────────────────────────────────────
# BUILD CIRCUITS
# ─────────────────────────────────────────────────────────────
print("Building circuits...")

# A: Prime-weighted (no delay) -- core comparison
qc_prime  = ghz_circuit(PRIME_PHASES, delay_ns=0, label="prime_0ns")

# B: Random-phase (no delay) -- control
qc_random = ghz_circuit(random_phases(), delay_ns=0, label="random_0ns")

# C: Delay scan -- prime phases at each delay value
qc_delay_scan = [
    ghz_circuit(PRIME_PHASES, delay_ns=d, label=f"prime_{d}ns")
    for d in DELAY_SCAN_NS
]

# D: Random baseline at each delay (for normalised comparison)
qc_random_delays = [
    ghz_circuit(random_phases(), delay_ns=d, label=f"random_{d}ns")
    for d in DELAY_SCAN_NS
]

all_circuits = [qc_prime, qc_random] + qc_delay_scan + qc_random_delays
print(f"  Total circuits: {len(all_circuits)}")
for qc in all_circuits:
    print(f"  {qc.name:20s}  depth={qc.depth()}")

# ─────────────────────────────────────────────────────────────
# TRANSPILE AND RUN
# ─────────────────────────────────────────────────────────────
print("\nTranspiling...")
t_circuits = transpile(all_circuits, backend, optimization_level=3)
for orig, t in zip(all_circuits, t_circuits):
    print(f"  {orig.name:20s}  depth after transpile: {t.depth()}")

print("\nRunning circuits...")
if USE_SIMULATOR:
    job = backend.run(t_circuits, shots=SHOTS)
    result = job.result()
    all_counts = [result.get_counts(i) for i in range(len(t_circuits))]
else:
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    sampler = Sampler(mode=backend)
    job = sampler.run(t_circuits, shots=SHOTS)
    print(f"  Job ID: {job.job_id()}")
    print("  Waiting for results...")
    raw = job.result()
    all_counts = [raw[i].data.c.get_counts() for i in range(len(t_circuits))]

print("  Done.\n")


# ─────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────
def parity_coherence(counts):
    """
    GHZ parity coherence in H-rotated basis.
    Even-parity outcomes (000, 011, 101, 110) = GHZ survived.
    Returns coherence C ∈ [0, 1].
    """
    total = sum(counts.values())
    even  = sum(v for k, v in counts.items() if k.count('1') % 2 == 0)
    return even / total


def sigma_score(c_prime, c_random, n_random_samples=1000):
    """
    Estimate how many sigma prime is above random baseline.
    Assumes binomial distribution for the random counts.
    """
    se = np.sqrt(c_random * (1 - c_random) / SHOTS)
    return (c_prime - c_random) / se if se > 0 else 0.0


print("=== ANALYSIS ===\n")

# A vs B: core comparison
C_prime  = parity_coherence(all_counts[0])
C_random = parity_coherence(all_counts[1])
revival  = C_prime - C_random
sigma    = sigma_score(C_prime, C_random)

print(f"[CORE TEST] Prime vs Random (no delay)")
print(f"  Prime-weighted GHZ parity:   {C_prime:.4f}")
print(f"  Random-phase GHZ parity:     {C_random:.4f}")
print(f"  Revival difference:          {revival:+.4f}")
print(f"  Statistical separation:      {sigma:+.1f}σ")
print(f"  Prediction confirmed (>0):   {'YES ✓' if revival > 0 else 'NO ✗'}\n")

# C: Delay scan
print(f"[TAU SCAN] Prime GHZ parity vs delay (τ = -delay/100)")
C_delay_prime  = []
C_delay_random = []
revival_delay  = []

for i, d in enumerate(DELAY_SCAN_NS):
    cp = parity_coherence(all_counts[2 + i])
    cr = parity_coherence(all_counts[2 + len(DELAY_SCAN_NS) + i])
    rev = cp - cr
    tau_equiv = -d / 100.0
    C_delay_prime.append(cp)
    C_delay_random.append(cr)
    revival_delay.append(rev)
    print(f"  delay={d:4d}ns  τ≡{tau_equiv:+.2f}  C_prime={cp:.4f}  C_rand={cr:.4f}  revival={rev:+.4f}")

# Find empirical τ* (delay with maximum coherence)
best_idx     = np.argmax(C_delay_prime)
best_delay   = DELAY_SCAN_NS[best_idx]
tau_star_emp = -best_delay / 100.0
tau_theory   = -1.25

print(f"\n  Empirical τ* = {tau_star_emp:.2f}  (delay = {best_delay} ns)")
print(f"  Theory  τ  = {tau_theory:.2f}  (delay = 125 ns)")
print(f"  |τ* - τ_theory| = {abs(tau_star_emp - tau_theory):.2f}")
tau_match = abs(tau_star_emp - tau_theory) < 0.4
print(f"  Match (< 0.4 units): {'YES ✓' if tau_match else 'NO -- use τ*=' + str(tau_star_emp) + ' in Exp 3'}")


# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────
print("\nGenerating figures...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Experiment 2: IBM Quantum Prime-Weighted GHZ Test\n"
             f"Backend: {backend.name}  |  Shots: {SHOTS}  |  "
             f"{'SIMULATOR' if USE_SIMULATOR else 'REAL HARDWARE'}",
             fontsize=12, fontweight='bold')

# Plot 1: Bar chart prime vs random
ax = axes[0]
vals   = [C_prime, C_random, 0.5]
labels = ["Prime\n(PFBMW)", "Random\n(control)", "Classical\nlimit"]
colors = ["steelblue", "gray", "lightgray"]
bars = ax.bar(labels, vals, color=colors, edgecolor='white', linewidth=0.8)
ax.axhline(0.5, color='tomato', linestyle='--', lw=1.0, label='Classical limit')
for bar, v in zip(bars, vals[:2]):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
            f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_ylabel('GHZ Parity Coherence  C')
ax.set_title(f'Core Test: Prime vs Random\nRevival = {revival:+.4f}  ({sigma:+.1f}σ)')
ax.set_ylim(0.4, max(C_prime, C_random) + 0.04)
ax.grid(True, axis='y', alpha=0.25)

# Plot 2: Delay (τ) scan curves
ax = axes[1]
tau_vals = [-d / 100.0 for d in DELAY_SCAN_NS]
ax.plot(tau_vals, C_delay_prime,  'o-', color='steelblue', lw=2, ms=7, label='Prime-weighted')
ax.plot(tau_vals, C_delay_random, 's--', color='gray',      lw=1.5, ms=6, label='Random-phase')
ax.axvline(tau_star_emp, color='steelblue', linestyle=':',  lw=1.2, label=f'τ* measured = {tau_star_emp:.2f}')
ax.axvline(tau_theory,   color='tomato',    linestyle='--', lw=1.2, label=f'τ* theory = {tau_theory:.2f}')
ax.set_xlabel('Retrocausal parameter  τ  (= −delay/100)')
ax.set_ylabel('GHZ Parity Coherence  C')
ax.set_title('τ Scan: Empirical Optimal\n(Exp 1 Test 2 follow-up)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.25)

# Plot 3: Revival difference vs τ
ax = axes[2]
ax.bar(tau_vals, revival_delay, color=['steelblue' if r > 0 else 'tomato' for r in revival_delay],
       width=0.08, edgecolor='white')
ax.axhline(0, color='black', lw=0.8)
ax.axvline(tau_star_emp, color='steelblue', linestyle=':', lw=1.2, label=f'τ* = {tau_star_emp:.2f}')
ax.axvline(tau_theory,   color='tomato',    linestyle='--', lw=1.2, label=f'Theory τ = {tau_theory:.2f}')
ax.set_xlabel('τ value')
ax.set_ylabel('Revival  (C_prime − C_random)')
ax.set_title('Revival Difference vs τ\n(positive = prime wins)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.25)

plt.tight_layout()
fig_path = 'Experiment2_Updated_Results.png'
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
print(f"Figure saved: {fig_path}")


# ─────────────────────────────────────────────────────────────
# SAVE JSON
# ─────────────────────────────────────────────────────────────
results = {
    "timestamp":       datetime.now().isoformat(),
    "backend":         backend.name,
    "mode":            "simulator" if USE_SIMULATOR else "real_hardware",
    "shots":           SHOTS,
    "prime_phases":    [float(p) for p in PRIME_PHASES],
    "core_test": {
        "C_prime":     float(C_prime),
        "C_random":    float(C_random),
        "revival":     float(revival),
        "sigma":       float(sigma),
        "confirmed":   bool(revival > 0),
    },
    "tau_scan": {
        "delay_ns":       DELAY_SCAN_NS,
        "tau_equivalents": [float(-d/100) for d in DELAY_SCAN_NS],
        "C_prime":         [float(c) for c in C_delay_prime],
        "C_random":        [float(c) for c in C_delay_random],
        "revival":         [float(r) for r in revival_delay],
        "tau_star_empirical": float(tau_star_emp),
        "tau_star_theory":    float(tau_theory),
        "tau_match":          bool(tau_match),
    },
    "pivot_notes": {
        "if_no_revival":    "Vary number of qubits (3→5→7) -- larger GHZ states amplify the prime signature",
        "if_tau_mismatch":  f"Use τ*={tau_star_emp:.2f} (empirical) in Experiment 3 instead of -1.25",
        "if_tau_match":     "τ=-1.25 confirmed -- proceed to Experiment 3: g