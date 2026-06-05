"""
Experiment 3: 6-Qubit GHZ Critical Test -- PFBMW Framework
===========================================================
Author: Earl Decker & Marco Gericke -- June 2026

THE CRITICAL QUESTION
---------------------
Experiment 2 on ibm_marrakesh showed prime-weighted GHZ parity = 0.964 vs
random = 0.395 (+74.6σ). But is this because of:

  (A) MERTENS ALIGNMENT: the 3-prime sum Σ ln(p)/p × 2π ≈ 2π by accident,
      making the ideal coherence 0.988 regardless of hardware?
  OR
  (B) PRIME STRUCTURE: the specific distribution of weights ln(p)/p is
      intrinsically robust against quantum decoherence?

This experiment separates these cleanly using the OPTIMAL prime set:
  6 primes (2,3,5,7,11,13): Mertens sum = 1.728 → ideal P(even) = 0.431
  This is near 0.5 (maximum measurement sensitivity = 0.981).
  The Mertens alignment advantage is absent, and the test is optimally sensitive.

  n=5 (p=2,3,5,7,11): sensitivity=0.037 (terrible -- ideal≈0, both near zero)
  n=6 (p=2,3,5,7,11,13): sensitivity=0.981 (optimal -- ideal≈0.43, mid-range)

THREE PHASE CONDITIONS (run on real hardware)
---------------------------------------------
  RAW PRIME:        φₖ = ln(pₖ)/pₖ × 2π  (no alignment, ideal ≈ 0%)
  NORMALIZED PRIME: φₖ scaled so Φ = 2π   (removes Mertens, tests structure)
  NORMALIZED RANDOM: random weights, Φ = 2π (control for normalized prime)

  + τ scan at empirical τ* = 50 ns and theory τ = 125 ns

PREDICTIONS
-----------
  If (A) only -- Mertens was the whole story:
    raw prime ≈ random ≈ 0.009 (no advantage at 5-qubit scale)
    norm prime ≈ norm random (structure doesn't help)

  If (B) -- prime structure is protective:
    norm prime >> norm random (even without Mertens advantage)
    raw prime may still show revival vs raw random of matched phase sum

  If (A) + (B) both:
    norm prime > norm random AND raw prime shows some advantage over matched random

IBM SETUP
---------
  Token and backend inherited from Experiment 2 settings.
  Requires: qiskit, qiskit-ibm-runtime (upgraded), qiskit-aer, numpy, matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from datetime import datetime
from qiskit import QuantumCircuit, transpile

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
IBM_TOKEN    = "96axnVJAp_PkXhi7mpX8t_CVj1NtzqmHjaApLQ5Pn96Q"
IBM_INSTANCE = "ibm-q/open/main"
BACKEND_NAME = None          # None = auto-pick least-busy ≥5 qubit backend

USE_SIMULATOR = False        # True = local Aer, False = real IBM hardware
SHOTS         = 4096
SEED          = 2026

# 6 primes -- optimal sensitivity (ideal P_even = 0.431, sensitivity = 0.981)
# n=5 (p=..11) has sensitivity=0.037 -- too insensitive (ideal near zero)
# n=6 (p=..13) has sensitivity=0.981 -- maximally sensitive
PRIMES = [2, 3, 5, 7, 11, 13]
N_QUBITS = 6

# Delay scan (ns) -- τ* = 50 ns (empirical Exp 2) + theory + control
DELAYS_NS = [0, 50, 125]
# τ equivalents: 0.0, -0.50, -1.25

# Number of random seeds for ensemble
N_RANDOM_SEEDS = 5

rng = np.random.default_rng(SEED)


# ─────────────────────────────────────────────────────────────
# PHASE CALCULATIONS
# ─────────────────────────────────────────────────────────────
raw_phases   = [np.log(p) / p * 2 * np.pi for p in PRIMES]
Phi_raw      = sum(raw_phases)
norm_factor  = 2 * np.pi / Phi_raw
norm_phases  = [phi * norm_factor for phi in raw_phases]

P_even_raw_ideal  = np.cos(Phi_raw / 2) ** 2
P_even_norm_ideal = 1.0  # cos²(π) = 1 by construction

print("=" * 65)
print("Experiment 3: 5-Qubit GHZ Critical Test -- PFBMW Framework")
print("=" * 65)
print(f"\nPrime set: {PRIMES}")
print(f"Mertens sum Σ ln(p)/p = {sum(np.log(p)/p for p in PRIMES):.4f}")
print(f"\nRAW phases:        {[round(p, 4) for p in raw_phases]}")
print(f"  Total Φ = {Phi_raw:.4f} rad ({Phi_raw/(2*np.pi):.4f} cycles)")
print(f"  Ideal P(even) = {P_even_raw_ideal:.4f}  ← Mertens protection ABSENT")
print(f"\nNORMALIZED phases: {[round(p, 4) for p in norm_phases]}")
print(f"  Total Φ = 2π by construction")
print(f"  Ideal P(even) = {P_even_norm_ideal:.4f}  ← equal footing for all conditions")


# ─────────────────────────────────────────────────────────────
# BACKEND SETUP
# ─────────────────────────────────────────────────────────────
if USE_SIMULATOR:
    from qiskit_aer import AerSimulator
    backend = AerSimulator()
    print(f"\nMode: LOCAL SIMULATOR ({backend.name})")
else:
    from qiskit_ibm_runtime import QiskitRuntimeService
    print("\nConnecting to IBM Quantum Platform...")
    try:
        QiskitRuntimeService.save_account(
            channel="ibm_quantum_platform",
            token=IBM_TOKEN,
            overwrite=True,
            set_as_default=True,
        )
        service = QiskitRuntimeService()
    except Exception:
        service = QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token=IBM_TOKEN,
        )

    print("Connected. Discovering backends...")
    if BACKEND_NAME:
        backend = service.backend(BACKEND_NAME)
    else:
        all_backends = service.backends(min_num_qubits=N_QUBITS)
        operational = [b for b in all_backends
                       if b.status().operational
                       and not getattr(b.configuration(), 'simulator', False)]
        if not operational:
            print("No real backends available -- falling back to simulator")
            from qiskit_aer import AerSimulator
            backend = AerSimulator()
        else:
            backend = min(operational, key=lambda b: b.status().pending_jobs)
    print(f"Using backend: {backend.name} ({N_QUBITS}-qubit circuits)")


# ─────────────────────────────────────────────────────────────
# CIRCUIT BUILDER
# ─────────────────────────────────────────────────────────────
def ghz5_circuit(phases, delay_ns=0, label=""):
    """
    N-qubit GHZ circuit with RZ phase kicks and optional delay.
    Measured in parity (H-rotated) basis.
    """
    n = len(phases)
    qc = QuantumCircuit(n, n, name=label)

    # GHZ preparation
    qc.h(0)
    for q in range(n - 1):
        qc.cx(q, q + 1)

    # Phase kicks (the experimental variable)
    for q, phi in enumerate(phases):
        qc.rz(phi, q)

    # Optional retrocausal delay
    if delay_ns > 0:
        for q in range(n):
            qc.delay(delay_ns, q, unit="ns")

    # Parity readout basis
    for q in range(n):
        qc.h(q)
    qc.measure(range(n), range(n))
    return qc


# ─────────────────────────────────────────────────────────────
# BUILD CIRCUIT BATCH
# ─────────────────────────────────────────────────────────────
print("\nBuilding circuits...")

circuit_meta = []   # (label, ideal_P_even, condition, delay_ns)
all_circuits = []

for delay in DELAYS_NS:
    tau = -delay / 100.0
    d_tag = f"d{delay}"

    # 1. Raw prime phases
    qc = ghz5_circuit(raw_phases, delay_ns=delay, label=f"raw_prime_{d_tag}")
    all_circuits.append(qc)
    circuit_meta.append(("raw_prime", delay, tau, P_even_raw_ideal))

    # 2. Normalized prime phases
    qc = ghz5_circuit(norm_phases, delay_ns=delay, label=f"norm_prime_{d_tag}")
    all_circuits.append(qc)
    circuit_meta.append(("norm_prime", delay, tau, P_even_norm_ideal))

    # 3. Normalized random (multiple seeds)
    for seed_i in range(N_RANDOM_SEEDS):
        rand_raw = rng.uniform(0, 2 * np.pi, N_QUBITS)
        Phi_rand = rand_raw.sum()
        rand_norm = rand_raw / Phi_rand * 2 * np.pi
        qc = ghz5_circuit(rand_norm.tolist(), delay_ns=delay,
                          label=f"norm_rand_{seed_i}_{d_tag}")
        all_circuits.append(qc)
        circuit_meta.append((f"norm_rand_{seed_i}", delay, tau, P_even_norm_ideal))

    # 4. Raw random matched to prime phase sum (for raw comparison)
    rand_raw2 = rng.uniform(0, 2 * np.pi, N_QUBITS)
    rand_matched = rand_raw2 / rand_raw2.sum() * Phi_raw
    qc = ghz5_circuit(rand_matched.tolist(), delay_ns=delay,
                      label=f"raw_rand_matched_{d_tag}")
    all_circuits.append(qc)
    circuit_meta.append(("raw_rand_matched", delay, tau, P_even_raw_ideal))

print(f"  Total circuits: {len(all_circuits)}")
for qc, (cond, delay, tau, ideal) in zip(all_circuits, circuit_meta):
    print(f"  {qc.name:35s}  depth={qc.depth()}  ideal={ideal:.4f}")


# ─────────────────────────────────────────────────────────────
# TRANSPILE AND RUN
# ─────────────────────────────────────────────────────────────
print("\nTranspiling...")
t_circuits = transpile(all_circuits, backend, optimization_level=3)
depths = [t.depth() for t in t_circuits]
print(f"  Transpiled depths: min={min(depths)}, max={max(depths)}, mean={np.mean(depths):.1f}")

print("Running circuits...")
if USE_SIMULATOR:
    job = backend.run(t_circuits, shots=SHOTS)
    result = job.result()
    all_counts = [result.get_counts(i) for i in range(len(t_circuits))]
else:
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    sampler = Sampler(mode=backend)
    job = sampler.run(t_circuits, shots=SHOTS)
    print(f"  Job ID: {job.job_id()}")
    print("  Waiting for results (may take several minutes)...")
    raw_result = job.result()
    all_counts = [raw_result[i].data.c.get_counts() for i in range(len(t_circuits))]
print("  Done.")


# ─────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────
def parity_coherence(counts, n_qubits):
    total = sum(counts.values())
    even  = sum(v for k, v in counts.items() if k.count('1') % 2 == 0)
    return even / total

def hardware_degradation(measured, ideal):
    """How much coherence was lost to hardware noise."""
    # For ideal near 0.5 (random), this is not meaningful -- only use for prime
    return ideal - measured if ideal > 0.5 else None

print("\n" + "=" * 65)
print("ANALYSIS")
print("=" * 65)

# Collect results by condition and delay
results_by_cond = {}
for i, (cond, delay, tau, ideal) in enumerate(circuit_meta):
    C = parity_coherence(all_counts[i], N_QUBITS)
    key = (cond, delay)
    if key not in results_by_cond:
        results_by_cond[key] = []
    results_by_cond[key].append((C, ideal, tau))

# Aggregate random ensembles
def get_C(cond, delay):
    key = (cond, delay)
    vals = [v[0] for v in results_by_cond.get(key, [])]
    return np.mean(vals), np.std(vals), results_by_cond[key][0][1], results_by_cond[key][0][2]

print("\n[NORMALIZED PRIME vs NORMALIZED RANDOM] -- Pure structural test")
print(f"{'Delay':>8}  {'tau':>6}  {'Norm-Prime':>12}  {'Norm-Rand (mean)':>18}  {'Revival':>10}  {'sigma':>8}")
norm_results = []
for delay in DELAYS_NS:
    tau = -delay / 100.0
    C_np,  _,    id_np, _ = get_C("norm_prime", delay)
    rand_vals = [results_by_cond[(f"norm_rand_{i}", delay)][0][0] for i in range(N_RANDOM_SEEDS)]
    C_nr_mean = np.mean(rand_vals)
    C_nr_std  = np.std(rand_vals)
    revival = C_np - C_nr_mean
    se = np.sqrt(C_nr_mean * (1 - C_nr_mean) / SHOTS)
    sigma = revival / se if se > 0 else 0
    print(f"{delay:>8}ns  {tau:>+6.2f}  {C_np:>12.4f}  {C_nr_mean:>12.4f}±{C_nr_std:.4f}  {revival:>+10.4f}  {sigma:>+8.1f}σ")
    norm_results.append({"delay": delay, "tau": tau, "C_norm_prime": float(C_np),
                         "C_norm_rand_mean": float(C_nr_mean), "C_norm_rand_std": float(C_nr_std),
                         "revival": float(revival), "sigma": float(sigma)})

print("\n[RAW PRIME vs MATCHED RANDOM] -- Full Mertens test")
print(f"  Raw prime ideal P(even) = {P_even_raw_ideal:.4f}  (Mertens protection: ABSENT)")
print(f"{'Delay':>8}  {'tau':>6}  {'Raw-Prime':>12}  {'Matched-Rand':>14}  {'Revival':>10}")
raw_results = []
for delay in DELAYS_NS:
    tau = -delay / 100.0
    C_rp,  _, _, _ = get_C("raw_prime", delay)
    C_rm,  _, _, _ = get_C("raw_rand_matched", delay)
    revival = C_rp - C_rm
    print(f"{delay:>8}ns  {tau:>+6.2f}  {C_rp:>12.4f}  {C_rm:>14.4f}  {revival:>+10.4f}")
    raw_results.append({"delay": delay, "tau": tau,
                        "C_raw_prime": float(C_rp), "C_raw_rand_matched": float(C_rm),
                        "revival": float(revival)})

# Critical verdict
print("\n" + "=" * 65)
print("CRITICAL VERDICT")
print("=" * 65)
norm_revival_0  = norm_results[0]["revival"]
norm_sigma_0    = norm_results[0]["sigma"]
raw_revival_0   = raw_results[0]["revival"]

if norm_revival_0 > 0 and norm_sigma_0 > 2:
    verdict = "PRIME STRUCTURE IS PROTECTIVE (independent of Mertens alignment)"
    conclusion = ("The prime weight distribution ln(p)/p is intrinsically decoherence-resistant. "
                  "This supports the PFBMW framework's claim that the prime-indexed fractal "
                  "environment has physical significance beyond number-theoretic coincidence.")
elif norm_revival_0 > 0 and norm_sigma_0 > 0:
    verdict = "WEAK PRIME ADVANTAGE (inconclusive -- needs more shots or qubits)"
    conclusion = ("Positive direction but below 2σ significance. Increase SHOTS to 8192 "
                  "or extend to 7-qubit GHZ (p=2,3,5,7,11,13,17) to amplify the signal.")
else:
    verdict = "NO PRIME STRUCTURAL ADVANTAGE (Mertens was the whole story in Exp 2)"
    conclusion = ("The Experiment 2 result was due to Mertens alignment (3-prime sum ≈ 2π), "
                  "not intrinsic prime structure. Pivot: the framework's prediction for GHZ "
                  "coherence needs to be reframed around the Mertens property specifically, "
                  "not general prime weighting.")

print(f"\nVerdict: {verdict}")
print(f"\nConclusion: {conclusion}")

best_tau_row = max(norm_results, key=lambda r: r["revival"])
print(f"\nBest τ for normalized prime: {best_tau_row['tau']:+.2f} (delay={best_tau_row['delay']}ns)")
print(f"  revival = {best_tau_row['revival']:+.4f}  sigma = {best_tau_row['sigma']:+.1f}")


# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────
print("\nGenerating figures...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f"Experiment 3: 5-Qubit GHZ Critical Test | {backend.name} | "
             f"{'Simulator' if USE_SIMULATOR else 'Real Hardware'}",
             fontsize=12, fontweight='bold')

# Plot 1: Normalized prime vs random across delays
ax = axes[0]
delays_arr = [r["delay"] for r in norm_results]
C_np_arr   = [r["C_norm_prime"] for r in norm_results]
C_nr_arr   = [r["C_norm_rand_mean"] for r in norm_results]
C_nr_std   = [r["C_norm_rand_std"] for r in norm_results]
tau_arr    = [r["tau"] for r in norm_results]

ax.errorbar(tau_arr, C_nr_arr, yerr=C_nr_std, fmt='s--', color='gray',
            capsize=5, lw=1.5, ms=7, label='Norm-random (mean±std)')
ax.plot(tau_arr, C_np_arr, 'o-', color='steelblue', lw=2, ms=9,
        label='Norm-prime (PFBMW)')
ax.axhline(P_even_norm_ideal, color='green', linestyle=':', lw=1, label='Ideal (1.0)')
ax.axhline(0.5, color='lightgray', linestyle='--', lw=0.8, label='Classical limit')
ax.set_xlabel('τ  (= −delay/100)')
ax.set_ylabel('GHZ Parity Coherence  C')
ax.set_title('Normalized Phases\n(Pure structural test)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.25)
ax.set_ylim(-0.05, 1.1)

# Plot 2: Raw prime vs matched random
ax = axes[1]
C_rp_arr = [r["C_raw_prime"] for r in raw_results]
C_rm_arr = [r["C_raw_rand_matched"] for r in raw_results]
x = np.arange(len(DELAYS_NS))
w = 0.3
bars_p = ax.bar(x - w/2, C_rp_arr, w, color='steelblue', label='Raw prime', alpha=0.85)
bars_r = ax.bar(x + w/2, C_rm_arr, w, color='gray', label='Raw rand (matched Φ)', alpha=0.85)
ax.axhline(P_even_raw_ideal, color='green', linestyle=':', lw=1.5,
           label=f'Raw ideal = {P_even_raw_ideal:.3f}')
for bar, v in zip(bars_p, C_rp_arr):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}',
            ha='center', va='bottom', fontsize=8)
for bar, v in zip(bars_r, C_rm_arr):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}',
            ha='center', va='bottom', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([f'τ={r["tau"]:+.2f}\n({r["delay"]}ns)' for r in raw_results])
ax.set_ylabel('GHZ Parity Coherence  C')
ax.set_title(f'Raw Phases (Mertens absent)\nIdeal = {P_even_raw_ideal:.3f}')
ax.legend(fontsize=8)
ax.grid(True, axis='y', alpha=0.25)

# Plot 3: Revival comparison and verdict
ax = axes[2]
norm_revivals = [r["revival"] for r in norm_results]
raw_revivals  = [r["revival"] for r in raw_results]
x = np.arange(len(DELAYS_NS))
ax.bar(x - 0.2, norm_revivals, 0.35, label='Norm-prime − Norm-rand',
       color=['steelblue' if r > 0 else 'tomato' for r in norm_revivals], alpha=0.85)
ax.bar(x + 0.2, raw_revivals, 0.35, label='Raw-prime − Raw-matched',
       color=['lightblue' if r > 0 else 'lightsalmon' for r in raw_revivals], alpha=0.85)
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f'τ={r["tau"]:+.2f}' for r in norm_results])
ax.set_ylabel('Revival (prime − random)')
ax.set_title('Revival by τ\n(positive = prime wins)')
ax.legend(fontsize=8)
ax.grid(True, axis='y', alpha=0.25)
ax.text(0.5, 0.97, f'Verdict: {verdict[:35]}...', transform=ax.transAxes,
        ha='center', va='top', fontsize=7, style='italic', wrap=True)

plt.tight_layout()
fig_path = 'Experiment3_5qubit_Results.png'
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
print(f"Figure saved: {fig_path}")


# ─────────────────────────────────────────────────────────────
# SAVE JSON
# ─────────────────────────────────────────────────────────────
output = {
    "timestamp":      datetime.now().isoformat(),
    "backend":        backend.name,
    "mode":           "simulator" if USE_SIMULATOR else "real_hardware",
    "shots":          SHOTS,
    "primes":         PRIMES,
    "n_qubits":       N_QUBITS,
    "raw_phases":     [float(p) for p in raw_phases],
    "norm_phases":    [float(p) for p in norm_phases],
    "Phi_raw":        float(Phi_raw),
    "P_even_raw_ideal": float(P_even_raw_ideal),
    "job_id":         job.job_id() if not USE_SIMULATOR else "simulator",
    "normalized_test": norm_results,
    "raw_test":        raw_results,
    "verdict":         verdict,
    "conclusion":      conclusion,
    "exp2_comparison": {
        "note": "Exp2 revival=+0.569 (3q, Mertens-aligned). Exp3 tests whether this survives without alignment.",
        "exp2_tau_star": -0.50,
        "exp3_best_tau": float(best_tau_row["tau"]),
    }
}

json_path = 'Experiment3_5qubit_Results.json'
with open(json_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Results saved: {json_path}")
print("=" * 65)
print("Experiment 3 complete.")
print("=" * 65)
