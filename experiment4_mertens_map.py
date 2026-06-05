"""
Experiment 4: The Mertens Mechanism Map
========================================
Author: Earl Decker — PFBMW Framework, June 2026

BACKGROUND
----------
Experiments 2 and 3 established:
  - Prime-weighted 3-qubit GHZ: +74.6σ revival vs random (Exp 2)
  - This advantage is driven by Mertens alignment (Φ ≈ 2π), not prime structure (Exp 3)
  - When ALL conditions normalized to Φ=2π, prime ≈ random ≈ 0.92

This experiment maps the mechanism directly.

TWO TESTS IN ONE JOB
--------------------

TEST A — Phase Distribution at Fixed Φ=2π (3-qubit)
  Question: when Φ=2π is fixed, does the DISTRIBUTION of individual phases matter?
  Conditions (all Φ=2π):
    1. norm_prime       φ_k = ln(p_k)/p_k × 2π, scaled to Φ=2π
    2. natural_246      φ_k = ln(k)/k × 2π for k={2,4,6}, naturally Φ≈2π (non-prime)
    3. uniform          all φ_k = 2π/3 (maximum symmetry)
    4. norm_rand ×3     random phases scaled to Φ=2π (3 seeds)

  Prediction if distribution is irrelevant: all ~0.92 (same as Exp 3)
  Prediction if distribution matters: deviations visible

TEST B — α-Scan: Mapping cos²(Φ/2) on Real Hardware (3-qubit)
  Question: does P(even) trace cos²(Φ/2) regardless of whether phases are prime or random?
  Method: scale prime_natural phases by α ∈ [0.24, 0.48, 0.73, 0.97, 1.21, 1.45, 1.69, 1.94]
          same for random_matched (same Φ_natural, different distribution)

  Φ = α × Φ_prime_natural = α × 6.501 rad
  At α ≈ 0.48: Φ ≈ π   → P(even) → 0  (minimum)
  At α ≈ 0.97: Φ ≈ 2π  → P(even) → 1  (Mertens resonance 1)
  At α ≈ 1.45: Φ ≈ 3π  → P(even) → 0  (second minimum)
  At α ≈ 1.94: Φ ≈ 4π  → P(even) → 1  (Mertens resonance 2)

  If both prime and random trace the same cos²(Φ/2) curve → phase sum is the only variable
  If prime curve differs from random → something about prime distribution still matters

TOTAL: 6 (TEST A) + 16 (TEST B) = 22 PUBs
"""

import numpy as np
import json
from datetime import datetime
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
IBM_TOKEN = "96axnVJAp_PkXhi7mpX8t_CVj1NtzqmHjaApLQ5Pn96Q"
IBM_INSTANCE = "crn:v1:bluemix:public:quantum-computing:us-east:a/8420df4c778d45e59489b345c26d2c81:73caf2a1-d677-4d15-b711-f8f3fc73732a::"

USE_SIMULATOR = False   # Set True to test locally with Aer
SHOTS         = 4096
N_QUBITS      = 3
PRIMES        = [2, 3, 5]
RAND_SEEDS_A  = [42, 137, 271]      # 3 seeds for TEST A
RAND_SEED_B   = 999                 # fixed seed for TEST B matched random

# Alpha values for TEST B — chosen to hit π/2 increments of Φ
# Φ_prime_natural = 6.501, period = 2π = 6.283
# α to hit Φ = π/2, π, 3π/2, 2π, 5π/2, 3π, 7π/2, 4π:
ALPHA_VALUES = [
    np.pi/2   / 6.501,   # α=0.242  Φ=π/2   P_ideal=0.50
    np.pi     / 6.501,   # α=0.484  Φ=π     P_ideal=0.00
    3*np.pi/2 / 6.501,   # α=0.726  Φ=3π/2  P_ideal=0.50
    2*np.pi   / 6.501,   # α=0.968  Φ≈2π    P_ideal=1.00  ← Mertens resonance 1
    5*np.pi/2 / 6.501,   # α=1.210  Φ=5π/2  P_ideal=0.50
    3*np.pi   / 6.501,   # α=1.452  Φ=3π    P_ideal=0.00
    7*np.pi/2 / 6.501,   # α=1.694  Φ=7π/2  P_ideal=0.50
    4*np.pi   / 6.501,   # α=1.936  Φ≈4π    P_ideal=1.00  ← Mertens resonance 2
]

# ─────────────────────────────────────────────────────────────
# PHASE SETS
# ─────────────────────────────────────────────────────────────
# Prime natural phases (raw, not normalized)
prime_natural = np.array([np.log(p) / p * 2 * np.pi for p in PRIMES])
Phi_prime     = prime_natural.sum()
P_ideal_prime = np.cos(Phi_prime / 2) ** 2

# Natural 246 phases: ln(k)/k × 2π for k={2,4,6}  — non-prime, naturally Φ≈2π
natural_246   = np.array([np.log(k) / k * 2 * np.pi for k in [2, 4, 6]])
Phi_246       = natural_246.sum()
P_ideal_246   = np.cos(Phi_246 / 2) ** 2

# Normalized prime (exact Φ=2π)
norm_prime    = prime_natural / Phi_prime * 2 * np.pi
P_ideal_norm  = np.cos(np.pi) ** 2   # = 1.0

# Uniform (all equal, exact Φ=2π)
uniform       = np.array([2 * np.pi / N_QUBITS] * N_QUBITS)

# Matched random for TEST B: random phases, sum matched to Phi_prime
rng_b         = np.random.default_rng(RAND_SEED_B)
rand_b_raw    = rng_b.uniform(0, 2 * np.pi, N_QUBITS)
rand_b_matched = rand_b_raw / rand_b_raw.sum() * Phi_prime   # same Φ, different distribution

print("=" * 65)
print("EXPERIMENT 4: THE MERTENS MECHANISM MAP")
print("=" * 65)
print(f"\nPRIME NATURAL PHASES (p=2,3,5):")
print(f"  φ = {np.round(prime_natural, 4)}")
print(f"  Φ = {Phi_prime:.4f} rad  (2π = {2*np.pi:.4f}, diff = {Phi_prime - 2*np.pi:+.4f})")
print(f"  Ideal P(even) = {P_ideal_prime:.4f}")

print(f"\nNATURAL 246 PHASES (k=2,4,6):")
print(f"  φ = {np.round(natural_246, 4)}")
print(f"  Φ = {Phi_246:.4f} rad  (diff from 2π = {Phi_246 - 2*np.pi:+.4f})")
print(f"  Ideal P(even) = {P_ideal_246:.4f}")

print(f"\nNORMALIZED PRIME (Φ=2π exactly):")
print(f"  φ = {np.round(norm_prime, 4)}")

print(f"\nUNIFORM (Φ=2π, all equal):")
print(f"  φ = {np.round(uniform, 4)}")

print(f"\nTEST B ALPHA VALUES:")
for a in ALPHA_VALUES:
    phi_a = a * Phi_prime
    p_a   = np.cos(phi_a / 2) ** 2
    print(f"  α={a:.3f}  Φ={phi_a:.3f}  Φ/(2π)={phi_a/(2*np.pi):.3f}  P_ideal={p_a:.3f}")

# ─────────────────────────────────────────────────────────────
# CIRCUIT BUILDER
# ─────────────────────────────────────────────────────────────
def ghz_circuit(phases, label=""):
    """3-qubit GHZ with RZ phase kicks, measured in parity (H-rotated) basis."""
    n = len(phases)
    qc = QuantumCircuit(n, n, name=label)
    qc.h(0)
    for q in range(n - 1):
        qc.cx(q, q + 1)
    for q, phi in enumerate(phases):
        qc.rz(float(phi), q)
    for q in range(n):
        qc.h(q)
    qc.measure(range(n), range(n))
    return qc

def parity_even(counts):
    total = sum(counts.values())
    even  = sum(v for k, v in counts.items() if k.count('1') % 2 == 0)
    return even / total

# ─────────────────────────────────────────────────────────────
# BUILD CIRCUITS
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("BUILDING CIRCUITS")
print("─" * 65)

all_circuits = []
circuit_meta = []   # (test, label, phases, alpha, Phi_total, P_ideal)

# ── TEST A: phase distribution at Φ=2π ───────────────────────
print("\nTEST A — Phase Distribution (Φ=2π fixed):")

# A1: normalized prime
qc = ghz_circuit(norm_prime, "A_norm_prime")
all_circuits.append(qc)
circuit_meta.append(("A", "norm_prime", norm_prime.tolist(), 1.0, float(norm_prime.sum()), 1.0))
print(f"  [A1] norm_prime        Φ={norm_prime.sum():.4f}  P_ideal=1.000")

# A2: natural_246 (non-prime, naturally aligned)
qc = ghz_circuit(natural_246, "A_natural_246")
all_circuits.append(qc)
circuit_meta.append(("A", "natural_246", natural_246.tolist(), 1.0, float(Phi_246), float(P_ideal_246)))
print(f"  [A2] natural_246       Φ={Phi_246:.4f}  P_ideal={P_ideal_246:.4f}")

# A3: uniform
qc = ghz_circuit(uniform, "A_uniform")
all_circuits.append(qc)
circuit_meta.append(("A", "uniform", uniform.tolist(), 1.0, float(uniform.sum()), 1.0))
print(f"  [A3] uniform           Φ={uniform.sum():.4f}  P_ideal=1.000")

# A4-A6: normalized random (3 seeds)
for i, seed in enumerate(RAND_SEEDS_A):
    rng_a = np.random.default_rng(seed)
    raw   = rng_a.uniform(0, 2 * np.pi, N_QUBITS)
    nr    = raw / raw.sum() * 2 * np.pi
    qc    = ghz_circuit(nr, f"A_norm_rand_{seed}")
    all_circuits.append(qc)
    circuit_meta.append(("A", f"norm_rand_{seed}", nr.tolist(), 1.0, float(nr.sum()), 1.0))
    print(f"  [A{4+i}] norm_rand_{seed}   Φ={nr.sum():.4f}  P_ideal=1.000")

# ── TEST B: alpha scan ────────────────────────────────────────
print(f"\nTEST B — α-Scan (prime_natural × α  vs  rand_matched × α):")

for i, alpha in enumerate(ALPHA_VALUES):
    # Prime scaled
    phases_p  = prime_natural * alpha
    Phi_p     = float(phases_p.sum())
    P_p       = float(np.cos(Phi_p / 2) ** 2)
    qc = ghz_circuit(phases_p, f"B_prime_a{i}")
    all_circuits.append(qc)
    circuit_meta.append(("B", f"prime_a{i}", phases_p.tolist(), float(alpha), Phi_p, P_p))

    # Random matched (same Φ, different distribution)
    phases_r  = rand_b_matched * alpha
    Phi_r     = float(phases_r.sum())
    P_r       = float(np.cos(Phi_r / 2) ** 2)
    qc = ghz_circuit(phases_r, f"B_rand_a{i}")
    all_circuits.append(qc)
    circuit_meta.append(("B", f"rand_a{i}", phases_r.tolist(), float(alpha), Phi_r, P_r))

    print(f"  α={alpha:.3f}  Φ={Phi_p:.3f}  P_ideal={P_p:.3f}  [prime+rand PUBs {6+2*i},{6+2*i+1}]")

n_total = len(all_circuits)
print(f"\nTotal circuits: {n_total}  (TEST A: 6,  TEST B: {n_total-6})")

# ─────────────────────────────────────────────────────────────
# CONNECT AND RUN
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("CONNECTING TO IBM QUANTUM")
print("─" * 65)

if USE_SIMULATOR:
    from qiskit_aer import AerSimulator
    backend = AerSimulator()
    print("Using Aer simulator.")
else:
    service = QiskitRuntimeService(
        channel='ibm_quantum_platform',
        token=IBM_TOKEN,
        instance=IBM_INSTANCE,
    )
    backend = service.least_busy(min_num_qubits=N_QUBITS, simulator=False)
    print(f"Backend: {backend.name}")

print("\nTranspiling...")
t_circuits = transpile(all_circuits, backend, optimization_level=3)
depths = [t.depth() for t in t_circuits]
print(f"  Depths: min={min(depths)}, max={max(depths)}, mean={np.mean(depths):.1f}")

print("Running...")
if USE_SIMULATOR:
    job = backend.run(t_circuits, shots=SHOTS)
    raw_result = job.result()
    all_counts = [raw_result.get_counts(i) for i in range(n_total)]
else:
    sampler = Sampler(mode=backend)
    job = sampler.run(t_circuits, shots=SHOTS)
    print(f"  Job ID: {job.job_id()}")
    print("  Waiting for results...")
    raw_result = job.result()
    all_counts = [raw_result[i].data.c.get_counts() for i in range(n_total)]

print("  Done.")

# ─────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("ANALYSIS — TEST A: Phase Distribution at Φ=2π")
print("=" * 65)

results_A = []
for i, (test, label, phases, alpha, Phi_total, P_ideal) in enumerate(circuit_meta):
    if test != "A":
        continue
    C = parity_even(all_counts[i])
    hw_deg = P_ideal - C
    print(f"  {label:20s}  Φ={Phi_total:.4f}  P_ideal={P_ideal:.4f}  P_hw={C:.4f}  degradation={hw_deg:+.4f}")
    results_A.append({"label": label, "Phi": Phi_total, "P_ideal": P_ideal, "P_hw": float(C)})

# Check if all normalized conditions are equal
norm_vals = [r["P_hw"] for r in results_A if r["label"] in ("norm_prime", "uniform") or "norm_rand" in r["label"]]
nat_246   = next(r["P_hw"] for r in results_A if r["label"] == "natural_246")
print(f"\n  Normalized conditions mean: {np.mean(norm_vals):.4f}  std: {np.std(norm_vals):.4f}")
print(f"  natural_246 (non-prime):    {nat_246:.4f}")
print(f"  natural_246 vs norm mean:   {nat_246 - np.mean(norm_vals):+.4f}")

print("\n" + "=" * 65)
print("ANALYSIS — TEST B: α-Scan (Mertens Mechanism)")
print("=" * 65)
print(f"  {'α':>6}  {'Φ':>7}  {'P_ideal':>9}  {'P_prime':>9}  {'P_rand':>9}  {'diff':>8}")

results_B = []
b_indices = [(i, meta) for i, meta in enumerate(circuit_meta) if meta[0] == "B"]
# Pair up prime and random
for j in range(0, len(b_indices), 2):
    i_p, (_, lp, phases_p, alpha, Phi_p, P_p) = b_indices[j]
    i_r, (_, lr, phases_r, alpha_r, Phi_r, P_r) = b_indices[j+1]
    C_p = parity_even(all_counts[i_p])
    C_r = parity_even(all_counts[i_r])
    diff = C_p - C_r
    print(f"  {alpha:>6.3f}  {Phi_p:>7.3f}  {P_p:>9.4f}  {C_p:>9.4f}  {C_r:>9.4f}  {diff:>+8.4f}")
    results_B.append({
        "alpha": float(alpha), "Phi": float(Phi_p), "P_ideal": float(P_p),
        "P_prime_hw": float(C_p), "P_rand_hw": float(C_r), "diff": float(diff),
    })

# Summary statistics for TEST B
diffs = [r["diff"] for r in results_B]
print(f"\n  Mean diff (prime - rand): {np.mean(diffs):+.4f}")
print(f"  Std of diffs:             {np.std(diffs):.4f}")
print(f"  Max |diff|:               {max(abs(d) for d in diffs):.4f}")

# Check if hardware curve tracks ideal cos²(Φ/2)
P_ideal_vals = [r["P_ideal"] for r in results_B]
P_prime_vals = [r["P_prime_hw"] for r in results_B]
P_rand_vals  = [r["P_rand_hw"] for r in results_B]
corr_prime   = float(np.corrcoef(P_ideal_vals, P_prime_vals)[0, 1])
corr_rand    = float(np.corrcoef(P_ideal_vals, P_rand_vals)[0, 1])
print(f"\n  Correlation with ideal cos²(Φ/2):")
print(f"    Prime curve:  r = {corr_prime:.4f}")
print(f"    Random curve: r = {corr_rand:.4f}")

# ─────────────────────────────────────────────────────────────
# VERDICT
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("VERDICT")
print("=" * 65)

# TEST A verdict
std_A = np.std([r["P_hw"] for r in results_A])
print(f"\nTEST A (phase distribution):")
if std_A < 0.02:
    verdict_A = "DISTRIBUTION IRRELEVANT — only total Φ matters"
elif std_A < 0.05:
    verdict_A = "WEAK DISTRIBUTION EFFECT — small but present"
else:
    verdict_A = "DISTRIBUTION MATTERS — phase spread affects coherence"
print(f"  Std across all conditions: {std_A:.4f}")
print(f"  Verdict: {verdict_A}")

# TEST B verdict
print(f"\nTEST B (α-scan mechanism):")
if corr_prime > 0.85 and corr_rand > 0.85:
    verdict_B = "BOTH prime and random track cos²(Φ/2) — phase SUM is the complete mechanism"
    if abs(np.mean(diffs)) < 0.03:
        verdict_B += " — distribution makes no difference"
    else:
        verdict_B += f" — but prime differs from random by {np.mean(diffs):+.3f} on average"
elif corr_prime > 0.85:
    verdict_B = "PRIME tracks cos²(Φ/2) but random does not — prime structure still matters"
else:
    verdict_B = "NEITHER cleanly tracks cos²(Φ/2) — hardware effects dominate"
print(f"  Prime r={corr_prime:.3f}, Random r={corr_rand:.3f}")
print(f"  Verdict: {verdict_B}")

# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────
output = {
    "timestamp":      datetime.now().isoformat(),
    "backend":        backend.name if not USE_SIMULATOR else "aer_simulator",
    "mode":           "simulator" if USE_SIMULATOR else "real_hardware",
    "shots":          SHOTS,
    "job_id":         job.job_id() if not USE_SIMULATOR else "simulator",
    "primes":         PRIMES,
    "prime_natural":  [float(x) for x in prime_natural],
    "Phi_prime":      float(Phi_prime),
    "natural_246":    [float(x) for x in natural_246],
    "Phi_246":        float(Phi_246),
    "alpha_values":   [float(a) for a in ALPHA_VALUES],
    "rand_b_seed":    RAND_SEED_B,
    "rand_b_matched": [float(x) for x in rand_b_matched],
    "results_A":      results_A,
    "results_B":      results_B,
    "verdict_A":      verdict_A,
    "verdict_B":      verdict_B,
    "std_A":          float(std_A),
    "corr_prime_B":   float(corr_prime),
    "corr_rand_B":    float(corr_rand),
    "mean_diff_B":    float(np.mean(diffs)),
}

out_file = "Experiment4_Mertens_Map_Results.json"
with open(out_file, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved: {out_file}")
print("=" * 65)
print("Experiment 4 complete.")
print("=" * 65)
                                                                                                                                                                                 