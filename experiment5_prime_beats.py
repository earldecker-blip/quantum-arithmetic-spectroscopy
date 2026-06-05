"""
Experiment 5: Prime Atomic Beats
=================================
Treats primes as physical atoms with fixed intrinsic frequencies.
Each prime p gets frequency f_p. The quantum state is a Bell pair
|Psi+> = (|01> + |10>) / sqrt(2), whose XX correlator measures
frequency DIFFERENCES: <X1 X2> = cos((f1 - f2) * alpha).

This is the first PFBMW experiment that can distinguish individual
prime frequencies — GHZ only senses the sum, Bell pairs sense the gap.

HYPOTHESES
----------
H1 (Prime Beats):  Prime frequency gaps produce slower decoherence
                   than matched non-prime gaps because prime gaps are
                   coprime / maximally non-degenerate.
H2 (Frequency Model): The Mertens model f_p = ln(p)/p should outperform
                       raw f_p = p or harmonic f_p = 1/p as a predictor
                       of coherence lifetime.
H3 (GHZ reference):  GHZ with same primes should track cos^2(Phi/2)
                      as established in Exp 4, providing a sanity check.

CIRCUIT DESIGN
--------------
Bell pair |Psi+> = (|01>+|10>)/sqrt(2):
  H(q0) -> CX(q0,q1) -> X(q1)

Phase kick (frequency assignment):
  RZ(2*pi*f1*alpha, q0)   # qubit 0 carries prime p1
  RZ(2*pi*f2*alpha, q1)   # qubit 1 carries prime p2

Measure in X basis:
  H(q0), H(q1) -> measure

XX parity = cos((f1-f2)*alpha)   [theoretical prediction]

CONDITIONS
----------
Frequency models tested:
  mertens:   f_p = ln(p)/p     (PFBMW weight, Exp 2-4 connection)
  primon:    f_p = ln(p)       (primon gas, quantum field theory link)
  raw:       f_p = p           (raw prime value)
  harmonic:  f_p = 1/p         (harmonic series)

Prime pairs and their gaps:
  (p=2, p=3) -> gaps: mertens=|ln2/2 - ln3/3|, primon=|ln2-ln3|, raw=1, harm=1/6
  (p=2, p=5) -> gaps: model-dependent
  (p=3, p=5) -> gaps: model-dependent

Non-prime controls (same raw gap as prime pair):
  prime (2,3) raw gap=1 -> control: (4,5) raw gap=1
  prime (2,5) raw gap=3 -> control: (1,4) raw gap=3  [or (6,9)]
  prime (3,5) raw gap=2 -> control: (4,6) raw gap=2

Alpha sweep: 8 values from 0 to 2*pi (full period of cos)
Decoherence scan: 4 delay values at fixed alpha=pi for select conditions

Total PUBs:
  Bell alpha sweep: 6 conditions * 4 models * 8 alphas = 192... too many.
  Simplified: use mertens model only for alpha sweep (most scientifically
  motivated), add raw model for comparison = 2 models * 6 conditions * 8 alphas = 96
  Decoherence: 6 conditions * 4 delays = 24
  GHZ reference (3-qubit, primes 2,3,5): 8 alpha values = 8
  Total: ~128 PUBs at 4096 shots = ~524k shots. Feasible.

  Further simplified for first run:
  - mertens model only (primary hypothesis)
  - 6 conditions (3 prime pairs + 3 matched controls)
  - 8 alpha values
  - 4 delays for decoherence test (alpha=pi only)
  - GHZ reference: 8 alpha values
  Total: 6*8 + 6*4 + 8 = 48 + 24 + 8 = 80 PUBs
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.circuit.library import RZGate
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

# ── Credentials ──────────────────────────────────────────────────────────────
TOKEN    = "96axnVJAp_PkXhi7mpX8t_CVj1NtzqmHjaApLQ5Pn96Q"
INSTANCE = "crn:v1:bluemix:public:quantum-computing:us-east:a/8420df4c778d45e59489b345c26d2c81:73caf2a1-d677-4d15-b711-f8f3fc73732a::"
CHANNEL  = "ibm_quantum_platform"

SHOTS    = 4096
DRY_RUN  = False   # set True to build circuits only, no submission

# ── Frequency Models ──────────────────────────────────────────────────────────
def freq_mertens(p):
    """f_p = ln(p)/p  — PFBMW weight connecting to Mertens theorem"""
    return np.log(p) / p

def freq_primon(p):
    """f_p = ln(p)  — primon gas energy level, zeta function partition function"""
    return np.log(p)

def freq_raw(p):
    """f_p = p  — raw prime value"""
    return float(p)

def freq_harmonic(p):
    """f_p = 1/p  — harmonic series"""
    return 1.0 / p

# ── Experiment Parameters ─────────────────────────────────────────────────────

# Primary model for this run
FREQ_MODEL = freq_mertens
MODEL_NAME = "mertens"

# Prime pairs (p1, p2) and matched non-prime controls with same raw gap
CONDITIONS = [
    # label,          p1,  p2,  is_prime_pair
    ("prime_23",       2,   3,   True),
    ("prime_25",       2,   5,   True),
    ("prime_35",       3,   5,   True),
    ("control_45",     4,   5,   False),   # raw gap 1, matches (2,3)
    ("control_46",     4,   6,   False),   # raw gap 2, matches (3,5)
    ("control_69",     6,   9,   False),   # raw gap 3, matches (2,5)
]

# Alpha sweep: 0 to 2*pi in 8 steps
ALPHA_SWEEP = np.linspace(0, 2 * np.pi, 8, endpoint=False)

# Delays for decoherence test (in seconds, IBM runtime uses seconds)
# 0, 50 ns, 125 ns, 250 ns
DELAYS_NS = [0, 50e-9, 125e-9, 250e-9]
DECO_ALPHA = np.pi   # fixed alpha for decoherence scan

# GHZ reference primes (reproduces Exp 4 setup)
GHZ_PRIMES = [2, 3, 5]

# ── Circuit Builders ──────────────────────────────────────────────────────────

def build_bell_alpha_circuit(f1, f2, alpha, label=""):
    """
    Bell pair |Psi+> XX correlator circuit.
    Theory: P(even parity) = cos^2((f1-f2)*alpha / 2)... actually:
    <X1 X2> = cos((f1-f2)*alpha)
    P(even) = (1 + cos((f1-f2)*alpha)) / 2

    Steps:
      1. Prepare |Psi+> = (|01>+|10>)/sqrt(2)
         H(q0), CX(q0,q1), X(q1)
      2. Phase kick: RZ(2*pi*f1*alpha, q0), RZ(2*pi*f2*alpha, q1)
      3. Measure X basis: H(q0), H(q1), measure
    """
    qr = QuantumRegister(2, 'q')
    cr = ClassicalRegister(2, 'c')
    qc = QuantumCircuit(qr, cr, name=label or f"bell_f{f1:.3f}_f{f2:.3f}_a{alpha:.3f}")

    # Prepare |Psi+>
    qc.h(qr[0])
    qc.cx(qr[0], qr[1])
    qc.x(qr[1])

    # Phase kicks — frequency assignment
    phase1 = 2 * np.pi * f1 * alpha
    phase2 = 2 * np.pi * f2 * alpha
    qc.rz(phase1, qr[0])
    qc.rz(phase2, qr[1])

    # Rotate to X basis
    qc.h(qr[0])
    qc.h(qr[1])

    qc.measure(qr, cr)
    return qc


def build_bell_delay_circuit(f1, f2, alpha, delay_s, label="", backend_dt=None):
    """
    Bell pair circuit with delay inserted after state preparation.
    delay_s: delay in seconds.
    backend_dt: backend.dt in seconds (e.g. 2.222e-10 for ibm_marrakesh).
                If provided, delay is expressed in integer dt units (required by
                most IBM backends). If None, falls back to unit='s' (may warn).
    """
    qr = QuantumRegister(2, 'q')
    cr = ClassicalRegister(2, 'c')
    qc = QuantumCircuit(qr, cr, name=label or f"bell_deco_d{int(delay_s*1e9)}ns")

    # Prepare |Psi+>
    qc.h(qr[0])
    qc.cx(qr[0], qr[1])
    qc.x(qr[1])

    # Delay (decoherence window)
    if delay_s > 0:
        if backend_dt is not None:
            # Express as integer number of dt ticks (required by IBM hardware)
            delay_dt = int(round(delay_s / backend_dt))
            qc.delay(delay_dt, qr[0], unit='dt')
            qc.delay(delay_dt, qr[1], unit='dt')
        else:
            # Fallback: seconds — transpiler will convert, but may emit a warning
            qc.delay(delay_s, qr[0], unit='s')
            qc.delay(delay_s, qr[1], unit='s')

    # Phase kicks
    phase1 = 2 * np.pi * f1 * alpha
    phase2 = 2 * np.pi * f2 * alpha
    qc.rz(phase1, qr[0])
    qc.rz(phase2, qr[1])

    # X basis measurement
    qc.h(qr[0])
    qc.h(qr[1])
    qc.measure(qr, cr)
    return qc


def build_ghz_alpha_circuit(primes, alpha, freq_model, label=""):
    """
    GHZ reference: n-qubit GHZ with phase sum Phi = 2*pi * sum(f_p) * alpha.
    Theory: P(even) = cos^2(Phi/2)   [reproduces Exp 4]
    """
    n = len(primes)
    qr = QuantumRegister(n, 'q')
    cr = ClassicalRegister(n, 'c')
    qc = QuantumCircuit(qr, cr, name=label or f"ghz_a{alpha:.3f}")

    # Prepare GHZ
    qc.h(qr[0])
    for i in range(n - 1):
        qc.cx(qr[i], qr[i + 1])

    # Phase kick each qubit with its prime frequency
    for i, p in enumerate(primes):
        phase = 2 * np.pi * freq_model(p) * alpha
        qc.rz(phase, qr[i])

    # Rotate to X basis
    for i in range(n):
        qc.h(qr[i])

    qc.measure(qr, cr)
    return qc

# ── Build All PUBs ────────────────────────────────────────────────────────────

def build_all_pubs(backend_dt=None):
    """
    Returns list of (circuit, shots, label, metadata) tuples.
    backend_dt: backend.dt in seconds; used to convert delays to integer dt ticks.
    """
    pubs = []

    # === Block 1: Bell alpha sweep ===
    print("Building Bell alpha sweep circuits...")
    for label, p1, p2, is_prime in CONDITIONS:
        f1 = FREQ_MODEL(p1)
        f2 = FREQ_MODEL(p2)
        gap = abs(f1 - f2)
        for alpha in ALPHA_SWEEP:
            circ_label = f"bell_alpha_{label}_a{alpha:.4f}"
            qc = build_bell_alpha_circuit(f1, f2, alpha, label=circ_label)
            meta = {
                "type": "bell_alpha",
                "condition": label,
                "p1": p1, "p2": p2,
                "f1": f1, "f2": f2,
                "gap": gap,
                "alpha": alpha,
                "is_prime": is_prime,
                "model": MODEL_NAME,
                "theory_p_even": 0.5 * (1 + np.cos((f1 - f2) * alpha)),
            }
            pubs.append((qc, SHOTS, circ_label, meta))

    print(f"  -> {len(pubs)} PUBs after Bell sweep")

    # === Block 2: Bell decoherence scan ===
    print("Building Bell decoherence circuits...")
    deco_start = len(pubs)
    for label, p1, p2, is_prime in CONDITIONS:
        f1 = FREQ_MODEL(p1)
        f2 = FREQ_MODEL(p2)
        for delay_s in DELAYS_NS:
            circ_label = f"bell_deco_{label}_d{int(delay_s*1e9)}ns"
            qc = build_bell_delay_circuit(f1, f2, DECO_ALPHA, delay_s,
                                          label=circ_label, backend_dt=backend_dt)
            meta = {
                "type": "bell_deco",
                "condition": label,
                "p1": p1, "p2": p2,
                "f1": f1, "f2": f2,
                "gap": abs(f1 - f2),
                "alpha": DECO_ALPHA,
                "delay_ns": delay_s * 1e9,
                "is_prime": is_prime,
                "model": MODEL_NAME,
            }
            pubs.append((qc, SHOTS, circ_label, meta))
    print(f"  -> {len(pubs) - deco_start} decoherence PUBs")

    # === Block 3: GHZ reference ===
    print("Building GHZ reference circuits...")
    ghz_start = len(pubs)
    for alpha in ALPHA_SWEEP:
        circ_label = f"ghz_ref_a{alpha:.4f}"
        qc = build_ghz_alpha_circuit(GHZ_PRIMES, alpha, FREQ_MODEL, label=circ_label)
        phi = 2 * np.pi * sum(FREQ_MODEL(p) for p in GHZ_PRIMES) * alpha
        meta = {
            "type": "ghz_reference",
            "primes": GHZ_PRIMES,
            "alpha": alpha,
            "phi": phi,
            "model": MODEL_NAME,
            "theory_p_even": np.cos(phi / 2) ** 2,
        }
        pubs.append((qc, SHOTS, circ_label, meta))
    print(f"  -> {len(pubs) - ghz_start} GHZ reference PUBs")

    print(f"\nTotal PUBs: {len(pubs)}")
    return pubs


# ── Analysis Helpers ──────────────────────────────────────────────────────────

def p_even_from_counts(counts, num_bits):
    """Compute P(even parity) from a counts dict {bitstring: count}."""
    total = sum(counts.values())
    even = sum(v for k, v in counts.items() if k.count('1') % 2 == 0)
    return even / total, total


def analyze_results(job_result, pub_metadata):
    """
    Parse SamplerV2 result and return analysis dict.
    job_result: the result object from job.result()
    pub_metadata: list of metadata dicts matching pub order
    """
    import json

    results = []
    for idx, meta in enumerate(pub_metadata):
        pub_result = job_result[idx]
        # Try counts first, fall back to bit array
        try:
            counts = pub_result.data.c.get_counts()
            p_even, n_shots = p_even_from_counts(counts, None)
        except Exception:
            # Fallback: raw bitarray
            bitarray = pub_result.data.c
            samples = bitarray.get_int_counts()
            total = sum(samples.values())
            even = sum(v for k, v in samples.items() if bin(k).count('1') % 2 == 0)
            p_even = even / total
            n_shots = total

        row = dict(meta)
        row["p_even_measured"] = p_even
        row["n_shots"] = n_shots
        if "theory_p_even" in meta:
            row["residual"] = p_even - meta["theory_p_even"]
        results.append(row)

    return results


def print_summary(results):
    """Print a readable summary of results."""
    from scipy.stats import pearsonr

    print("\n" + "="*70)
    print("EXPERIMENT 5: PRIME ATOMIC BEATS — RESULTS SUMMARY")
    print("="*70)

    # Bell alpha sweep: fit cos((f1-f2)*alpha) per condition
    bell_alpha = [r for r in results if r["type"] == "bell_alpha"]
    conditions_done = set(r["condition"] for r in bell_alpha)

    print("\n[Bell Alpha Sweep] Pearson r vs cos((f1-f2)*alpha) theory:")
    print(f"{'Condition':<20} {'Prime?':<8} {'gap_mertens':<14} {'r':>8} {'mean_p_even':>12}")
    for cond in CONDITIONS:
        label = cond[0]
        is_prime = cond[3]
        rows = [r for r in bell_alpha if r["condition"] == label]
        if not rows:
            continue
        measured = np.array([r["p_even_measured"] for r in rows])
        theory   = np.array([r["theory_p_even"] for r in rows])
        r_val, _ = pearsonr(measured, theory)
        gap = rows[0]["gap"]
        print(f"  {label:<18} {'YES' if is_prime else 'NO':<8} {gap:<14.4f} {r_val:>8.4f} {measured.mean():>12.4f}")

    # Bell decoherence: P(even) vs delay for prime vs control
    bell_deco = [r for r in results if r["type"] == "bell_deco"]
    if bell_deco:
        print("\n[Decoherence Scan] P(even) at alpha=pi vs delay:")
        print(f"{'Condition':<20} {'Prime?':<8} ", end="")
        delays = sorted(set(r["delay_ns"] for r in bell_deco))
        for d in delays:
            print(f"  {int(d)}ns", end="")
        print()
        for cond in CONDITIONS:
            label = cond[0]
            is_prime = cond[3]
            rows = sorted([r for r in bell_deco if r["condition"] == label],
                          key=lambda x: x["delay_ns"])
            if not rows:
                continue
            print(f"  {label:<18} {'YES' if is_prime else 'NO':<8} ", end="")
            for r in rows:
                print(f"  {r['p_even_measured']:.3f}", end="")
            print()

    # GHZ reference
    ghz = [r for r in results if r["type"] == "ghz_reference"]
    if ghz:
        measured = np.array([r["p_even_measured"] for r in ghz])
        theory   = np.array([r["theory_p_even"] for r in ghz])
        r_val, _ = pearsonr(measured, theory)
        print(f"\n[GHZ Reference] r = {r_val:.4f} vs cos^2(Phi/2)  (Exp 4 sanity check)")

    print("\n" + "="*70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if DRY_RUN:
        # Build without backend_dt — uses unit='s' fallback
        pubs = build_all_pubs(backend_dt=None)
        circuits = [p[0] for p in pubs]
        metadata = [p[3] for p in pubs]
        print("\nDRY RUN — circuits built, not submitted.")
        print(f"Circuit depths: {[c.depth() for c in circuits[:6]]} ...")
        for qc in circuits[:2]:
            print(qc.draw(output='text', fold=80))
        return None, metadata

    # Connect to IBM first — need backend.dt before building delay circuits
    print("\nConnecting to IBM Quantum...")
    service = QiskitRuntimeService(
        channel=CHANNEL,
        instance=INSTANCE,
        token=TOKEN,
    )

    backend = service.least_busy(min_num_qubits=3, simulator=False)
    print(f"Selected backend: {backend.name}")

    # Fetch dt (hardware clock period, typically ~0.222 ns for IBM backends)
    backend_dt = backend.dt
    print(f"Backend dt = {backend_dt:.4e} s  ({backend_dt * 1e9:.4f} ns)")
    for delay_s in DELAYS_NS:
        if delay_s > 0:
            pri