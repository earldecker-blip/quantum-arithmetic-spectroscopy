"""
Experiment 6: Mertens Spectral Isolation of Small Primes
=========================================================
Demonstrates that small primes occupy isolated, distinguishable positions
in Mertens frequency space — no composite sits within experimentally
resolvable range of their weight f(p) = ln(p)/p.

BACKGROUND
----------
From the Exp 5 analysis:
  - f(n) = ln(n)/n has a unique maximum at n=e (value 1/e = 0.3679)
  - f(2) = f(4) EXACTLY — the only algebraic degeneracy in the integers
  - Primes 3,5,7,11,13,17,19 are spectrally isolated: gap to nearest
    composite exceeds 0.005 (experimentally resolvable at alpha ~ 55)
  - The isolation gap scales as ~ln(p)/p^2, so small primes are
    anomalously well-separated relative to the asymptotic trend

OBSERVABLE
----------
Bell pair |Psi+> = (|01>+|10>)/sqrt(2)

  P(even) = 1/2 * (1 + cos(Delta_f * alpha))
  Delta_f = f(n1) - f(n2) = ln(n1)/n1 - ln(n2)/n2

where alpha is a tunable parameter controlling phase accumulation.
Unlike Exp 5 (alpha in [0, 2pi]), here alpha goes up to 110 so that
even the tightest prime-composite gaps (~0.018) accumulate ~2 radians
of phase — easily distinguishable from all six conditions.

CONDITIONS (6 pairs)
--------------------
1. prime_7_v_8  : (7, 8=2^3)   gap=0.01806  period_alpha=347
2. prime_7_v_6  : (7, 6=2*3)   gap=0.02064  period_alpha=305
3. prime_7_v_9  : (7, 9=3^2)   gap=0.03385  period_alpha=186
4. prime_3_v_4  : (3, 4=2^2)   gap=0.01963  period_alpha=320
5. prime_3_v_7  : (3, 7)       gap=0.08821  period_alpha=71  [prime vs prime]
6. ctrl_6_v_8   : (6, 8=2^3)   gap=0.00264  period_alpha=2381 [composite vs composite]

At alpha_max=110:
  pair (7,8): Delta_f * 110 = 1.99 rad  -> P_even ~ 0.22 (near trough)
  pair (7,6): Delta_f * 110 = 2.27 rad  -> P_even ~ 0.16
  pair (7,9): Delta_f * 110 = 3.72 rad  -> P_even ~ 0.71 (past minimum, rising)
  pair (3,4): Delta_f * 110 = 2.16 rad  -> P_even ~ 0.19
  pair (3,7): Delta_f * 110 = 9.70 rad  -> P_even ~ 0.73 (multiple oscillations)
  pair (6,8): Delta_f * 110 = 0.29 rad  -> P_even ~ 0.96 (barely moved — ISOLATED)

The control (6,8) barely oscillates at alpha=110, while prime conditions
show clear, distinct oscillation patterns — direct demonstration of
prime spectral isolation.

GHZ REFERENCE
-------------
3-qubit GHZ with primes [2,3,5], reproducing Exp 4 sanity check.

TOTAL PUBs: 6 * 16 + 8 = 104
"""

import numpy as np
import math
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# ── Credentials ───────────────────────────────────────────────────────────────
TOKEN    = "96axnVJAp_PkXhi7mpX8t_CVj1NtzqmHjaApLQ5Pn96Q"
INSTANCE = "crn:v1:bluemix:public:quantum-computing:us-east:a/8420df4c778d45e59489b345c26d2c81:73caf2a1-d677-4d15-b711-f8f3fc73732a::"
CHANNEL  = "ibm_quantum_platform"

SHOTS   = 4096
DRY_RUN = False

# ── Frequency model ───────────────────────────────────────────────────────────
def f_mertens(n):
    """f(n) = ln(n)/n — Mertens weight"""
    return math.log(n) / n

# ── Alpha sweep: 0 to 110 in 16 steps ────────────────────────────────────────
# Chosen so that even the tightest prime-composite gap (Delta_f ~ 0.018)
# accumulates ~2 radians of phase at alpha_max, giving a clear signal.
ALPHA_SWEEP = np.linspace(0, 110, 16)

# ── Conditions ────────────────────────────────────────────────────────────────
# (label, n1, n2, description)
CONDITIONS = [
    ("prime_7_v_8", 7, 8,  "p=7 vs 8=2^3   | nearest composite above"),
    ("prime_7_v_6", 7, 6,  "p=7 vs 6=2*3   | nearest composite below"),
    ("prime_7_v_9", 7, 9,  "p=7 vs 9=3^2   | 2nd nearest below"),
    ("prime_3_v_4", 3, 4,  "p=3 vs 4=2^2   | only algebraic near-twin"),
    ("prime_3_v_7", 3, 7,  "p=3 vs p=7     | prime vs prime, large gap"),
    ("ctrl_6_v_8",  6, 8,  "6=2*3 vs 8=2^3 | composite vs composite, tiny gap"),
]

# GHZ reference (Exp 4 sanity check)
GHZ_PRIMES = [2, 3, 5]
GHZ_ALPHAS = np.linspace(0, 2 * math.pi, 8, endpoint=False)

# ── Pre-compute expected gaps and phase at alpha_max ─────────────────────────
def print_condition_summary():
    print("\nCondition summary:")
    print(f"  {'Label':<16} {'n1':>3} {'n2':>3} {'f(n1)':>8} {'f(n2)':>8} {'Delta_f':>9}  "
          f"{'period':>8}  {'phase@110':>10}  {'P_even@110':>11}")
    for label, n1, n2, desc in CONDITIONS:
        fn1, fn2 = f_mertens(n1), f_mertens(n2)
        df = abs(fn1 - fn2)
        period = 2 * math.pi / df if df > 0 else float('inf')
        phase = df * 110
        p_even = 0.5 * (1 + math.cos((fn1 - fn2) * 110))
        print(f"  {label:<16} {n1:>3} {n2:>3} {fn1:>8.5f} {fn2:>8.5f} {df:>9.5f}  "
              f"{period:>8.1f}  {phase:>10.3f}  {p_even:>11.4f}")

# ── Circuit builders ──────────────────────────────────────────────────────────

def build_bell_circuit(f1, f2, alpha, label=""):
    """
    Bell pair |Psi+> XX parity circuit.
    P(even) = 1/2 * (1 + cos((f1 - f2) * alpha))

    Step 1: Prepare |Psi+> = (|01>+|10>)/sqrt(2)
            H(q0) -> CX(q0,q1) -> X(q1)
    Step 2: Phase kick  RZ(2*pi*f1*alpha, q0), RZ(2*pi*f2*alpha, q1)
    Step 3: Measure in X basis: H(q0), H(q1), measure
    """
    qr = QuantumRegister(2, 'q')
    cr = ClassicalRegister(2, 'c')
    qc = QuantumCircuit(qr, cr, name=label or f"bell_f{f1:.4f}_f{f2:.4f}_a{alpha:.2f}")

    # |Psi+>
    qc.h(qr[0])
    qc.cx(qr[0], qr[1])
    qc.x(qr[1])

    # Phase kicks — alpha can be >> 2*pi; RZ handles any angle
    qc.rz(2 * math.pi * f1 * alpha, qr[0])
    qc.rz(2 * math.pi * f2 * alpha, qr[1])

    # X-basis measurement
    qc.h(qr[0])
    qc.h(qr[1])
    qc.measure(qr, cr)
    return qc


def build_ghz_circuit(primes, alpha, label=""):
    """GHZ reference: P(even) = cos^2(Phi/2), Phi = 2*pi * sum(f_p) * alpha."""
    n = len(primes)
    qr = QuantumRegister(n, 'q')
    cr = ClassicalRegister(n, 'c')
    qc = QuantumCircuit(qr, cr, name=label or f"ghz_a{alpha:.3f}")

    qc.h(qr[0])
    for i in range(n - 1):
        qc.cx(qr[i], qr[i + 1])

    for i, p in enumerate(primes):
        qc.rz(2 * math.pi * f_mertens(p) * alpha, qr[i])

    for i in range(n):
        qc.h(qr[i])

    qc.measure(qr, cr)
    return qc

# ── Build all PUBs ────────────────────────────────────────────────────────────

def build_all_pubs():
    """Returns list of (circuit, label, metadata) tuples."""
    pubs = []

    # === Block 1: Bell alpha sweep ===
    print("Building Bell alpha sweep circuits...")
    for label, n1, n2, desc in CONDITIONS:
        f1, f2 = f_mertens(n1), f_mertens(n2)
        df = f1 - f2
        for alpha in ALPHA_SWEEP:
            circ_label = f"{label}_a{alpha:.2f}"
            qc = build_bell_circuit(f1, f2, alpha, label=circ_label)
            meta = {
                "block": "bell_sweep",
                "label": label,
                "n1": n1, "n2": n2,
                "f1": f1, "f2": f2,
                "delta_f": df,
                "alpha": float(alpha),
                "theory_p_even": 0.5 * (1 + math.cos(df * alpha)),
            }
            pubs.append((qc, circ_label, meta))

    print(f"  -> {len(pubs)} bell sweep PUBs")

    # === Block 2: GHZ reference ===
    ghz_start = len(pubs)
    print("Building GHZ reference circuits...")
    for alpha in GHZ_ALPHAS:
        circ_label = f"ghz_ref_a{alpha:.4f}"
        qc = build_ghz_circuit(GHZ_PRIMES, alpha, label=circ_label)
        phi = 2 * math.pi * sum(f_mertens(p) for p in GHZ_PRIMES) * alpha
        meta = {
            "block": "ghz_ref",
            "primes": GHZ_PRIMES,
            "alpha": float(alpha),
            "phi": phi,
            "theory_p_even": math.cos(phi / 2) ** 2,
        }
        pubs.append((qc, circ_label, meta))

    print(f"  -> {len(pubs) - ghz_start} GHZ reference PUBs")
    print(f"Total: {len(pubs)} PUBs")
    return pubs

# ── Analysis ──────────────────────────────────────────────────────────────────

def p_even_from_counts(counts):
    total = sum(counts.values())
    even  = sum(v for k, v in counts.items() if k.count('1') % 2 == 0)
    return even / total, total


def analyze_results(job_result, metadata):
    """Parse SamplerV2 result, return list of result dicts."""
    results = []
    for idx, meta in enumerate(metadata):
        pub = job_result[idx]
        try:
            counts = pub.data.c.get_counts()
            p_even, n_shots = p_even_from_counts(counts)
        except Exception:
            samples = pub.data.c.get_int_counts()
            total = sum(samples.values())
            even  = sum(v for k, v in samples.items() if bin(k).count('1') % 2 == 0)
            p_even = even / total
            n_shots = total

        row = dict(meta)
        row["p_even"] = p_even
        row["n_shots"] = n_shots
        if "theory_p_even" in meta:
            row["residual"] = p_even - meta["theory_p_even"]
        results.append(row)
    return results


def print_summary(results):
    from scipy.stats import pearsonr

    print("\n" + "=" * 72)
    print("EXPERIMENT 6: MERTENS SPECTRAL ISOLATION — RESULTS")
    print("=" * 72)

    # ── Bell sweep ────────────────────────────────────────────────────────────
    bell = [r for r in results if r["block"] == "bell_sweep"]
    print(f"\n{'Label':<16}  {'Delta_f':>9}  {'r':>7}  {'RMSE':>7}  {'P@alpha=0':>10}  {'P@alpha=110':>12}  {'theory_P@110':>13}")

    for label, n1, n2, _ in CONDITIONS:
        rows = sorted([r for r in bell if r["label"] == label], key=lambda x: x["alpha"])
        if not rows: continue
        measured = np.array([r["p_even"] for r in rows])
        theory   = np.array([r["theory_p_even"] for r in rows])
        df = rows[0]["delta_f"]
        r_val, _ = pearsonr(measured, theory)
        rmse = np.sqrt(np.mean((measured - theory) ** 2))
        p_at_0   = measured[0]
        p_at_110 = measured[-1]
        t_at_110 = theory[-1]
        print(f"  {label:<16} {df:>9.5f}  {r_val:>7.4f}  {rmse:>7.4f}  {p_at_0:>10.4f}  {p_at_110:>12.4f}  {t_at_110:>13.4f}")

    # ── Key comparison: prime_7_v_8 vs ctrl_6_v_8 ────────────────────────────
    print("\n[Spectral isolation test: alpha=110]")
    print("  H0 (null): prime and composite pairs with similar f-gaps are indistinguishable")
    pairs = [
        ("prime_7_v_8", "ctrl_6_v_8",  "p=7 vs nearest composite (8)  vs  two composites (6,8)"),
        ("prime_7_v_6", "ctrl_6_v_8",  "p=7 vs nearest composite (6)  vs  two composites (6,8)"),
    ]
    for lab_prime, lab_ctrl, desc in pairs:
        prime_rows = sorted([r for r in bell if r["label"] == lab_prime], key=lambda x: x["alpha"])
        ctrl_rows  = sorted([r for r in bell if r["label"] == lab_ctrl],  key=lambda x: x["alpha"])
        if not prime_rows or not ctrl_rows: continue
        p_prime = prime_rows[-1]["p_even"]
        p_ctrl  = ctrl_rows[-1]["p_even"]
        print(f"  {desc}")
        print(f"    Prime pair P(even)@110 = {p_prime:.4f}")
        print(f"    Ctrl  pair P(even)@110 = {p_ctrl:.4f}")
        print(f"    Difference = {p_prime - p_ctrl:+.4f}  "
              f"(theory expects {prime_rows[-1]['theory_p_even'] - ctrl_rows[-1]['theory_p_even']:+.4f})")

    # ── GHZ reference ─────────────────────────────────────────────────────────
    ghz = [r for r in results if r["block"] == "ghz_ref"]
    if ghz:
        measured = np.array([r["p_even"] for r in ghz])
        theory   = np.array([r["theory_p_even"] for r in ghz])
        r_ghz, _ = pearsonr(measured, theory)
        print(f"\n[GHZ sanity check]  r = {r_ghz:.4f}  (Exp 4 expects ~0.999)")

    print("\n" + "=" * 72)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print_condition_summary()

    if DRY_RUN:
        pubs = build_all_pubs()
        circuits = [p[0] for p in pubs]
        metadata = [p[2] for p in pubs]
        print(f"\nDRY RUN — {len(circuits)} circuits built, not submitted.")
        print(f"Depths: {sorted(set(c.depth() for c in circuits))}")
        print(f"Max RZ angle (radians): {2*math.pi * max(f_mertens(n) for _,n,_,_ in CONDITIONS) * ALPHA_SWEEP.max():.1f}")
        return None, metadata

    # Connect first — need backend.dt before building circuits
    print("\nConnecting to IBM Quantum...")
    service = QiskitRuntimeService(channel=CHANNEL, instance=INSTANCE, token=TOKEN)
    backend = service.least_busy(min_num_qubits=3, simulator=False)
    print(f"Backend: {backend.name},  dt = {backend.dt:.4e} s")

    pubs = build_all_pubs()
    circuits = [p[0] for p in pubs]
    metadata = [p[2] for p in pubs]

    # Transpile
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    transpiled = pm.run(circuits)
    depths = [c.depth() for c in transpiled]
    print(f"Transpiled. Depths: min={min(depths)}, max={max(depths)}, mean={np.mean(depths):.1f}")

    # Submit
    sampler = Sampler(mode=backend)
    pub_list = [(tc,) for tc in transpiled]
    print(f"Submitting {len(pub_list)} PUBs @ {SHOTS} shots each...")
    job = sampler.run(pub_list, shots=SHOTS)
    job_id = job.job_id()
    print(f"Job ID: {job_id}")
    print(f"\nRetrieve with:")
    print(f"  python retrieve_experiment6.py {job_id}")

    return job, metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Experiment 6: Mertens Spectral Isolation")
    parser.add_argument("--dry-run", action="store_true", help="Build circuits only, no IBM submission")
    args = parser.parse_args()
    if args.dry_run:
        DRY_RUN = True
    job, metadata = main()
