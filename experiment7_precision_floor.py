"""
Experiment 7: Mertens Spectrometer — Precision Floor
=====================================================
Branch A: Quantum Mertens Spectrometer

Determines the minimum Mertens frequency gap the Bell-pair XX correlator
can resolve on real IBM hardware. Each condition is tuned so alpha_max
covers exactly two full oscillation periods — equal statistical power
across all gaps regardless of their magnitude.

DESIGN PRINCIPLE
----------------
For a gap Δf, oscillation period T = 1/Δf in alpha-space.
Set alpha_max = 2T, sample 20 points (10 per period).
Every condition then accumulates ~12.5 radians of total phase,
giving equivalent fit sensitivity regardless of the gap size.

The precision floor is set by:
  σ(Δf) ~ 1 / (alpha_max * sqrt(N_shots))
         = Δf / (2 * sqrt(N_shots))     [since alpha_max = 2/Δf]

For N_shots = 4096:  σ(Δf) ~ Δf / 128
This predicts ~0.8% relative error at any gap — consistent with Exp 6.
The experiment tests whether hardware noise breaks this scaling for small gaps.

CONDITIONS
----------
  cal_7_v_8   : (p=7, n=8=2^3)  Δf=0.01806  alpha_max=100   [calibration, Exp 6 known]
  prec_23_v_24: (p=23, n=24)    Δf=0.00391  alpha_max=500
  prec_29_v_30: (p=29, n=30)    Δf=0.00274  alpha_max=750
  prec_31_v_32: (p=31, n=32)    Δf=0.00247  alpha_max=800
  prec_37_v_38: (p=37, n=38)    Δf=0.00187  alpha_max=1050
  prec_41_v_42: (p=41, n=42)    Δf=0.00158  alpha_max=1250

GHZ reference: primes [2,3,5], alpha in [0, 2*pi], 8 points.

TOTAL PUBs: (8 + 5*20) + 8 = 116
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

def f_mertens(n):
    return math.log(n) / n

# ── Conditions — each tuned to cover 2 oscillation periods ───────────────────
# (label, n1, n2, alpha_max, n_alpha_points)
CONDITIONS = [
    ("cal_7_v_8",    7,  8,  100,   8),   # Exp 6 calibration reference
    ("prec_23_v_24", 23, 24, 500,  20),
    ("prec_29_v_30", 29, 30, 750,  20),
    ("prec_31_v_32", 31, 32, 800,  20),
    ("prec_37_v_38", 37, 38, 1050, 20),
    ("prec_41_v_42", 41, 42, 1250, 20),
]

GHZ_PRIMES = [2, 3, 5]
GHZ_ALPHAS = np.linspace(0, 2 * math.pi, 8, endpoint=False)


def print_condition_summary():
    print("\nCondition summary (alpha_max = 2 * period for each):")
    print(f"  {'Label':<16} {'n1':>3} {'n2':>3} {'Delta_f':>10} {'period':>9} "
          f"{'alpha_max':>10} {'n_pts':>6} {'phase@max':>10}")
    for label, n1, n2, amax, npts in CONDITIONS:
        df  = abs(f_mertens(n1) - f_mertens(n2))
        per = 1.0 / df if df > 0 else float('inf')
        ph  = 2 * math.pi * df * amax
        print(f"  {label:<16} {n1:>3} {n2:>3} {df:>10.6f} {per:>9.1f} "
              f"{amax:>10} {npts:>6} {ph:>10.2f} rad")
    print(f"\n  Theoretical precision: sigma(Delta_f)/Delta_f ~ 1/(2*sqrt(N)) "
          f"= {1/(2*math.sqrt(SHOTS))*100:.2f}% at {SHOTS} shots")


# ── Circuit builder ───────────────────────────────────────────────────────────

def build_bell_circuit(f1, f2, alpha, label=""):
    """Bell pair |Psi+> XX parity. P(even) = 1/2*(1+cos(2*pi*(f1-f2)*alpha))"""
    qr = QuantumRegister(2, 'q')
    cr = ClassicalRegister(2, 'c')
    qc = QuantumCircuit(qr, cr, name=label or f"bell_a{alpha:.1f}")
    qc.h(qr[0])
    qc.cx(qr[0], qr[1])
    qc.x(qr[1])
    qc.rz(2 * math.pi * f1 * alpha, qr[0])
    qc.rz(2 * math.pi * f2 * alpha, qr[1])
    qc.h(qr[0])
    qc.h(qr[1])
    qc.measure(qr, cr)
    return qc


def build_ghz_circuit(primes, alpha, label=""):
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
    pubs = []

    print("Building Bell precision circuits...")
    for label, n1, n2, amax, npts in CONDITIONS:
        f1, f2 = f_mertens(n1), f_mertens(n2)
        df = f1 - f2
        alphas = np.linspace(0, amax, npts)
        for alpha in alphas:
            qc = build_bell_circuit(f1, f2, alpha,
                                    label=f"{label}_a{alpha:.1f}")
            meta = {
                "block": "bell_precision",
                "label": label,
                "n1": n1, "n2": n2,
                "f1": f1, "f2": f2,
                "delta_f": df,
                "alpha": float(alpha),
                "alpha_max": amax,
                "theory_p_even": 0.5 * (1 + math.cos(2 * math.pi * df * alpha)),
            }
            pubs.append((qc, meta))
        print(f"  {label}: {npts} PUBs, alpha_max={amax}")

    print("Building GHZ reference circuits...")
    phi_sum = sum(f_mertens(p) for p in GHZ_PRIMES)
    for alpha in GHZ_ALPHAS:
        phi = 2 * math.pi * phi_sum * alpha
        qc = build_ghz_circuit(GHZ_PRIMES, alpha)
        meta = {
            "block": "ghz_ref",
            "primes": GHZ_PRIMES,
            "alpha": float(alpha),
            "phi": phi,
            "theory_p_even": math.cos(phi / 2) ** 2,
        }
        pubs.append((qc, meta))

    print(f"Total: {len(pubs)} PUBs")
    return pubs


# ── Analysis ──────────────────────────────────────────────────────────────────

def p_even_from_counts(counts):
    total = sum(counts.values())
    even  = sum(v for k, v in counts.items() if k.count('1') % 2 == 0)
    return even / total, total


def analyze_results(job_result, pub_metas):
    results = []
    for idx, meta in enumerate(pub_metas):
        pub = job_result[idx]
        try:
            counts = pub.data.c.get_counts()
            p_even, n_shots = p_even_from_counts(counts)
        except Exception:
            samples = pub.data.c.get_int_counts()
            total = sum(samples.values())
            even  = sum(v for k, v in samples.items()
                       if bin(k).count('1') % 2 == 0)
            p_even = even / total
            n_shots = total
        row = dict(meta)
        row["p_even"]  = p_even
        row["n_shots"] = n_shots
        if "theory_p_even" in meta:
            row["residual"] = p_even - meta["theory_p_even"]
        results.append(row)
    return results


def print_summary(results):
    from scipy.stats import pearsonr
    from scipy.optimize import curve_fit

    def model(alpha, df, amp, baseline):
        return baseline + amp * 0.5 * (1 + np.cos(2 * math.pi * df * alpha))

    print("\n" + "=" * 72)
    print("EXPERIMENT 7: PRECISION FLOOR - RESULTS")
    print("=" * 72)
    print(f"\n{'Label':<16} {'Delta_f_theory':>14} {'Delta_f_fit':>12} {'err%':>7} "
          f"{'amp':>6} {'r':>7} {'RESOLVED?':>10}")

    bell = [r for r in results if r["block"] == "bell_precision"]
    for label, n1, n2, amax, npts in CONDITIONS:
        rows = sorted([r for r in bell if r["label"] == label],
                      key=lambda x: x["alpha"])
        if not rows:
            continue
        alphas  = np.array([r["alpha"]  for r in rows])
        measured= np.array([r["p_even"] for r in rows])
        theory  = np.array([r["theory_p_even"] for r in rows])
        df_th   = abs(rows[0]["delta_f"])
        r_th, _ = pearsonr(measured, theory)

        try:
            popt, _ = curve_fit(model, alphas, measured,
                                p0=[df_th, 0.9, 0.05],
                                bounds=([0, 0, 0], [1, 1, 1]),
                                maxfev=20000)
            df_fit, amp, baseline = popt
            err_pct = (df_fit - df_th) / df_th * 100
            r_fit, _ = pearsonr(measured, model(alphas, *popt))
            resolved = "YES" if r_fit > 0.95 else ("MARGINAL" if r_fit > 0.80 else "NO")
            print(f"  {label:<16} {df_th:>11.6f} {df_fit:>10.6f} "
                  f"{err_pct:>+7.2f}% {amp:>6.3f} {r_fit:>7.4f} {resolved:>10}")
        except Exception as e:
            print(f"  {label:<16} {df_th:>11.6f} {'FIT FAILED':>10}  [{e}]")

    # GHZ reference
    ghz = [r for r in results if r["block"] == "ghz_ref"]
    if ghz:
        m = np.array([r["p_even"]       for r in ghz])
        t = np.array([r["theory_p_even"] for r in ghz])
        r_ghz, _ = pearsonr(m, t)
        print(f"\nGHZ sanity check: r = {r_ghz:.4f}")

    print("\n" + "=" * 72)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print_condition_summary()

    if DRY_RUN:
        pubs = build_all_pubs()
        circuits = [p[0] for p in pubs]
        metas    = [p[1] for p in pubs]
        print(f"\nDRY RUN — {len(circuits)} circuits, not submitted.")
        print(f"Circuit depths: {sorted(set(c.depth() for c in circuits))}")
        max_angle = max(
            2 * math.pi * max(f_mertens(n1), f_mertens(n2)) * amax
            for _, n1, n2, amax, _ in CONDITIONS
        )
        print(f"Max RZ angle: {max_angle:.1f} rad")
        return None, metas

    print("\nConnecting to IBM Quantum...")
    service = QiskitRuntimeService(channel=CHANNEL, instance=INSTANCE, token=TOKEN)
    backend = service.least_busy(min_num_qubits=3, simulator=False)
    print(f"Backend: {backend.name},  dt = {backend.dt:.4e} s")

    pubs   = build_all_pubs()
    circuits = [p[0] for p in pubs]
    metas    = [p[1] for p in pubs]

    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    transpiled = pm.run(circuits)
    depths = [c.depth() for c in transpiled]
    print(f"Transpiled. Depths: min={min(depths)}, max={max(depths)}")

    sampler  = Sampler(mode=backend)
    pub_list = [(tc,) for tc in transpiled]
    print(f"Submitting {len(pub_list)} PUBs @ {SHOTS} shots...")
    job = sampler.run(pub_list, shots=SHOTS)
    print(f"Job ID: {job.job_id()}")
    print(f"Retrieve with:  python retrieve_experiment7.py {job.job_id()}")
    return job, metas


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    if ap.parse_args().dry_run:
        DRY_RUN = True
   