"""
Experiment 8: Mertens Full Spectrum
=====================================
Branch A: Quantum Mertens Spectrometer

Measures f(n) = ln(n)/n for every integer n = 2..20 in a single IBM job.
Each n is paired against n=1 (f(1)=0), so the Bell-pair XX correlator
directly oscillates at the Mertens frequency:

    P(even) = 1/2 * (1 + cos(2*pi * f(n) * alpha))

This produces the complete Mertens spectrum on real quantum hardware,
confirming spectral isolation of primes from composites.

DESIGN PRINCIPLE
----------------
Same equal-power design as Experiment 7:
    alpha_max(n) = 2 / f(n)   =>  exactly 2 oscillation periods per condition
    Total phase = 4*pi ~ 12.57 rad  (constant across all n)

Max RZ angle = 4*pi ~ 12.57 rad for every condition (hardware-safe).

CONDITIONS (n=2..20, each vs n=1 reference)
-------------------------------------------
  n    f(n)       alpha_max
  2    0.346574    5.77   PRIME
  3    0.366204    5.46   PRIME
  4    0.346574    5.77   [degenerate with n=2: f(4)=f(2)]
  5    0.321888    6.21   PRIME
  6    0.298627    6.70
  7    0.277987    7.19   PRIME
  8    0.259930    7.69
  9    0.244136    8.19
  10   0.230259    8.69
  11   0.217990    9.17   PRIME
  12   0.207076    9.66
  13   0.197304   10.14   PRIME
  14   0.188504   10.61
  15   0.180537   11.08
  16   0.173287   11.54
  17   0.166660   12.00   PRIME
  18   0.160576   12.46
  19   0.154970   12.91   PRIME
  20   0.149787   13.35

GHZ reference: primes [2,3,5], alpha in [0, 2*pi], 8 points.

TOTAL PUBs: 19*20 + 8 = 388
"""

import numpy as np
import math
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# -- Credentials ----------------------------------------------------------
TOKEN    = "96axnVJAp_PkXhi7mpX8t_CVj1NtzqmHjaApLQ5Pn96Q"
INSTANCE = "crn:v1:bluemix:public:quantum-computing:us-east:a/8420df4c778d45e59489b345c26d2c81:73caf2a1-d677-4d15-b711-f8f3fc73732a::"
CHANNEL  = "ibm_quantum_platform"

SHOTS       = 4096
DRY_RUN     = False
N_ALPHA_PTS = 20

def f_mertens(n):
    if n == 1:
        return 0.0
    return math.log(n) / n

# Build conditions: every n from 2 to 20 paired against n=1 (f=0)
def build_conditions():
    conds = []
    for n in range(2, 21):
        fn = f_mertens(n)
        # alpha_max covers exactly 2 full oscillation periods
        amax = round(2.0 / fn, 4)
        label = f"n{n:02d}"
        conds.append((label, n, amax))
    return conds

CONDITIONS = build_conditions()

GHZ_PRIMES = [2, 3, 5]
GHZ_ALPHAS = np.linspace(0, 2 * math.pi, 8, endpoint=False)

PRIMES_2_20 = {2, 3, 5, 7, 11, 13, 17, 19}


def print_condition_summary():
    print("\nFull Mertens Spectrum — condition summary:")
    print(f"  {'Label':<6} {'n':>3} {'f(n)':>10} {'period':>8} "
          f"{'alpha_max':>10} {'phase@max':>10}  type")
    for label, n, amax in CONDITIONS:
        fn   = f_mertens(n)
        T    = 1.0 / fn
        ph   = 2 * math.pi * fn * amax
        tag  = "PRIME" if n in PRIMES_2_20 else ("=f(2)" if abs(fn - f_mertens(2)) < 1e-9 and n != 2 else "")
        print(f"  {label:<6} {n:>3} {fn:>10.6f} {T:>8.3f} "
              f"{amax:>10.4f} {ph:>10.3f} rad  {tag}")
    total_pubs = len(CONDITIONS) * N_ALPHA_PTS + len(GHZ_ALPHAS)
    print(f"\n  {len(CONDITIONS)} conditions x {N_ALPHA_PTS} points + {len(GHZ_ALPHAS)} GHZ = {total_pubs} total PUBs")
    print(f"  Theoretical precision: sigma(f)/f ~ 1/(2*sqrt(N)) = "
          f"{1/(2*math.sqrt(SHOTS))*100:.2f}% at {SHOTS} shots")


# -- Circuit builder ------------------------------------------------------

def build_bell_circuit(f_ref, f_n, alpha, label=""):
    """
    Bell pair |Psi+> XX parity.
    P(even) = 1/2 * (1 + cos(2*pi*(f_ref - f_n)*alpha))
    With f_ref=0: P(even) = 1/2 * (1 + cos(2*pi*f_n*alpha))
    """
    qr = QuantumRegister(2, 'q')
    cr = ClassicalRegister(2, 'c')
    qc = QuantumCircuit(qr, cr, name=label or f"bell_{alpha:.2f}")
    qc.h(qr[0])
    qc.cx(qr[0], qr[1])
    qc.x(qr[1])
    if f_ref != 0.0:
        qc.rz(2 * math.pi * f_ref * alpha, qr[0])
    qc.rz(2 * math.pi * f_n * alpha, qr[1])
    qc.h(qr[0])
    qc.h(qr[1])
    qc.measure(qr, cr)
    return qc


def build_ghz_circuit(primes, alpha, label=""):
    n = len(primes)
    qr = QuantumRegister(n, 'q')
    cr = ClassicalRegister(n, 'c')
    qc = QuantumCircuit(qr, cr, name=label or f"ghz_{alpha:.3f}")
    qc.h(qr[0])
    for i in range(n - 1):
        qc.cx(qr[i], qr[i + 1])
    for i, p in enumerate(primes):
        qc.rz(2 * math.pi * f_mertens(p) * alpha, qr[i])
    for i in range(n):
        qc.h(qr[i])
    qc.measure(qr, cr)
    return qc


# -- Build all PUBs -------------------------------------------------------

def build_all_pubs():
    pubs = []

    print("Building Bell spectrum circuits...")
    for label, n, amax in CONDITIONS:
        fn     = f_mertens(n)
        alphas = np.linspace(0, amax, N_ALPHA_PTS)
        for alpha in alphas:
            qc = build_bell_circuit(0.0, fn, alpha, label=f"{label}_a{alpha:.3f}")
            meta = {
                "block":        "bell_spectrum",
                "label":        label,
                "n":            n,
                "f_n":          fn,
                "alpha":        float(alpha),
                "alpha_max":    amax,
                "is_prime":     n in PRIMES_2_20,
                "theory_p_even": 0.5 * (1 + math.cos(2 * math.pi * fn * alpha)),
            }
            pubs.append((qc, meta))
        print(f"  {label} (n={n}, f={fn:.6f}): {N_ALPHA_PTS} PUBs, alpha_max={amax}")

    print("Building GHZ reference circuits...")
    phi_sum = sum(f_mertens(p) for p in GHZ_PRIMES)
    for alpha in GHZ_ALPHAS:
        phi = 2 * math.pi * phi_sum * alpha
        qc  = build_ghz_circuit(GHZ_PRIMES, alpha)
        meta = {
            "block":         "ghz_ref",
            "primes":        GHZ_PRIMES,
            "alpha":         float(alpha),
            "phi":           phi,
            "theory_p_even": math.cos(phi / 2) ** 2,
        }
        pubs.append((qc, meta))

    print(f"Total: {len(pubs)} PUBs")
    return pubs


# -- Analysis -------------------------------------------------------------

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
            total   = sum(samples.values())
            even    = sum(v for k, v in samples.items()
                         if bin(k).count('1') % 2 == 0)
            p_even  = even / total
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

    def model(alpha, fn, amp, baseline):
        return baseline + amp * 0.5 * (1 + np.cos(2 * math.pi * fn * alpha))

    print("\n" + "=" * 76)
    print("EXPERIMENT 8: FULL MERTENS SPECTRUM - RESULTS")
    print("=" * 76)
    print(f"\n{'Label':<7} {'n':>3} {'f_theory':>10} {'f_fit':>10} "
          f"{'err%':>7} {'amp':>6} {'r':>7} {'type':<8} {'RESOLVED?':>10}")

    bell = [r for r in results if r["block"] == "bell_spectrum"]
    spectrum_results = []

    for label, n, amax in CONDITIONS:
        rows = sorted([r for r in bell if r["label"] == label],
                      key=lambda x: x["alpha"])
        if not rows:
            continue
        alphas   = np.array([r["alpha"]          for r in rows])
        measured = np.array([r["p_even"]          for r in rows])
        fn_th    = rows[0]["f_n"]
        tag      = "PRIME" if n in PRIMES_2_20 else ("=f(2)" if abs(fn_th - f_mertens(2)) < 1e-9 and n != 2 else "comp")

        try:
            popt, _ = curve_fit(model, alphas, measured,
                                p0=[fn_th, 0.9, 0.05],
                                bounds=([0, 0, 0], [1, 1, 1]),
                                maxfev=20000)
            fn_fit, amp, baseline = popt
            err_pct  = (fn_fit - fn_th) / fn_th * 100
            r_fit, _ = pearsonr(measured, model(alphas, *popt))
            resolved = "YES" if r_fit > 0.95 else ("MARGINAL" if r_fit > 0.80 else "NO")
            print(f"  {label:<7} {n:>3} {fn_th:>10.6f} {fn_fit:>10.6f} "
                  f"{err_pct:>+7.2f}% {amp:>6.3f} {r_fit:>7.4f} {tag:<8} {resolved:>10}")
            spectrum_results.append({
                "n": n, "label": label, "f_theory": fn_th,
                "f_fit": fn_fit, "err_pct": err_pct,
                "amp": amp, "r_fit": r_fit, "resolved": resolved,
                "is_prime": n in PRIMES_2_20,
            })
        except Exception as e:
            print(f"  {label:<7} {n:>3} {fn_th:>10.6f} {'FIT FAILED':>10}  [{e}]")

    # GHZ reference
    ghz = [r for r in results if r["block"] == "ghz_ref"]
    if ghz:
        m = np.array([r["p_even"]        for r in ghz])
        t = np.array([r["theory_p_even"] for r in ghz])
        r_ghz, _ = pearsonr(m, t)
        print(f"\nGHZ sanity check: r = {r_ghz:.4f}")

    # Summary statistics
    yes_all    = [s for s in spectrum_results if s["resolved"] == "YES"]
    yes_prime  = [s for s in yes_all  if s["is_prime"]]
    yes_comp   = [s for s in yes_all  if not s["is_prime"]]
    if spectrum_results:
        mean_err = np.mean([abs(s["err_pct"]) for s in spectrum_results])
        mean_amp = np.mean([s["amp"] for s in spectrum_results])
        print(f"\nResolved: {len(yes_all)}/{len(spectrum_results)} total "
              f"({len(yes_prime)}/8 primes, {len(yes_comp)}/11 composites)")
        print(f"Mean |err%|: {mean_err:.2f}%,  mean amplitude: {mean_amp:.3f}")

        # Check f(4) = f(2) degeneracy
        r2 = next((s for s in spectrum_results if s["n"] == 2), None)
        r4 = next((s for s in spectrum_results if s["n"] == 4), None)
        if r2 and r4:
            diff = abs(r2["f_fit"] - r4["f_fit"])
            print(f"f(2) vs f(4): fitted {r2['f_fit']:.6f} vs {r4['f_fit']:.6f}, "
                  f"diff={diff:.2e}  (theory: 0.000000)")

    print("\n" + "=" * 76)
    return spectrum_results


# -- Main -----------------------------------------------------------------

def main():
    print_condition_summary()

    if DRY_RUN:
        pubs     = build_all_pubs()
        circuits = [p[0] for p in pubs]
        metas    = [p[1] for p in pubs]
        print(f"\nDRY RUN -- {len(circuits)} circuits, not submitted.")
        print(f"Circuit depths: {sorted(set(c.depth() for c in circuits))}")
        max_angle = max(
            2 * math.pi * f_mertens(n) * amax for _, n, amax in CONDITIONS
        )
        print(f"Max RZ angle: {max_angle:.3f} rad")
        return None, metas

    print("\nConnecting to IBM Quantum...")
    service = QiskitRuntimeService(channel=CHANNEL, instance=INSTANCE, token=TOKEN)
    backend = service.least_busy(min_num_qubits=3, simulator=False)
    print(f"Backend: {backend.name},  dt = {backend.dt:.4e} s")

    pubs     = build_all_pubs()
    circuits = [p[0] for p in pubs]
    metas    = [p[1] for p in pubs]

    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pm         = generate_preset_pass_manager(optimization_level=1, backend=backend)
    transpiled = pm.run(circuits)
    depths     = [c.depth() for c in transpiled]
    print(f"Transpiled. Depths: min={min(depths)}, max={max(depths)}")

    sampler  = Sampler(mode=backend)
    pub_list = [(tc,) for tc in transpiled]
    print(f"Submitting {len(pub_list)} PUBs @ {SHOTS} shots...")
    job = sampler.run(pub_list, shots=SHOTS)
    print(f"Job ID: {job.job_id()}")
    print(f"Retrieve with:  python retrieve_experiment8.py {job.job_id()}")
    return job, metas


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    if ap.parse_args().dry_run:
        DRY_RUN = True
    job, metas = main()
