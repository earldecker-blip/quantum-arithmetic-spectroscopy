"""
Branch C: Riemann Branch
========================
C1 Mertens Accumulator  -- GHZ spectrometer, N=1..20 cumulative prime sums
C2 Riemann Zero Hunter  -- Ramsey single-qubit, reconstructs Z_M(t) and
                           detects sign changes near Riemann zeros gamma_1..5

C1 DESIGN (N-qubit GHZ accumulator)
-------------------------------------
  H(q0), CNOT chain, RZ(2*pi*f(p_k)*alpha, q_k) for k=0..N-1, H all, measure
  P(even parity) = cos^2(pi * F_N * alpha)
  F_N = sum_{k=1}^{N} f(p_k),   f(p) = ln(p)/p
  alpha_max(N) = 2/F_N  =>  4*pi total phase accumulation (equal-power design)
  C1_N_ALPHA points per condition

  Physical meaning: each N-qubit GHZ run is a quantum balance-scale comparing
  the entangled prime-weighted phase F_N against a reference of zero.  The
  cos^2 oscillation encodes the exact cumulative Mertens weight to within the
  photon-noise limit 1/(2*sqrt(SHOTS)).

C2 DESIGN (single-qubit Ramsey, Riemann-Siegel formula)
---------------------------------------------------------
  For each t in [C2_T_MIN, C2_T_MAX] and each n = 1..M(t):
      M(t) = max(1, floor(sqrt(t / (2*pi))))
      Circuit: H, RZ(theta(t) - t*ln(n)), H, measure
      P_n(t)  = 0.5 * (1 + cos(theta(t) - t*ln(n)))
      Z_M(t)  = 2 * sum_{n=1}^{M} n^{-1/2} * (2*P_n(t) - 1)
      theta(t) = (t/2)*ln(t/(2*pi)) - t/2 - pi/8 + 1/(48*t)

  Z_M(t) is the Riemann-Siegel approximation to zeta(1/2 + it) * exp(i*theta(t)).
  It is real-valued and its sign changes bracket the nontrivial Riemann zeros.

  Physical meaning: each circuit directly encodes one term of the Euler product
  on the critical line as a quantum phase.  Classical post-processing reconstructs
  Z_M(t) from the measured probabilities without any classical simulation.

KNOWN RIEMANN ZEROS in [10, 35]
---------------------------------
  gamma_1 = 14.134725
  gamma_2 = 21.022040
  gamma_3 = 25.010858
  gamma_4 = 30.424876
  gamma_5 = 32.935062

SCALE
------
  C1 : 20 conditions x C1_N_ALPHA = 320 PUBs
  C2 : ~140 PUBs  (C2_N_T t-points, M(t) Ramsey circuits each)
  Total: ~460 PUBs  -- one IBM job, min_num_qubits=20 backend required
"""

import numpy as np
import math
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------
TOKEN    = "96axnVJAp_PkXhi7mpX8t_CVj1NtzqmHjaApLQ5Pn96Q"
INSTANCE = "crn:v1:bluemix:public:quantum-computing:us-east:a/8420df4c778d45e59489b345c26d2c81:73caf2a1-d677-4d15-b711-f8f3fc73732a::"
CHANNEL  = "ibm_quantum_platform"

SHOTS        = 4096
DRY_RUN      = False

C1_N_ALPHA   = 16    # alpha points per C1 condition (N = 1..20)
C2_N_T       = 100   # t-sweep points for C2
C2_T_MIN     = 10.0
C2_T_MAX     = 35.0

# First 20 primes
PRIMES_20 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

# Known Riemann zeros in range for comparison
RIEMANN_ZEROS = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def f_mertens(p):
    """Mertens weight: f(p) = ln(p)/p"""
    return math.log(p) / p


def F_partial(N):
    """F_N = sum_{k=1}^{N} f(p_k), cumulative Mertens sum over first N primes."""
    return sum(f_mertens(p) for p in PRIMES_20[:N])


def riemann_siegel_theta(t):
    """
    Riemann-Siegel theta function.
    theta(t) = (t/2)*ln(t/(2*pi)) - t/2 - pi/8 + 1/(48*t)
    """
    return (t / 2.0) * math.log(t / (2.0 * math.pi)) - t / 2.0 \
           - math.pi / 8.0 + 1.0 / (48.0 * t)


def M_terms(t):
    """Riemann-Siegel truncation: M = max(1, floor(sqrt(t/(2*pi))))"""
    return max(1, int(math.floor(math.sqrt(t / (2.0 * math.pi)))))


def z_theory(t):
    """Theoretical Z_M(t) using full Riemann-Siegel formula."""
    M     = M_terms(t)
    theta = riemann_siegel_theta(t)
    return 2.0 * sum(n ** (-0.5) * math.cos(theta - t * math.log(n))
                     for n in range(1, M + 1))


# ---------------------------------------------------------------------------
# C1 circuit: N-qubit GHZ accumulator
# ---------------------------------------------------------------------------

def build_c1_circuit(N, alpha, label=""):
    """
    N-qubit GHZ spectrometer encoding F_N = sum_{k=1}^{N} f(p_k).
    P(even parity) = cos^2(pi * F_N * alpha)

    For N=1 this reduces to a single-qubit Ramsey: H-RZ-H.
    """
    qr = QuantumRegister(N, 'q')
    cr = ClassicalRegister(N, 'c')
    qc = QuantumCircuit(qr, cr, name=label or f"c1N{N:02d}a{alpha:.4f}")

    # Entangle: GHZ state (|00..0> + |11..1>) / sqrt(2)
    qc.h(qr[0])
    for i in range(N - 1):
        qc.cx(qr[i], qr[i + 1])

    # Phase kick: qubit k encodes prime p_k at frequency f(p_k)
    for k in range(N):
        phi = 2.0 * math.pi * f_mertens(PRIMES_20[k]) * alpha
        qc.rz(phi, qr[k])

    # Rotate to measurement basis
    for i in range(N):
        qc.h(qr[i])

    qc.measure(qr, cr)
    return qc


# ---------------------------------------------------------------------------
# C2 circuit: single-qubit Ramsey (Riemann-Siegel term)
# ---------------------------------------------------------------------------

def build_c2_circuit(n, t, label=""):
    """
    Single-qubit Ramsey circuit for the n-th Riemann-Siegel term at parameter t.
    Phase: phi = theta(t) - t * ln(n)
    H -> RZ(phi) -> H -> measure
    P(0) = 0.5 * (1 + cos(phi))   =>   cos(phi) = 2*P(0) - 1
    """
    qr = QuantumRegister(1, 'q')
    cr = ClassicalRegister(1, 'c')
    qc = QuantumCircuit(qr, cr, name=label or f"c2n{n}t{t:.3f}")

    phase = riemann_siegel_theta(t) - t * math.log(n)
    qc.h(qr[0])
    qc.rz(phase, qr[0])
    qc.h(qr[0])
    qc.measure(qr, cr)
    return qc


# ---------------------------------------------------------------------------
# Build PUB lists
# ---------------------------------------------------------------------------

def build_c1_pubs(verbose=True):
    """320 PUBs: N=1..20, C1_N_ALPHA alpha points each."""
    pubs = []
    if verbose:
        print("Building C1 (Mertens Accumulator) circuits...")
    for N in range(1, 21):
        F_N   = F_partial(N)
        amax  = round(2.0 / F_N, 6)
        alphas = np.linspace(0.0, amax, C1_N_ALPHA)
        for alpha in alphas:
            theory = math.cos(math.pi * F_N * float(alpha)) ** 2
            qc = build_c1_circuit(N, float(alpha))
            meta = {
                "block":         "c1_accumulator",
                "N":             N,
                "primes":        PRIMES_20[:N],
                "F_N":           F_N,
                "alpha":         float(alpha),
                "alpha_max":     amax,
                "theory_p_even": theory,
            }
            pubs.append((qc, meta))
        if verbose:
            tag = f"  N={N:2d}  F_N={F_N:.5f}  alpha_max={amax:.5f}  ({C1_N_ALPHA} PUBs)"
            print(tag)
    if verbose:
        print(f"  C1 total: {len(pubs)} PUBs")
    return pubs


def build_c2_pubs(verbose=True):
    """~140 PUBs: t in [C2_T_MIN, C2_T_MAX], M(t) Ramsey circuits per t-point."""
    pubs   = []
    t_vals = np.linspace(C2_T_MIN, C2_T_MAX, C2_N_T)
    if verbose:
        print("Building C2 (Riemann Zero Hunter) circuits...")
    for t in t_vals:
        M     = M_terms(float(t))
        theta = riemann_siegel_theta(float(t))
        for n in range(1, M + 1):
            phase     = theta - float(t) * math.log(n)
            theory_p0 = 0.5 * (1.0 + math.cos(phase))
            qc = build_c2_circuit(n, float(t))
            meta = {
                "block":      "c2_riemann",
                "t":          float(t),
                "n":          n,
                "M":          M,
                "theta":      theta,
                "phase":      phase,
                "theory_p0":  theory_p0,
            }
            pubs.append((qc, meta))
    if verbose:
        M_min = M_terms(C2_T_MIN)
        M_max = M_terms(C2_T_MAX)
        print(f"  C2 total: {len(pubs)} PUBs  "
              f"(t=[{C2_T_MIN},{C2_T_MAX}], {C2_N_T} pts, M(t)={M_min}..{M_max})")
    return pubs


def build_all_pubs(verbose=True):
    c1 = build_c1_pubs(verbose=verbose)
    c2 = build_c2_pubs(verbose=verbose)
    all_pubs = c1 + c2
    if verbose:
        print(f"\nTotal: {len(all_pubs)} PUBs  ({len(c1)} C1 + {len(c2)} C2)")
    return all_pubs, len(c1), len(c2)


# ---------------------------------------------------------------------------
# Analysis: decode job results from live IBM job
# ---------------------------------------------------------------------------

def p_from_counts(counts, n_bits):
    """
    From a counts dict (str keys like '001'), return
      n_bits==1: P(bit='0')
      n_bits>1 : P(even parity = even number of '1' characters)
    """
    total = sum(counts.values())
    if n_bits == 1:
        good = counts.get('0', 0)
    else:
        good = sum(v for k, v in counts.items() if k.count('1') % 2 == 0)
    return good / total, total


def analyze_results(job_result, pub_metas):
    """Decode a live job result and attach measured probabilities."""
    results = []
    for idx, meta in enumerate(pub_metas):
        pub    = job_result[idx]
        n_bits = 1 if meta["block"] == "c2_riemann" else meta["N"]
        try:
            counts = pub.data.c.get_counts()
            p_val, n_shots = p_from_counts(counts, n_bits)
        except Exception:
            samples = pub.data.c.get_int_counts()
            total   = sum(samples.values())
            if n_bits == 1:
                good = samples.get(0, 0)
            else:
                good = sum(v for k, v in samples.items()
                           if bin(k).count('1') % 2 == 0)
            p_val   = good / total
            n_shots = total
        row = dict(meta)
        row["p_measured"] = p_val
        row["n_shots"]    = n_shots
        results.append(row)
    return results


# ---------------------------------------------------------------------------
# Summary: C1
# ---------------------------------------------------------------------------

def print_c1_summary(results):
    from scipy.optimize import curve_fit
    from scipy.stats import pearsonr

    bell = [r for r in results if r["block"] == "c1_accumulator"]
    print("\n" + "=" * 72)
    print("C1: MERTENS ACCUMULATOR -- RESULTS")
    print("=" * 72)
    print(f"  {'N':>3}  {'F_theory':>10}  {'F_fit':>10}  {'err%':>7}  "
          f"{'amp':>6}  {'r':>7}  {'resolved?':>10}")

    def model(a, FN, amp, base):
        return base + amp * np.cos(math.pi * FN * a) ** 2

    c1_spectrum = []
    for N in range(1, 21):
        rows = sorted([r for r in bell if r["N"] == N], key=lambda x: x["alpha"])
        if not rows:
            continue
        F_N      = rows[0]["F_N"]
        alphas   = np.array([r["alpha"]      for r in rows])
        measured = np.array([r["p_measured"] for r in rows])

        try:
            popt, _ = curve_fit(model, alphas, measured,
                                p0=[F_N, 0.85, 0.05],
                                bounds=([0.0, 0.0, 0.0], [20.0, 1.0, 1.0]),
                                maxfev=30000)
            F_fit, amp, base = popt
            err_pct  = (F_fit - F_N) / F_N * 100.0
            r_fit, _ = pearsonr(measured, model(alphas, *popt))
            resolved = "YES" if r_fit > 0.95 else ("MARGINAL" if r_fit > 0.80 else "NO")
            print(f"  {N:>3}  {F_N:>10.6f}  {F_fit:>10.6f}  {err_pct:>+7.2f}%  "
                  f"{amp:>6.3f}  {r_fit:>7.4f}  {resolved:>10}")
            c1_spectrum.append({
                "N":        N,
                "F_theory": F_N,
                "F_fit":    float(F_fit),
                "err_pct":  float(err_pct),
                "amp":      float(amp),
                "r_fit":    float(r_fit),
                "resolved": resolved,
            })
        except Exception as e:
            print(f"  {N:>3}  {F_N:>10.6f}  {'FIT FAIL':>10}  [{e}]")

    yes = [s for s in c1_spectrum if s["resolved"] == "YES"]
    if c1_spectrum:
        mean_err = np.mean([abs(s["err_pct"]) for s in c1_spectrum])
        mean_amp = np.mean([s["amp"] for s in c1_spectrum])
        print(f"\n  Resolved: {len(yes)}/{len(c1_spectrum)},  "
              f"mean |err%| = {mean_err:.2f}%,  "
              f"mean amp = {mean_amp:.3f}")
    print("=" * 72)
    return c1_spectrum


# ---------------------------------------------------------------------------
# Summary: C2 -- reconstruct Z_M(t) and detect sign changes
# ---------------------------------------------------------------------------

def print_c2_summary(results):
    from scipy.stats import pearsonr

    c2 = [r for r in results if r["block"] == "c2_riemann"]
    if not c2:
        print("No C2 results.")
        return []

    # Group by t-value and reconstruct Z_M(t)
    t_set  = sorted(set(r["t"] for r in c2))
    Z_data = []
    for t in t_set:
        pts = sorted([r for r in c2 if r["t"] == t], key=lambda x: x["n"])
        Z_meas = 0.0
        Z_th   = 0.0
        for r in pts:
            n     = r["n"]
            P_n   = r["p_measured"]
            Z_meas += 2.0 * n ** (-0.5) * (2.0 * P_n - 1.0)
            Z_th   += 2.0 * n ** (-0.5) * math.cos(r["phase"])
        Z_data.append({
            "t":          t,
            "Z_measured": Z_meas,
            "Z_theory":   Z_th,
            "M":          pts[0]["M"],
        })

    Zm = np.array([d["Z_measured"] for d in Z_data])
    Zt = np.array([d["Z_theory"]   for d in Z_data])
    r_z, _ = pearsonr(Zm, Zt)

    # Sign changes
    def find_crossings(vals, data, key):
        crossings = []
        for i in range(len(data) - 1):
            v1, v2 = vals[i], vals[i + 1]
            if v1 * v2 < 0.0:
                t1, t2 = data[i]["t"], data[i + 1]["t"]
                t_cross = t1 + (t2 - t1) * abs(v1) / (abs(v1) + abs(v2))
                crossings.append(float(t_cross))
        return crossings

    meas_cross   = find_crossings(Zm, Z_data, "Z_measured")
    theory_cross = find_crossings(Zt, Z_data, "Z_theory")

    print("\n" + "=" * 72)
    print("C2: RIEMANN ZERO HUNTER -- RESULTS")
    print("=" * 72)
    print(f"  Z_M(t) correlation (measured vs theory): r = {r_z:.4f}")
    print(f"\n  Theory sign changes:   "
          f"{[f'{t:.3f}' for t in theory_cross]}")
    print(f"  Measured sign changes: "
          f"{[f'{t:.3f}' for t in meas_cross]}")
    print(f"\n  Known Riemann zeros:   "
          f"{[f'{g:.3f}' for g in RIEMANN_ZEROS]}")

    if meas_cross:
        print(f"\n  {'Detected t':>12}  {'Nearest zero':>14}  {'Delta t':>8}")
        for t_sc in meas_cross:
            nearest = min(RIEMANN_ZEROS, key=lambda g: abs(g - t_sc))
            delta   = t_sc - nearest
            print(f"  {t_sc:>12.4f}  {nearest:>14.6f}  {delta:>+8.4f}")

    print("=" * 72)
    return Z_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Branch C: Riemann Branch (C1+C2)")
    ap.add_argument("--dry-run",  action="store_true", help="Build circuits, do not submit")
    ap.add_argument("--c1-only",  action="store_true", help="Submit only C1 circuits")
    ap.add_argument("--c2-only",  action="store_true", help="Submit only C2 circuits")
    args = ap.parse_args()

    print("=" * 72)
    print("Branch C: Riemann Branch")
    print("C1 Mertens Accumulator + C2 Riemann Zero Hunter")
    print("=" * 72)
    print()

    all_pubs, n_c1, n_c2 = build_all_pubs(verbose=True)

    if args.c1_only:
        submit_pubs = all_pubs[:n_c1]
        print(f"\nC1-only mode: {len(submit_pubs)} PUBs")
    elif args.c2_only:
        submit_pubs = all_pubs[n_c1:]
        print(f"\nC2-only mode: {len(submit_pubs)} PUBs")
    else:
        submit_pubs = all_pubs

    circuits    = [p[0] for p in submit_pubs]
    metas       = [p[1] for p in submit_pubs]
    max_qubits  = max(c.num_qubits for c in circuits)
    raw_depths  = [c.depth() for c in circuits]

    print(f"\nMax qubits in any circuit : {max_qubits}")
    print(f"Raw circuit depths        : min={min(raw_depths)}, max={max(raw_depths)}, "
          f"mean={np.mean(raw_depths):.1f}")

    if args.dry_run:
        print(f"\nDRY RUN -- {len(circuits)} circuits built, NOT submitted to IBM.")
        print(f"  C1: 20 conditions x {C1_N_ALPHA} alpha points  = {n_c1} PUBs")
        print(f"  C2: {C2_N_T} t-points, M(t) terms each         = {n_c2} PUBs")
        return

    print("\nConnecting to IBM Quantum...")
    service = QiskitRuntimeService(channel=CHANNEL, instance=INSTANCE, token=TOKEN)
    backend = service.least_busy(min_num_qubits=max_qubits, simulator=False)
    print(f"Backend: {backend.name}")

    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pm         = generate_preset_pass_manager(optimization_level=1, backend=backend)
    transpiled = pm.run(circuits)
    depths     = [c.depth() for c in transpiled]
    print(f"Transpiled. Depths: min={min(depths)}, max={max(depths)}, "
          f"mean={np.mean(depths):.1f}")

    sampler  = Sampler(mode=backend)
    pub_list = [(tc,) for tc in transpiled]
    print(f"\nSubmitting {len(pub_list)} PUBs @ {SHOTS} shots each...")
    job = sampler.run(pub_list, shots=SHOTS)
    jid = job.job_id()

    print(f"\nJob ID: {jid}")
    print(f"\nRetrieve with:")
    print(f"  python retrieve_c1c2.py {jid}")
    print(f"  python retrieve_c1c2.py {jid} --local   (parse local JSON once done)")


if __name__ == "__main__":
    main()
