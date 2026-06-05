"""
simulate_c3_aer.py -- C3: Summatory Harmonies (Möbius-weighted GHZ)
=====================================================================
Simulates the Möbius-weighted Mertens sum S(N) = sum_{k=2}^{N} mu(k)*f(k)
using local Qiskit Aer.  No IBM credentials required.

CIRCUIT DESIGN (from manuscript supplementary)
-----------------------------------------------
For each condition N, let {k_i} be the squarefree integers in [2, N].
Each k_i gets one qubit with signed phase phi_i = 2*pi * mu(k_i) * f(k_i) * alpha.

  H(q0)
  CNOT chain: q0 -> q1 -> ... -> q_{m-1}     (create GHZ)
  RZ(phi_i, q_i) for i = 0 .. m-1             (signed phase kicks)
  CNOT chain reversed: q_{m-2}->q_{m-1}, ..., q0->q1  (disentangle)
  H(q0)
  measure q0

  P(0) = cos^2(pi * S(N) * alpha)
       = cos^2(pi * |S(N)| * alpha)   [since cos^2 is even]

WHAT S(N) IS
-------------
S(N) oscillates near -1 as N grows.  This is the partial sum of the
Dirichlet series sum_{k=1}^inf mu(k)*ln(k)/k^s at s=1, which connects
to -(zeta'(1)/zeta(1)^2) -- a fingerprint of the Riemann zeta function.
As N->inf, S(N) -> -1 (equivalent to the Prime Number Theorem).

CONDITIONS
-----------
N = 2..30: one GHZ condition per N, growing from 1 to 18 qubits.
alpha_max = 2 / |S(N)|  =>  4*pi total phase (equal-power design).
C3_N_ALPHA = 16 points per condition.

KNOWN RESULT (manuscript claim)
---------------------------------
S(30) = -1.106673  =>  |S(30)| = 1.106673
Ideal simulation recovers this with 0.0000% error.
Shot-noise simulation: +0.1118% error.

Modes: --ideal (default), --noisy, --shots N, --n-max M (default 30)
"""

import sys
import os
import json
import math
import time
import datetime
import argparse
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Möbius function (pure Python, no sympy required)
# ---------------------------------------------------------------------------

def mobius(n):
    """
    Möbius function mu(n):
      0  if n has a squared prime factor
     +1  if n is squarefree with an even number of prime factors
     -1  if n is squarefree with an odd number of prime factors
    """
    if n == 1:
        return 1
    n_orig = n
    n_factors = 0
    d = 2
    while d * d <= n:
        if n % d == 0:
            n_factors += 1
            n //= d
            if n % d == 0:
                return 0          # squared prime factor
        d += 1
    if n > 1:
        n_factors += 1
    return (-1) ** n_factors


def f_mertens(k):
    """Mertens weight f(k) = ln(k)/k"""
    return math.log(k) / k


# ---------------------------------------------------------------------------
# Precompute S(N) table for N = 2..30
# ---------------------------------------------------------------------------

def compute_S_table(n_max=30):
    """
    Returns two dicts:
      S_table[N]    = cumulative sum S(N) = sum_{k=2}^{N} mu(k)*f(k)
      terms[N]      = list of (k, mu_k, f_k) for squarefree k in [2..N]
                      (the qubits needed for the GHZ circuit at condition N)
    """
    S_running = 0.0
    squarefree_ks = []   # accumulate squarefree k values seen so far
    S_table = {}
    terms   = {}
    for k in range(2, n_max + 1):
        mu_k = mobius(k)
        if mu_k != 0:
            squarefree_ks.append((k, mu_k, f_mertens(k)))
            S_running += mu_k * f_mertens(k)
        S_table[k] = S_running
        terms[k]   = list(squarefree_ks)   # snapshot up to this N
    return S_table, terms


# ---------------------------------------------------------------------------
# Circuit builder
# ---------------------------------------------------------------------------

def build_c3_circuit(qubit_terms, alpha, label=""):
    """
    qubit_terms: list of (k, mu_k, f_k) -- one entry per qubit
    alpha      : sweep parameter
    Returns a QuantumCircuit measuring P(q0=0) = cos^2(pi * S * alpha)
    """
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

    m = len(qubit_terms)
    if m == 0:
        return None

    qr = QuantumRegister(m, 'q')
    cr = ClassicalRegister(1, 'c')      # measure only q0
    qc = QuantumCircuit(qr, cr, name=label or f"c3m{m}a{alpha:.4f}")

    # Create GHZ state
    qc.h(qr[0])
    for i in range(m - 1):
        qc.cx(qr[i], qr[i + 1])

    # Signed phase kicks: mu(k)*f(k)*2*pi*alpha on each qubit
    for i, (k, mu_k, f_k) in enumerate(qubit_terms):
        phi = 2.0 * math.pi * mu_k * f_k * alpha
        qc.rz(phi, qr[i])

    # Disentangle (reverse CNOT chain)
    for i in range(m - 2, -1, -1):
        qc.cx(qr[i], qr[i + 1])

    # Interfere on q0 and measure
    qc.h(qr[0])
    qc.measure(qr[0], cr[0])
    return qc


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------

def make_noise_model(p1q=0.001, p2q=0.010, p_readout=0.010):
    from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
    nm  = NoiseModel()
    e1  = depolarizing_error(p1q, 1)
    e2  = depolarizing_error(p2q, 2)
    nm.add_all_qubit_quantum_error(e1, ['h', 'rz', 'id'])
    nm.add_all_qubit_quantum_error(e2, ['cx', 'ecr'])
    re  = ReadoutError([[1 - p_readout, p_readout], [p_readout, 1 - p_readout]])
    nm.add_all_qubit_readout_error(re)
    return nm


# ---------------------------------------------------------------------------
# Run a batch of circuits through Aer
# ---------------------------------------------------------------------------

def run_aer_batch(circuits, shots, noise_model=None):
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    kwargs = {}
    if noise_model:
        kwargs['noise_model'] = noise_model
    sim   = AerSimulator(**kwargs)
    trans = transpile(circuits, sim, optimization_level=0)
    job   = sim.run(trans, shots=shots)
    res   = job.result()
    return [res.get_counts(i) for i in range(len(circuits))]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def print_c3_summary(results):
    from scipy.optimize import curve_fit
    from scipy.stats import pearsonr

    print("\n" + "=" * 74)
    print("C3: SUMMATORY HARMONIES (Möbius-Mertens) -- RESULTS")
    print("=" * 74)
    print(f"  {'N':>3}  {'qubits':>7}  {'|S|_theory':>12}  {'|S|_fit':>12}"
          f"  {'err%':>7}  {'r':>7}  {'resolved?':>10}")

    spectrum = []
    for N in sorted(set(r["N"] for r in results)):
        rows = sorted([r for r in results if r["N"] == N], key=lambda x: x["alpha"])
        if not rows:
            continue
        S_th  = rows[0]["S_N"]           # signed theoretical value
        absS  = abs(S_th)
        n_q   = rows[0]["n_qubits"]
        alphas   = np.array([r["alpha"]      for r in rows])
        measured = np.array([r["p_measured"] for r in rows])

        def model(a, F, amp, base):
            return base + amp * np.cos(math.pi * F * a) ** 2

        try:
            popt, _ = curve_fit(model, alphas, measured,
                                p0=[absS, 0.85, 0.05],
                                bounds=([0.0, 0.0, 0.0], [10.0, 1.0, 1.0]),
                                maxfev=30000)
            F_fit, amp, base = popt
            err_pct  = (F_fit - absS) / absS * 100.0
            r_fit, _ = pearsonr(measured, model(alphas, *popt))
            resolved = "YES" if r_fit > 0.95 else ("MARGINAL" if r_fit > 0.80 else "NO")
            print(f"  {N:>3}  {n_q:>7}  {absS:>12.6f}  {F_fit:>12.6f}"
                  f"  {err_pct:>+7.3f}%  {r_fit:>7.4f}  {resolved:>10}")
            spectrum.append({
                "N": N, "S_theory": S_th, "absS_theory": absS,
                "absS_fit": float(F_fit), "err_pct": float(err_pct),
                "amp": float(amp), "r_fit": float(r_fit),
                "resolved": resolved, "n_qubits": n_q,
            })
        except Exception as e:
            print(f"  {N:>3}  {n_q:>7}  {absS:>12.6f}  FIT FAIL  [{e}]")

    yes = [s for s in spectrum if s["resolved"] == "YES"]
    if spectrum:
        mean_err = np.mean([abs(s["err_pct"]) for s in spectrum])
        mean_amp = np.mean([s["amp"] for s in spectrum])
        print(f"\n  Resolved: {len(yes)}/{len(spectrum)},  "
              f"mean |err%| = {mean_err:.3f}%,  mean amp = {mean_amp:.3f}")

    # Highlight S(30)
    s30 = next((s for s in spectrum if s["N"] == 30), None)
    if s30:
        print(f"\n  S(30) check:")
        print(f"    Theory  |S(30)| = {s30['absS_theory']:.6f}")
        print(f"    Fitted  |S(30)| = {s30['absS_fit']:.6f}")
        print(f"    Error           = {s30['err_pct']:+.4f}%")
        print(f"    Manuscript says: 0.0000% (ideal), +0.1118% (noisy)")

    # Show S(N) trajectory
    print(f"\n  S(N) trajectory (oscillates toward -1 as N -> inf):")
    for s in spectrum[::5]:     # every 5th entry to keep output compact
        bar = "#" * int(abs(s["S_theory"]) * 20)
        print(f"    N={s['N']:2d}  S={s['S_theory']:+.5f}  [{bar:<25}]")

    print("=" * 74)
    return spectrum


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def simulate(args):
    tag       = "noisy" if args.noisy else "ideal"
    shots     = args.shots
    n_max     = args.n_max
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 74)
    print(f"C3: Summatory Harmonies -- Aer Simulation  [{tag.upper()}, {shots} shots]")
    print(f"Möbius-weighted Mertens sum S(N), N=2..{n_max}")
    print("=" * 74)
    print()

    S_table, terms_table = compute_S_table(n_max)

    # Print condition summary
    print(f"{'N':>4}  {'squarefree k count':>20}  {'S(N)':>10}  {'|S(N)|':>8}")
    for N in range(2, n_max + 1):
        n_q = len(terms_table[N])
        print(f"{N:>4}  {n_q:>20}  {S_table[N]:>+10.6f}  {abs(S_table[N]):>8.6f}")

    print()
    noise_model = make_noise_model() if args.noisy else None
    if args.noisy:
        print("Noise model: depolarizing p1q=0.001, p2q=0.010, readout=0.010")
    else:
        print("Noise model: none (ideal)")

    # Build and run all circuits
    all_results = []
    n_alpha     = 16

    print(f"\nSimulating {n_max - 1} conditions x {n_alpha} alpha points...")
    t_total = time.time()

    for N in range(2, n_max + 1):
        S_N     = S_table[N]
        absS    = abs(S_N)
        qt      = terms_table[N]
        n_q     = len(qt)
        amax    = round(2.0 / absS, 6)
        alphas  = np.linspace(0.0, amax, n_alpha)

        circuits = []
        metas    = []
        for alpha in alphas:
            qc = build_c3_circuit(qt, float(alpha))
            circuits.append(qc)
            theory = math.cos(math.pi * S_N * float(alpha)) ** 2
            metas.append({
                "N": N, "n_qubits": n_q,
                "S_N": S_N, "absS": absS,
                "alpha": float(alpha), "alpha_max": amax,
                "theory_p0": theory,
            })

        t0       = time.time()
        counts_l = run_aer_batch(circuits, shots, noise_model)
        elapsed  = time.time() - t0

        for meta, counts in zip(metas, counts_l):
            total = sum(counts.values())
            p0    = counts.get('0', 0) / total
            row   = dict(meta)
            row["p_measured"] = p0
            row["n_shots"]    = total
            all_results.append(row)

        print(f"  N={N:2d}  {n_q:2d} qubits  S={S_N:+.5f}  "
              f"alpha_max={amax:.4f}  {elapsed:.2f}s")

    elapsed_total = time.time() - t_total
    print(f"\nTotal simulation time: {elapsed_total:.1f}s")

    # Analysis
    spectrum = print_c3_summary(all_results)

    # Save outputs
    out = os.path.join(script_dir, f"sim_c3_results_{tag}_{timestamp}.json")
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved  : {os.path.basename(out)}")

    if spectrum:
        sp_out = os.path.join(script_dir, f"sim_c3_spectrum_{tag}_{timestamp}.json")
        with open(sp_out, 'w') as f:
            json.dump(spectrum, f, indent=2)
        print(f"Spectrum saved : {os.path.basename(sp_out)}")

    print("\nDone.")
    return all_results, spectrum


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="C3 Summatory Harmonies: Möbius-weighted Mertens sum on Aer"
    )
    ap.add_argument("--ideal",   action="store_true", help="Noiseless (default)")
    ap.add_argument("--noisy",   action="store_true", help="IBM-realistic noise")
    ap.add_argument("--shots",   type=int, default=2048, help="Shots per circuit")
    ap.add_argument("--n-max",   type=int, default=30,
                    help="Upper limit of sum, conditions N=2..n-max (default 30)")
    args = ap.parse_args()

    if not args.noisy:
        args.ideal = True

    simulate(args)


if __name__ == "__main__":
    main()
