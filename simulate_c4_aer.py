"""
simulate_c4_aer.py -- C4: Deep Decoherence Scan (T1/T2 Thermal Model)
=======================================================================
Measures how GHZ oscillation amplitude A(tau) decays as idle delay time
tau increases, for three distinct weight ensembles:

  prime-weighted  : phases from f(p) = ln(p)/p for primes p in [2,11]
  composite-weighted: phases from f(n) for composites n in [4,6,8,9,10]
  random-weighted : random phases drawn from the same magnitude range

Manuscript claim:
  "All remained usable (A > 0.90) up to ~1000 ns.
   Prime-weighted sums showed slightly faster decay, indicating no
   anomalous coherence advantage beyond spectral isolation."

CIRCUIT DESIGN
--------------
5-qubit GHZ spectrometer with mid-circuit idle delay:

  H(q0)
  CNOT chain: q0->q1->...->q4         [create GHZ state]
  RZ(2*pi*f_k*alpha, q_k) for k=0..4  [phase kicks]
  id(q_k) for k=0..4                  [idle -- thermal_relaxation applied here]
  H(q_k) for k=0..4                   [basis rotation]
  measure all                          [parity readout]

  P(even parity) = A(tau) * cos^2(pi*F*alpha) + baseline

NOISE MODEL
-----------
  thermal_relaxation_error(T1, T2, tau) applied to identity gates.
  Default IBM-realistic values:
    T1 = 150000 ns (150 us)
    T2 = 100000 ns (100 us)

  Expected GHZ amplitude decay: A(tau) ~ exp(-5*tau/T2) for N=5 qubits.
  At tau=1000ns, T2=100000ns: A ~ exp(-0.05) ~ 0.951  =>  A > 0.90  (check)
  At tau=2000ns:               A ~ exp(-0.10) ~ 0.905  =>  A > 0.90  (check)

SCALE
-----
  3 ensembles x 20 tau points x 10 alpha points = 600 circuits.
  Expected runtime: ~60-120s on Aer (one noise model per tau value).
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
# IBM-realistic hardware parameters (nanoseconds)
# ---------------------------------------------------------------------------
T1_DEFAULT  = 150_000   # ns  (150 us)
T2_DEFAULT  = 100_000   # ns  (100 us)

TAU_MAX     = 2_000     # ns
N_TAU       = 20        # tau sweep points
N_ALPHA     = 10        # alpha points per (tau, ensemble)
N_QUBITS    = 5         # qubits per GHZ circuit

# ---------------------------------------------------------------------------
# Weight ensembles
# ---------------------------------------------------------------------------

def f_mertens(n):
    return math.log(n) / n

def make_ensembles(seed=42):
    """
    Returns dict: ensemble_name -> list of 5 weights (f values)
    """
    primes     = [2, 3, 5, 7, 11]
    composites = [4, 6, 8, 9, 10]

    prime_w = [f_mertens(p) for p in primes]
    comp_w  = [f_mertens(c) for c in composites]

    rng = np.random.default_rng(seed)
    f_min = min(min(prime_w), min(comp_w))
    f_max = max(max(prime_w), max(comp_w))
    rand_w = rng.uniform(f_min, f_max, size=N_QUBITS).tolist()

    return {
        "prime":     prime_w,
        "composite": comp_w,
        "random":    rand_w,
    }

# ---------------------------------------------------------------------------
# Circuit builder
# ---------------------------------------------------------------------------

def build_ghz_circuit(weights, alpha, tau_ns, label=""):
    """
    N-qubit GHZ with phase kicks and mid-circuit idle delay.
    P(even parity) = A(tau) * cos^2(pi * F * alpha) + baseline
    where F = sum(weights) and A(tau) decays with thermal relaxation.
    """
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

    N  = len(weights)
    qr = QuantumRegister(N, 'q')
    cr = ClassicalRegister(N, 'c')
    qc = QuantumCircuit(qr, cr, name=label or f"c4a{alpha:.3f}t{tau_ns}")

    # Create GHZ
    qc.h(qr[0])
    for i in range(N - 1):
        qc.cx(qr[i], qr[i + 1])

    # Phase kicks
    for k, w in enumerate(weights):
        qc.rz(2.0 * math.pi * w * alpha, qr[k])

    # Idle delay: identity gates (thermal_relaxation applied here by noise model)
    if tau_ns > 0:
        for i in range(N):
            qc.id(qr[i])

    # Basis rotation
    for i in range(N):
        qc.h(qr[i])

    qc.measure(qr, cr)
    return qc


# ---------------------------------------------------------------------------
# Noise model for a specific tau
# ---------------------------------------------------------------------------

def make_thermal_noise(tau_ns, T1_ns, T2_ns):
    """
    Thermal relaxation error channel for idle time tau_ns.
    Returns a NoiseModel with the error applied to identity gates.
    """
    from qiskit_aer.noise import NoiseModel, thermal_relaxation_error

    nm    = NoiseModel()
    error = thermal_relaxation_error(T1_ns, T2_ns, tau_ns)
    nm.add_all_qubit_quantum_error(error, ['id'])
    return nm


# ---------------------------------------------------------------------------
# Run batch
# ---------------------------------------------------------------------------

def run_aer(circuits, shots, noise_model=None):
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    kwargs = {}
    if noise_model is not None:
        kwargs['noise_model'] = noise_model

    # Must allow 'id' gate through transpiler
    sim   = AerSimulator(**kwargs)
    trans = transpile(circuits, sim,
                      optimization_level=0,
                      basis_gates=['h', 'cx', 'rz', 'id', 'measure', 'reset'])
    job   = sim.run(trans, shots=shots)
    result = job.result()
    return [result.get_counts(i) for i in range(len(circuits))]


# ---------------------------------------------------------------------------
# Fit oscillation amplitude at one (tau, ensemble)
# ---------------------------------------------------------------------------

def fit_amplitude(alphas, p_even_vals, F_theory):
    """
    Fit: p_even = base + amp * cos^2(pi * F * alpha)
    Returns (amp, F_fit, r) or (None, None, None) on failure.
    """
    from scipy.optimize import curve_fit
    from scipy.stats import pearsonr

    def model(a, amp, base):
        return base + amp * np.cos(math.pi * F_theory * a) ** 2

    try:
        popt, _ = curve_fit(model, alphas, p_even_vals,
                            p0=[0.9, 0.05],
                            bounds=([0.0, 0.0], [1.0, 1.0]),
                            maxfev=10000)
        amp, base = popt
        pred     = model(alphas, *popt)
        r, _     = pearsonr(p_even_vals, pred)
        return float(amp), float(r)
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def simulate(args):
    T1_ns   = args.t1
    T2_ns   = args.t2
    shots   = args.shots
    tag     = "c4_thermal"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 72)
    print("C4: Deep Decoherence Scan -- Aer Simulation")
    print(f"T1={T1_ns/1000:.0f} us, T2={T2_ns/1000:.0f} us, {N_QUBITS} qubits, {shots} shots")
    print("=" * 72)
    print()

    ensembles = make_ensembles()

    print("Ensemble summary:")
    for name, weights in ensembles.items():
        F = sum(weights)
        print(f"  {name:<12}  weights={[f'{w:.4f}' for w in weights]}  F={F:.5f}")

    print()

    # GHZ amplitude decay theory: A(tau) = exp(-N * tau / T2) (ideal GHZ)
    print(f"Theoretical A(tau) for ideal {N_QUBITS}-qubit GHZ [exp(-{N_QUBITS}*tau/T2)]:")
    for tau_check in [0, 500, 1000, 1500, 2000]:
        A_th = math.exp(-N_QUBITS * tau_check / T2_ns)
        print(f"  tau={tau_check:5d} ns  A_theory={A_th:.4f}  "
              f"{'OK (>0.90)' if A_th > 0.90 else 'DEGRADED'}")
    print()

    tau_vals = np.linspace(0, TAU_MAX, N_TAU)

    # Store results: {ensemble: {tau: {alpha: p_even}}}
    all_records   = []
    decay_data    = {name: [] for name in ensembles}

    t_total = time.time()

    for tau in tau_vals:
        tau_ns_val = float(tau)
        noise      = make_thermal_noise(tau_ns_val, T1_ns, T2_ns) if tau_ns_val > 0 else None

        # Build all circuits for this tau (all ensembles, all alpha)
        batch_circuits = []
        batch_meta     = []

        for ens_name, weights in ensembles.items():
            F    = sum(weights)
            amax = round(2.0 / F, 6)
            alphas = np.linspace(0.0, amax, N_ALPHA)
            for alpha in alphas:
                qc = build_ghz_circuit(weights, float(alpha), tau_ns_val)
                batch_circuits.append(qc)
                batch_meta.append({
                    "ensemble": ens_name,
                    "tau_ns":   tau_ns_val,
                    "alpha":    float(alpha),
                    "alpha_max": amax,
                    "F":        F,
                    "weights":  weights,
                    "theory_A": math.exp(-N_QUBITS * tau_ns_val / T2_ns),
                    "theory_p_even": math.cos(math.pi * F * float(alpha)) ** 2
                                     * math.exp(-N_QUBITS * tau_ns_val / T2_ns)
                                     + 0.5 * (1 - math.exp(-N_QUBITS * tau_ns_val / T2_ns)),
                })

        t0       = time.time()
        counts_l = run_aer(batch_circuits, shots, noise)
        elapsed  = time.time() - t0

        # Decode and record
        for meta, counts in zip(batch_meta, counts_l):
            total = sum(counts.values())
            n_even = sum(v for k, v in counts.items() if k.count('1') % 2 == 0)
            p_even = n_even / total
            row = dict(meta)
            row["p_measured"] = p_even
            row["n_shots"]    = total
            all_records.append(row)

        # Fit amplitude for each ensemble at this tau
        tau_str = f"tau={tau_ns_val:6.0f}ns"
        amp_str = []
        for ens_name, weights in ensembles.items():
            F      = sum(weights)
            amax   = round(2.0 / F, 6)
            alphas = np.linspace(0.0, amax, N_ALPHA)
            rows   = [r for r in all_records
                      if r["ensemble"] == ens_name and r["tau_ns"] == tau_ns_val]
            p_vals = np.array([r["p_measured"] for r in rows])
            amp, r = fit_amplitude(alphas, p_vals, F)
            if amp is not None:
                decay_data[ens_name].append({
                    "tau_ns": tau_ns_val,
                    "amp":    amp,
                    "r_fit":  r,
                    "theory_A": math.exp(-N_QUBITS * tau_ns_val / T2_ns),
                })
                amp_str.append(f"{ens_name}:{amp:.3f}")

        print(f"  {tau_str}  {' | '.join(amp_str)}  ({elapsed:.1f}s)")

    elapsed_total = time.time() - t_total
    print(f"\nTotal time: {elapsed_total:.1f}s")

    # Summary table
    print("\n" + "=" * 72)
    print("C4: DECOHERENCE SCAN -- AMPLITUDE A(tau) BY ENSEMBLE")
    print("=" * 72)
    print(f"  {'tau (ns)':>10}  {'A_theory':>10}  "
          f"{'A_prime':>10}  {'A_composite':>13}  {'A_random':>10}")

    for i, tau in enumerate(tau_vals):
        A_th = math.exp(-N_QUBITS * float(tau) / T2_ns)
        row_parts = [f"  {float(tau):>10.0f}  {A_th:>10.4f}"]
        for ens_name in ["prime", "composite", "random"]:
            pts = [d for d in decay_data[ens_name] if d["tau_ns"] == float(tau)]
            if pts:
                row_parts.append(f"  {pts[0]['amp']:>10.4f}")
            else:
                row_parts.append(f"  {'--':>10}")
        print("".join(row_parts))

    # Exponential fit to A(tau) decay for each ensemble
    print(f"\n  Exponential decay fits: A(tau) = exp(-tau / tau_decay)")
    print(f"  {'Ensemble':>12}  {'tau_decay (ns)':>16}  "
          f"{'T2_eff (ns)':>14}  {'A@1000ns':>10}  {'A@2000ns':>10}")
    print(f"  {'(theory)':>12}  {T2_ns/N_QUBITS:>16.0f}  "
          f"{T2_ns:>14.0f}  "
          f"{math.exp(-N_QUBITS*1000/T2_ns):>10.4f}  "
          f"{math.exp(-N_QUBITS*2000/T2_ns):>10.4f}")

    for ens_name in ["prime", "composite", "random"]:
        pts      = decay_data[ens_name]
        tau_arr  = np.array([p["tau_ns"] for p in pts])
        amp_arr  = np.array([p["amp"]    for p in pts])
        amp_arr  = np.clip(amp_arr, 1e-6, 1.0)

        try:
            from scipy.optimize import curve_fit
            def exp_decay(t, tau_d):
                return np.exp(-t / tau_d)
            popt, _ = curve_fit(exp_decay, tau_arr, amp_arr,
                                p0=[T2_ns / N_QUBITS], maxfev=5000)
            tau_decay = popt[0]
            T2_eff    = tau_decay * N_QUBITS
            A_1000    = math.exp(-1000 / tau_decay)
            A_2000    = math.exp(-2000 / tau_decay)
            usable_to = -tau_decay * math.log(0.90)
            print(f"  {ens_name:>12}  {tau_decay:>16.0f}  "
                  f"{T2_eff:>14.0f}  "
                  f"{A_1000:>10.4f}  {A_2000:>10.4f}"
                  f"  [usable to ~{usable_to:.0f} ns]")
        except Exception as e:
            print(f"  {ens_name:>12}  FIT FAILED [{e}]")

    print(f"\n  Manuscript claims: A > 0.90 at tau=1000ns for all ensembles.")
    print(f"  Prime-weighted decay slightly faster than composites (spectral isolation,")
    print(f"  not anomalous coherence advantage).")
    print("=" * 72)

    # Save
    out = os.path.join(script_dir, f"sim_c4_results_{timestamp}.json")
    with open(out, 'w') as f:
        json.dump(all_records, f, indent=2, default=str)
    print(f"\nFull results : {os.path.basename(out)}")

    decay_out = os.path.join(script_dir, f"sim_c4_decay_{timestamp}.json")
    with open(decay_out, 'w') as f:
        json.dump(decay_data, f, indent=2, default=str)
    print(f"Decay data   : {os.path.basename(decay_out)}")

    print("\nDone.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="C4: Decoherence scan -- T1/T2 thermal model on Aer"
    )
    ap.add_argument("--t1",    type=float, default=T1_DEFAULT,
                    help=f"T1 relaxation time in ns (default {T1_DEFAULT})")
    ap.add_argument("--t2",    type=float, default=T2_DEFAULT,
                    help=f"T2 dephasing time in ns (default {T2_DEFAULT})")
    ap.add_argument("--shots", type=int,   default=2048,
                    help="Shots per circuit (default 2048)")
    args = ap.parse_args()
    simulate(args)


if __name__ == "__main__":
    main()
