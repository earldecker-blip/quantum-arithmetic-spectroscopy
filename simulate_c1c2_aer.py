"""
simulate_c1c2_aer.py -- Local Aer simulation of Branch C (C1+C2)
=================================================================
Runs all 460 Branch C PUBs on a local Qiskit Aer simulator.
No IBM credentials or quantum credits required.

Modes
------
  --ideal  (default)
      Noiseless shot simulation.  Validates that every circuit encodes the
      right phase and that the retrieve pipeline (fitting, sign-change
      detection) works correctly before spending hardware time.

  --noisy
      Adds a realistic IBM-style depolarizing + readout noise model:
        single-qubit gates (H, RZ): depolarizing p=0.001
        two-qubit gates (CX)      : depolarizing p=0.010
        readout                   : p_error=0.010 per qubit
      This predicts how C1 GHZ fidelity degrades with N and how robust
      C2 sign-change detection is against typical gate noise.

  --shots N  (default 2048)
      Shots per circuit.  2048 is fast; use 4096 to match hardware precision.

  --c1-only / --c2-only
      Simulate only one sub-experiment.

Outputs
--------
  sim_results_ideal_<timestamp>.json   or
  sim_results_noisy_<timestamp>.json
  -- same format as c1c2_results_*.json from retrieve_c1c2.py

The script calls print_c1_summary() and print_c2_summary() from
experiment_c1c2_riemann_branch.py, so the output tables are identical
to what you will see after the real IBM job.
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
sys.path.insert(0, script_dir)

from experiment_c1c2_riemann_branch import (
    C1_N_ALPHA, C2_N_T, C2_T_MIN, C2_T_MAX,
    PRIMES_20, RIEMANN_ZEROS,
    F_partial, riemann_siegel_theta, M_terms,
    build_c1_pubs, build_c2_pubs,
    print_c1_summary, print_c2_summary,
)


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------

def make_noise_model(p1q=0.001, p2q=0.010, p_readout=0.010):
    """
    Simple depolarizing + readout noise model approximating IBM Eagle/Heron.
      p1q      : single-qubit gate depolarizing probability
      p2q      : two-qubit gate depolarizing probability
      p_readout: per-qubit readout bit-flip probability
    """
    from qiskit_aer.noise import (
        NoiseModel, depolarizing_error, ReadoutError
    )
    nm = NoiseModel()

    err1 = depolarizing_error(p1q, 1)
    err2 = depolarizing_error(p2q, 2)
    nm.add_all_qubit_quantum_error(err1, ['h', 'rz', 'x', 'id'])
    nm.add_all_qubit_quantum_error(err2, ['cx', 'ecr', 'cz'])

    re = ReadoutError([[1 - p_readout, p_readout],
                       [p_readout, 1 - p_readout]])
    nm.add_all_qubit_readout_error(re)
    return nm


# ---------------------------------------------------------------------------
# Run circuits through Aer
# ---------------------------------------------------------------------------

def run_aer(circuits, shots, noise_model=None, label=""):
    """
    Submit a list of circuits to AerSimulator and return list of count dicts.
    Circuits are batched into one job for speed.
    """
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    kwargs = {}
    if noise_model is not None:
        kwargs['noise_model'] = noise_model

    sim        = AerSimulator(**kwargs)
    transpiled = transpile(circuits, sim, optimization_level=0)

    t0  = time.time()
    job = sim.run(transpiled, shots=shots)
    result = job.result()
    elapsed = time.time() - t0

    counts_list = [result.get_counts(i) for i in range(len(circuits))]
    if label:
        print(f"  [{label}] {len(circuits)} circuits in {elapsed:.1f}s "
              f"({elapsed/len(circuits)*1000:.1f} ms/circuit)")
    return counts_list


# ---------------------------------------------------------------------------
# Decode counts -> p_measured
# ---------------------------------------------------------------------------

def p_from_counts(counts, n_bits):
    """
    counts : dict of bitstring -> count  (e.g. {'001': 512, '110': 512})
    n_bits == 1 : P(bit == '0')
    n_bits > 1  : P(even parity = even number of '1' chars)
    """
    total = sum(counts.values())
    if n_bits == 1:
        good = counts.get('0', 0)
    else:
        good = sum(v for k, v in counts.items() if k.count('1') % 2 == 0)
    return float(good) / total, total


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def simulate(args):
    tag       = "noisy" if args.noisy else "ideal"
    shots     = args.shots
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 72)
    print(f"Branch C: Aer Simulation  [{tag.upper()}, {shots} shots]")
    print("=" * 72)
    print()

    # Build PUBs
    c1_pubs = [] if args.c2_only else build_c1_pubs(verbose=True)
    c2_pubs = [] if args.c1_only else build_c2_pubs(verbose=True)
    n_c1    = len(c1_pubs)
    n_c2    = len(c2_pubs)
    print(f"\nTotal: {n_c1 + n_c2} circuits  ({n_c1} C1 + {n_c2} C2)")

    noise_model = make_noise_model() if args.noisy else None
    if args.noisy:
        print("\nNoise model: depolarizing p1q=0.001, p2q=0.010, readout=0.010")
    else:
        print("\nNoise model: none (ideal)")

    # --- Simulate C1 -------------------------------------------------------
    results = []

    if c1_pubs:
        print(f"\nSimulating C1 ({n_c1} circuits)...")
        c1_circuits = [p[0] for p in c1_pubs]
        c1_metas    = [p[1] for p in c1_pubs]

        # Batch by N (qubit count) for cleaner progress reporting
        for N in range(1, 21):
            idxs = [i for i, m in enumerate(c1_metas) if m["N"] == N]
            if not idxs:
                continue
            batch    = [c1_circuits[i] for i in idxs]
            batch_m  = [c1_metas[i]    for i in idxs]
            counts_l = run_aer(batch, shots, noise_model,
                               label=f"C1 N={N:2d} ({len(batch)} circs)")
            for meta, counts in zip(batch_m, counts_l):
                p_val, n_shots = p_from_counts(counts, N)
                row = dict(meta)
                row["p_measured"] = p_val
                row["n_shots"]    = n_shots
                results.append(row)

    # --- Simulate C2 -------------------------------------------------------
    if c2_pubs:
        print(f"\nSimulating C2 ({n_c2} circuits)...")
        c2_circuits = [p[0] for p in c2_pubs]
        c2_metas    = [p[1] for p in c2_pubs]
        counts_l    = run_aer(c2_circuits, shots, noise_model,
                              label=f"C2 all ({n_c2} circs)")
        for meta, counts in zip(c2_metas, counts_l):
            p_val, n_shots = p_from_counts(counts, 1)
            row = dict(meta)
            row["p_measured"] = p_val
            row["n_shots"]    = n_shots
            results.append(row)

    # --- Analysis ----------------------------------------------------------
    c1_spectrum = []
    Z_data      = []

    if not args.c2_only:
        c1_spectrum = print_c1_summary(results)

    if not args.c1_only:
        Z_data = print_c2_summary(results)

    # --- Noise degradation report (noisy mode) -----------------------------
    if args.noisy and c1_spectrum:
        print("\n--- C1 Noise Degradation by N ---")
        print(f"  {'N':>3}  {'amp (ideal~1)':>14}  {'r_fit':>7}  {'status':>10}")
        for s in c1_spectrum:
            bar_len = int(s["amp"] * 20)
            bar = "#" * bar_len + "." * (20 - bar_len)
            print(f"  {s['N']:>3}  {s['amp']:>6.3f}  [{bar}]  "
                  f"{s['r_fit']:>7.4f}  {s['resolved']:>10}")
        print()
        resolved = [s for s in c1_spectrum if s["resolved"] == "YES"]
        print(f"  Resolved: {len(resolved)}/20  "
              f"(noise degrades GHZ fidelity for large N -- "
              f"confirmed useful range for hardware)")

    # --- Save outputs ------------------------------------------------------
    out = os.path.join(script_dir, f"sim_results_{tag}_{timestamp}.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {os.path.basename(out)}")

    if c1_spectrum:
        c1_out = os.path.join(script_dir, f"sim_c1_spectrum_{tag}_{timestamp}.json")
        with open(c1_out, 'w') as f:
            json.dump(c1_spectrum, f, indent=2)
        print(f"C1 spectrum  : {os.path.basename(c1_out)}")

    if Z_data:
        c2_out = os.path.join(script_dir, f"sim_c2_Zdata_{tag}_{timestamp}.json")
        with open(c2_out, 'w') as f:
            json.dump(Z_data, f, indent=2, default=str)
        print(f"C2 Z data    : {os.path.basename(c2_out)}")

    print("\nDone.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Local Aer simulation of Branch C (C1+C2)"
    )
    ap.add_argument("--ideal",   action="store_true",
                    help="Noiseless simulation (default if neither flag given)")
    ap.add_argument("--noisy",   action="store_true",
                    help="Depolarizing + readout noise model")
    ap.add_argument("--shots",   type=int, default=2048,
                    help="Shots per circuit (default 2048)")
    ap.add_argument("--c1-only", action="store_true")
    ap.add_argument("--c2-only", action="store_true")
    args = ap.parse_args()

    # Default to ideal if neither flag set
    if not args.noisy:
        args.ideal = True

    simulate(args)


if __name__ == "__main__":
    main()
