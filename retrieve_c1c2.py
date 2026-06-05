"""
retrieve_c1c2.py -- retrieve and analyse Branch C (C1+C2) results

Usage:
  python retrieve_c1c2.py <JOB_ID>              # fetch from IBM
  python retrieve_c1c2.py <JOB_ID> --local      # parse local job-*-result.json
  python retrieve_c1c2.py <JOB_ID> --c1-only    # skip C2 analysis
  python retrieve_c1c2.py <JOB_ID> --c2-only    # skip C1 analysis

BitArray decode (IBM SamplerV2 result format):
  Each PUB result stores a zlib-compressed .npy array of shape (n_shots, 1).
  Each shot is an integer:
    1-qubit (C2):  0 = |0>, 1 = |1>    =>  P(0) = count(0)/total
    N-qubit (C1):  integer in [0,2^N-1] =>  P(even parity) = count(even popcount)/total

Outputs:
  c1c2_results_<short>.json  -- full per-PUB results
  c1_spectrum_<short>.json   -- C1 fitted F_N spectrum
  c2_Zdata_<short>.json      -- C2 Z_M(t) reconstruction + sign-change table
"""

import sys
import os
import json
import zlib
import base64
import io
import math
import glob
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from experiment_c1c2_riemann_branch import (
    C1_N_ALPHA, C2_N_T, C2_T_MIN, C2_T_MAX,
    PRIMES_20, RIEMANN_ZEROS,
    f_mertens, F_partial, riemann_siegel_theta, M_terms,
    print_c1_summary, print_c2_summary,
    TOKEN, INSTANCE, CHANNEL,
)


# ---------------------------------------------------------------------------
# Meta reconstruction (silent -- no circuit building)
# ---------------------------------------------------------------------------

def build_meta_list(c1_only=False, c2_only=False):
    """
    Reconstruct PUB metadata in the exact same order as build_all_pubs().
    Does NOT build circuits, so produces no output.
    Returns (all_metas, n_c1, n_c2).
    """
    metas = []

    # C1 metas
    if not c2_only:
        for N in range(1, 21):
            F_N    = F_partial(N)
            amax   = round(2.0 / F_N, 6)
            alphas = np.linspace(0.0, amax, C1_N_ALPHA)
            for alpha in alphas:
                metas.append({
                    "block":         "c1_accumulator",
                    "N":             N,
                    "primes":        PRIMES_20[:N],
                    "F_N":           F_N,
                    "alpha":         float(alpha),
                    "alpha_max":     amax,
                    "theory_p_even": math.cos(math.pi * F_N * float(alpha)) ** 2,
                })
    n_c1 = len(metas)

    # C2 metas
    if not c1_only:
        t_vals = np.linspace(C2_T_MIN, C2_T_MAX, C2_N_T)
        for t in t_vals:
            M     = M_terms(float(t))
            theta = riemann_siegel_theta(float(t))
            for n in range(1, M + 1):
                phase = theta - float(t) * math.log(n)
                metas.append({
                    "block":     "c2_riemann",
                    "t":         float(t),
                    "n":         n,
                    "M":         M,
                    "theta":     theta,
                    "phase":     phase,
                    "theory_p0": 0.5 * (1.0 + math.cos(phase)),
                })
    n_c2 = len(metas) - n_c1

    return metas, n_c1, n_c2


# ---------------------------------------------------------------------------
# BitArray decode
# ---------------------------------------------------------------------------

def decode_bitarray(ba_val, n_bits):
    """
    Decode IBM BitArray stored as zlib-compressed .npy.
    n_bits == 1 : returns P(shot == 0), n_shots
    n_bits > 1  : returns P(even parity of n_bits bits), n_shots
    """
    raw   = zlib.decompress(base64.b64decode(ba_val['array']['__value__']))
    arr   = np.load(io.BytesIO(raw))
    shots = arr[:, 0].astype(np.int64)
    total = len(shots)

    if n_bits == 1:
        good = int(np.sum(shots == 0))
    else:
        # even parity = even number of set bits
        good = int(np.sum(
            np.array([bin(int(s)).count('1') % 2 for s in shots]) == 0
        ))

    return float(good) / total, total


# ---------------------------------------------------------------------------
# Parse local result file
# ---------------------------------------------------------------------------

def analyze_local(result_path, all_metas):
    """
    Parse a local job-<ID>-result.json without IBM credentials.
    Decodes each PUB's BitArray according to its circuit width.
    """
    with open(result_path) as f:
        data = json.load(f)

    pub_results = data['__value__']['pub_results']
    results     = []

    for idx, meta in enumerate(all_metas):
        n_bits = 1 if meta["block"] == "c2_riemann" else meta["N"]
        ba_val = (pub_results[idx]
                  ['__value__']['data']['__value__']['fields']['c']['__value__'])
        p_val, n_shots = decode_bitarray(ba_val, n_bits)

        row = dict(meta)
        row["p_measured"] = p_val
        row["n_shots"]    = n_shots
        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Lomb-Scargle residual analysis for C1
# ---------------------------------------------------------------------------

def lomb_scargle_c1(c1_spectrum):
    """
    Optional: look for any periodic structure in the C1 F_N residuals
    (delta_F = F_fit - F_theory as a function of N).
    Uses the Lomb-Scargle periodogram -- valid even if N-spacing is irregular.
    """
    try:
        from astropy.timeseries import LombScargle
    except ImportError:
        print("  (astropy not installed -- skipping Lomb-Scargle analysis)")
        return

    N_vals    = np.array([s["N"] for s in c1_spectrum])
    delta_F   = np.array([s["F_fit"] - s["F_theory"] for s in c1_spectrum])

    if len(delta_F) < 5:
        return

    # Frequency grid: periods from 2 to 20 in units of prime-index steps
    freq_min = 1.0 / 20.0
    freq_max = 1.0 / 2.0
    frequency = np.linspace(freq_min, freq_max, 200)

    ls        = LombScargle(N_vals, delta_F)
    power     = ls.power(frequency)
    peak_freq = frequency[np.argmax(power)]
    peak_pow  = power.max()

    print(f"\n  Lomb-Scargle on C1 residuals (delta_F vs N):")
    print(f"    Peak frequency: {peak_freq:.4f} (period ~ {1/peak_freq:.1f} prime indices)")
    print(f"    Peak power    : {peak_pow:.4f}  (0=noise, 1=perfect sine)")
    print(f"    Interpretation: {'possible periodicity' if peak_pow > 0.5 else 'no strong periodicity'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Retrieve and analyse Branch C results")
    ap.add_argument("job_id",   help="IBM job ID")
    ap.add_argument("--local",   action="store_true",
                    help="Force parsing local JSON (skip IBM connection attempt)")
    ap.add_argument("--c1-only", action="store_true", help="Analyse only C1 results")
    ap.add_argument("--c2-only", action="store_true", help="Analyse only C2 results")
    ap.add_argument("--lomb",    action="store_true",
                    help="Run Lomb-Scargle periodogram on C1 residuals (requires astropy)")
    args = ap.parse_args()

    job_id = args.job_id
    short  = job_id[:12]

    print(f"Branch C retrieve: job {job_id}")

    # Build meta list without circuit construction
    all_metas, n_c1, n_c2 = build_meta_list(
        c1_only=args.c1_only, c2_only=args.c2_only
    )
    print(f"Expecting {len(all_metas)} PUBs  ({n_c1} C1 + {n_c2} C2)")

    # Locate local result file
    patterns = [
        os.path.join(script_dir, f"job-{job_id}-result.json"),
        os.path.join(script_dir, f"job-{job_id}-result-*.json"),
    ]
    local_file = None
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            local_file = matches[0]
            break

    # Fetch or parse
    if local_file:
        print(f"Parsing local file: {os.path.basename(local_file)}")
        results = analyze_local(local_file, all_metas)
    elif not args.local:
        print("Local file not found -- fetching from IBM...")
        from qiskit_ibm_runtime import QiskitRuntimeService
        from experiment_c1c2_riemann_branch import analyze_results
        service    = QiskitRuntimeService(channel=CHANNEL, instance=INSTANCE, token=TOKEN)
        job        = service.job(job_id)
        job_result = job.result()
        results    = analyze_results(job_result, all_metas)
    else:
        sys.exit(f"ERROR: --local flag set but no local file found for job {job_id}")

    # ---- C1 analysis -------------------------------------------------------
    c1_spectrum = []
    if not args.c2_only:
        c1_spectrum = print_c1_summary(results)
        if args.lomb and c1_spectrum:
            lomb_scargle_c1(c1_spectrum)

    # ---- C2 analysis -------------------------------------------------------
    Z_data = []
    if not args.c1_only:
        Z_data = print_c2_summary(results)

    # ---- Save outputs ------------------------------------------------------
    out = os.path.join(script_dir, f"c1c2_results_{short}.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved : {os.path.basename(out)}")

    if c1_spectrum:
        c1_out = os.path.join(script_dir, f"c1_spectrum_{short}.json")
        with open(c1_out, 'w') as f:
            json.dump(c1_spectrum, f, indent=2)
        print(f"C1 spectrum saved  : {os.path.basename(c1_out)}")

    if Z_data:
        c2_out = os.path.join(script_dir, f"c2_Zdata_{short}.json")
        with open(c2_out, 'w') as f:
            json.dump(Z_data, f, indent=2, default=str)
        print(f"C2 Z data saved    : {os.path.basename(c2_out)}")


if __name__ == "__main__":
    main()
