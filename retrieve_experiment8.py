"""
Retrieve and analyse results for Experiment 8: Full Mertens Spectrum.
Usage: python retrieve_experiment8.py <JOB_ID>

Can also parse local result files without IBM connection:
  python retrieve_experiment8.py <JOB_ID> --local
  (expects job-<JOB_ID>-result.json and job-<JOB_ID>-info.json in cwd)
"""

import sys
import os
import json
import zlib
import base64
import io
import math
import numpy as np

# -- Import experiment definitions -----------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from experiment8_full_spectrum import (
    CONDITIONS, GHZ_PRIMES, GHZ_ALPHAS, N_ALPHA_PTS,
    f_mertens, PRIMES_2_20, TOKEN, INSTANCE, CHANNEL,
    build_all_pubs, print_summary
)


def p_even_from_numpy(ba_val):
    """Decode IBM BitArray stored as zlib-compressed .npy file."""
    raw   = zlib.decompress(base64.b64decode(ba_val['array']['__value__']))
    arr   = np.load(io.BytesIO(raw))
    shots = arr[:, 0]
    n_even = np.sum((shots == 0) | (shots == 3))
    return n_even / len(shots), len(shots)


def analyze_local(result_path):
    """Parse a local job-result JSON without IBM credentials."""
    with open(result_path) as f:
        data = json.load(f)
    pubs = data['__value__']['pub_results']

    # Rebuild metadata in submission order
    pub_metas = []
    for label, n, amax in CONDITIONS:
        fn     = f_mertens(n)
        alphas = np.linspace(0, amax, N_ALPHA_PTS)
        for alpha in alphas:
            pub_metas.append({
                "block": "bell_spectrum", "label": label,
                "n": n, "f_n": fn, "alpha": float(alpha), "alpha_max": amax,
                "is_prime": n in PRIMES_2_20,
                "theory_p_even": 0.5 * (1 + math.cos(2 * math.pi * fn * alpha)),
            })
    phi_sum = sum(f_mertens(p) for p in GHZ_PRIMES)
    for alpha in GHZ_ALPHAS:
        phi = 2 * math.pi * phi_sum * alpha
        pub_metas.append({
            "block": "ghz_ref", "alpha": float(alpha), "phi": phi,
            "theory_p_even": math.cos(phi / 2) ** 2,
        })

    results = []
    for idx, meta in enumerate(pub_metas):
        ba = pubs[idx]['__value__']['data']['__value__']['fields']['c']['__value__']
        p_even, n_shots = p_even_from_numpy(ba)
        row = dict(meta)
        row['p_even']  = p_even
        row['n_shots'] = n_shots
        if 'theory_p_even' in meta:
            row['residual'] = p_even - meta['theory_p_even']
        results.append(row)
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id")
    ap.add_argument("--local", action="store_true",
                    help="Parse local result files instead of fetching from IBM")
    args = ap.parse_args()

    job_id = args.job_id
    short  = job_id[:12]

    # Try local files first (or if --local flag)
    result_candidates = [
        os.path.join(script_dir, f"job-{job_id}-result.json"),
        os.path.join(script_dir, f"job-{job_id}-result-*.json"),
    ]

    local_file = None
    for pat in result_candidates:
        import glob
        matches = glob.glob(pat)
        if matches:
            local_file = matches[0]
            break

    if local_file and (args.local or not False):
        print(f"Parsing local result file: {os.path.basename(local_file)}")
        results = analyze_local(local_file)
    else:
        print(f"Retrieving job: {job_id}")
        from qiskit_ibm_runtime import QiskitRuntimeService
        from experiment8_full_spectrum import analyze_results, build_all_pubs
        service = QiskitRuntimeService(channel=CHANNEL, instance=INSTANCE, token=TOKEN)
        job     = service.job(job_id)
        job_result = job.result()
        pubs   = build_all_pubs()
        metas  = [p[1] for p in pubs]
        results = analyze_results(job_result, metas)

    spectrum = print_summary(results)

    out_path = os.path.join(script_dir, f"exp8_results_{short}.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {os.path.basename(out_path)}")

    if spectrum:
        spec_path = os.path.join(script_dir, f"exp8_spectrum_{short}.json")
        with open(spec_path, 'w') as f:
            json.dump(spectrum, f, indent=2)
        print(f"Spectrum: {os.path.basename(spec_path)}")


if __name__ == "__main__":
    main()
