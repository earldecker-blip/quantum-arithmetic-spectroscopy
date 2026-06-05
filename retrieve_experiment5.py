"""
retrieve_experiment5.py
========================
Retrieve and analyze completed Experiment 5 IBM Quantum job.

Usage:
    python retrieve_experiment5.py <JOB_ID>

Example:
    python retrieve_experiment5.py d8fesk3o3njc73f06090
"""

import sys
import json
import numpy as np
from scipy.stats import pearsonr

# ── Import experiment5 helpers ────────────────────────────────────────────────
# Pull in the condition list, analysis functions, and credentials from exp5
sys.path.insert(0, ".")
from experiment5_prime_beats import (
    TOKEN, INSTANCE, CHANNEL,
    CONDITIONS, FREQ_MODEL, MODEL_NAME,
    analyze_results, print_summary,
)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python retrieve_experiment5.py <JOB_ID>")
        sys.exit(1)

    job_id = sys.argv[1].strip()
    print(f"Retrieving job: {job_id}")

    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(
        channel=CHANNEL,
        instance=INSTANCE,
        token=TOKEN,
    )

    job = service.job(job_id)
    status = job.status()
    print(f"Job status: {status}")

    if str(status) not in ("JobStatus.DONE", "DONE", "done"):
        print("Job not yet complete. Try again later.")
        sys.exit(0)

    print("Fetching results...")
    job_result = job.result()

    # Rebuild metadata from experiment5 (same parameter order as submission)
    from experiment5_prime_beats import build_all_pubs
    pubs = build_all_pubs(backend_dt=None)   # metadata only, no hardware needed
    metadata = [p[3] for p in pubs]

    # Analyze
    results = analyze_results(job_result, metadata)

    # Print summary
    print_summary(results)

    # Save raw results to JSON
    out_file = f"exp5_results_{job_id[:12]}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results saved to: {out_file}")


if __name__ == "__main__":
    main()
