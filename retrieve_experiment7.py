"""
retrieve_experiment7.py
Usage:  python retrieve_experiment7.py <JOB_ID>
"""
import sys, json, base64, zlib, numpy as np, math

sys.path.insert(0, ".")
from experiment7_precision_floor import (
    TOKEN, INSTANCE, CHANNEL,
    CONDITIONS, GHZ_PRIMES, GHZ_ALPHAS,
    f_mertens, build_all_pubs, analyze_results, print_summary,
)

def main():
    if len(sys.argv) < 2:
        print("Usage: python retrieve_experiment7.py <JOB_ID>")
        sys.exit(1)

    job_id = sys.argv[1].strip()
    print(f"Retrieving job: {job_id}")

    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(channel=CHANNEL, instance=INSTANCE, token=TOKEN)
    job    = service.job(job_id)
    status = str(job.status())
    print(f"Status: {status}")

    if status not in ("JobStatus.DONE", "DONE", "done"):
        print("Job not yet complete.")
        sys.exit(0)

    print("Fetching results...")
    job_result = job.result()

    pubs  = build_all_pubs()
    metas = [p[1] for p in pubs]

    results = analyze_results(job_result, metas)
    print_summary(results)

    out = f"exp7_results_{job_id[:12]}.json"
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
