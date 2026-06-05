"""
Retrieve Experiment 3 results from completed IBM job.
Job ID: d8f4bbpvjngc73apa6jg  (ibm_marrakesh, Jun 1 2026)
Uses exact instance string from IBM dashboard.
"""
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService

IBM_TOKEN = "96axnVJAp_PkXhi7mpX8t_CVj1NtzqmHjaApLQ5Pn96Q"
JOB_ID    = "d8f4bbpvjngc73apa6jg"
N_QUBITS  = 6

service = QiskitRuntimeService(
    channel='ibm_quantum_platform',
    token=IBM_TOKEN,
    instance='crn:v1:bluemix:public:quantum-computing:us-east:a/8420df4c778d45e59489b345c26d2c81:73caf2a1-d677-4d15-b711-f8f3fc73732a::'
)

print(f"Retrieving job {JOB_ID} ...")
job = service.job(JOB_ID)
print(f"Status: {job.status()}")

result = job.result()
n_pubs = len(result)
print(f"Number of PUBs: {n_pubs}")

# Auto-detect classical register name
sample_data = result[0].data
reg_name = list(vars(sample_data).keys())[0]
print(f"Classical register name: '{reg_name}'\n")

def get_counts(pub_result):
    return getattr(pub_result.data, reg_name).get_counts()

def parity_even(bitstring):
    return bitstring.count('1') % 2 == 0

def extract(pub_result):
    counts = get_counts(pub_result)
    total  = sum(counts.values())
    even   = sum(v for k, v in counts.items() if parity_even(k))
    return even / total

print("=== ALL PUB PARITIES ===")
all_coh = []
for i in range(n_pubs):
    c = extract(result[i])
    all_coh.append(c)
    print(f"  PUB {i:2d}: P(even) = {c:.4f}")

# ── PUB layout (from experiment3_5qubit_ghz.py) ──────────────────────────────
# PUBs  0.. 2: norm_prime  at delays [0, 50, 125] ns
# PUBs  3.. 7: norm_random seed 0 at delays [0,50,125] — BUT script used 5 seeds × 3 delays
# Actually layout:
#   norm_prime:       PUBs 0,1,2        (delays 0,50,125)
#   norm_random ×5:   PUBs 3..17        (seeds 0-4, each 3 delays → 15 PUBs)
#   raw_prime:        PUBs 18,19,20     (delays 0,50,125)
#   raw_rand_matched: PUBs 21,22,23     (delays 0,50,125)
# Total = 3 + 15 + 3 + 3 = 24  ✓

delays = [0, 50, 125]

if n_pubs >= 24:
    norm_prime = [all_coh[i]          for i in range(3)]
    norm_rand  = [[all_coh[3 + s*3 + d] for d in range(3)] for s in range(5)]
    raw_prime  = [all_coh[18 + i]     for i in range(3)]
    raw_rand   = [all_coh[21 + i]     for i in range(3)]

    nr_mean = [np.mean([norm_rand[s][d] for s in range(5)]) for d in range(3)]
    nr_std  = [np.std( [norm_rand[s][d] for s in range(5)]) for d in range(3)]

    print("\n=== NORMALIZED TEST (prime structure, no Mertens advantage) ===")
    print(f"{'Delay':>8} {'norm_prime':>12} {'rand_mean':>12} {'rand_std':>10} {'revival':>10} {'sigma':>8}")
    for d, delay in enumerate(delays):
        rev = norm_prime[d] - nr_mean[d]
        sig = rev / nr_std[d] if nr_std[d] > 1e-9 else float('nan')
        print(f"{delay:>7}ns {norm_prime[d]:>12.4f} {nr_mean[d]:>12.4f} {nr_std[d]:>10.4f} {rev:>+10.4f} {sig:>+8.2f}s")

    print("\n=== RAW TEST (unaligned prime phases, ideal P(even)=0.431) ===")
    print(f"{'Delay':>8} {'raw_prime':>12} {'raw_rand':>12} {'revival':>10}")
    for d, delay in enumerate(delays):
        rev = raw_prime[d] - raw_rand[d]
        print(f"{delay:>7}ns {raw_prime[d]:>12.4f} {raw_rand[d]:>12.4f} {rev:>+10.4f}")

    rev0 = norm_prime[0] - nr_mean[0]
    sig0 = rev0 / nr_std[0] if nr_std[0] > 1e-9 else 0.0

    print("\n=== VERDICT ===")
    if rev0 > 0 and sig0 > 2:
        verdict = "PRIME STRUCTURE IS PROTECTIVE (independent of Mertens alignment)"
    elif rev0 > 0 and sig0 > 0:
        verdict = "WEAK PRIME ADVANTAGE (inconclusive)"
    else:
        verdict = "NO PRIME STRUCTURAL ADVANTAGE (Mertens was the whole story in Exp 2)"

    print(f"norm_revival at delay=0: {rev0:+.4f}  ({sig0:+.2f} sigma)")
    print(f"Verdict: {verdict}")

else:
    print(f"\nUnexpected number of PUBs: {n_pubs} (expected 24)")
    print("Raw parities above — manual analysis needed.")
