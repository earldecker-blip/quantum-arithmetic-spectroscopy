from qiskit_ibm_runtime import QiskitRuntimeService
import numpy as np
import matplotlib.pyplot as plt

# ====================== CONFIG ======================
JOB_ID = "d8fkb63alsvc7391spug"
# ====================================================

print("Connecting to IBM Quantum...")
service = QiskitRuntimeService(channel="ibm_quantum_platform")

print(f"Retrieving job {JOB_ID} ...")
job = service.job(JOB_ID)
result = job.result()

print(f"Job completed. Number of PUBs: {len(result.pub_results)}\n")

p_even_list = []
num_bits_list = []

for i, pub_result in enumerate(result.pub_results):
    try:
        # Get the classical register data
        c_data = pub_result.data.c
        
        # Get counts
        counts = c_data.get_counts()
        
        total = sum(counts.values())
        even = sum(v for k, v in counts.items() if k.count('1') % 2 == 0)
        p_even = even / total if total > 0 else 0.0
        
        # num_bits from the BitArray
        n_bits = c_data.num_bits
        
        p_even_list.append(p_even)
        num_bits_list.append(n_bits)
        
        if i < 10 or i % 10 == 0:
            print(f"PUB {i:02d}  (n_bits={n_bits})  P(even) = {p_even:.4f}")
            
    except Exception as e:
        print(f"Error on PUB {i}: {e}")
        p_even_list.append(0.0)
        num_bits_list.append(0)

# Summary
print(f"\n=== Summary ===")
print(f"Total PUBs: {len(p_even_list)}")

p_even_3bit = [p for p, n in zip(p_even_list, num_bits_list) if n == 3]
p_even_2bit = [p for p, n in zip(p_even_list, num_bits_list) if n == 2]

if p_even_3bit:
    print(f"3-bit circuits: {len(p_even_3bit)}  |  Mean P(even) = {np.mean(p_even_3bit):.4f}")
if p_even_2bit:
    print(f"2-bit circuits: {len(p_even_2bit)}  |  Mean P(even) = {np.mean(p_even_2bit):.4f}")

# Plot
plt.figure(figsize=(12, 4))
plt.plot(p_even_list, 'o-', markersize=3, alpha=0.7)
plt.axhline(0.5, color='gray', linestyle='--', label='Random (0.5)')
plt.title("Experiment 5 – P(even) across 80 PUBs (Job d8fkb63alsvc7391spug)")
plt.xlabel("PUB index")
plt.ylabel("P(even parity)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Experiment5_Peven.png", dpi=150)
plt.show()

print("\nPlot saved as Experiment5_Peven.png")