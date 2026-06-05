# Quantum Arithmetic Spectroscopy

**Earl Decker** — Independent Researcher  
earldecker@gmail.com

Code repository for:

> **QAS I:** *Quantum Arithmetic Spectroscopy I: Near-Degeneracies, Prime Structure, and Riemann Zero Detection* (2026)
>
> **QAS II:** *Quantum Arithmetic Spectroscopy II: Mertens-Ramsey Circuits, GUE Statistics, and IBM Circuit Design* (2026)

---

## Overview

This repository contains all Python scripts used to generate the results in both papers. The work investigates the spectral properties of the logarithmic weight function f(n) = ln(n)/n and the Mertens function as tools for detecting structure related to the Riemann Hypothesis. Experiments progress from classical number-theoretic analysis (Phases 1–7) through quantum circuit design (C1–C5).

All quantum experiments use **local Aer simulation only** — no live IBM hardware access is required to reproduce results.

---

## Repository Structure

### Classical Spectral Experiments (QAS I)

| Script | Phase | Description |
|--------|-------|-------------|
| `experiment1_corrected.py` | 1 | Spectral weight f(n) = ln(n)/n: basic near-degeneracy search |
| `experiment1_redesigned.py` | 1 | Redesigned Phase 1 with improved numerical precision |
| `experiment2_ibm_corrected.py` | 2 | Extended near-degeneracy analysis, corrected IBM-style circuits |
| `experiment2_ibm_updated.py` | 2 | Updated Phase 2 with additional degeneracy cases |
| `experiment3_5qubit_ghz.py` | 3 | 5-qubit GHZ state entanglement experiments (Aer) |
| `experiment4_mertens_map.py` | 4 | Mertens function spectral mapping |
| `experiment5_prime_beats.py` | 5 | Prime beat frequency analysis |
| `experiment6_spectral_isolation.py` | 6 | Spectral isolation of individual primes |
| `experiment7_precision_floor.py` | 7 | Precision floor analysis |
| `experiment8_full_spectrum.py` | 8 | Full spectrum computation |
| `near_degeneracy_search.py` | — | Systematic search for near-degeneracies in f(n) |
| `spectral_gap_analysis.py` | — | Spectral gap and level spacing statistics |
| `spectral_gap_zero_proxy.py` | — | Spectral gap as a proxy for Riemann zeros |
| `zero_proxy_extended.py` | — | Extended zero proxy analysis |
| `analyze_experiment5.py` | — | Analysis utilities for Phase 5 data |

### Quantum Circuit Simulations (QAS I — Appendix)

| Script | Circuit | Description |
|--------|---------|-------------|
| `simulate_c1c2_aer.py` | C1, C2 | GHZ Ramsey circuits for zero frequency estimation (Aer) |
| `simulate_c3_aer.py` | C3 | Phase estimation circuit, prime-power counting (Aer) |
| `simulate_c4_aer.py` | C4 | Extended C3 with error mitigation (Aer) |
| `experiment_c1c2_riemann_branch.py` | C1/C2 | Riemann branch variant |

### Phase 5–7 Spectroscopy (QAS II)

| Script | Phase | Description |
|--------|-------|-------------|
| `mertens_phase5_spectroscopy.py` | 5 | Mertens-based Riemann zero detection; all 30 zeros identified with Z-scores up to 476σ |
| `gue_pair_correlation.py` | 6 | GUE nearest-neighbor spacing statistics; KS p=0.74 vs GUE prediction |
| `explicit_formula_phase7.py` | 7 | ψ(x) reconstruction from first 100 zeros; f(2)=f(4) anomaly identified |

### C5 Quantum Ramsey Spectroscopy (QAS II)

| Script | Description |
|--------|-------------|
| `simulate_c5_ramsey_spectroscopy.py` | Main C5 simulation: k-qubit GHZ matched filter for Riemann zero detection. Demonstrates exact Heisenberg scaling (k=3: 3.00×, k=5: 5.00×). Generates IBM-ready circuit batch. |

**C5 output files:**
- `c5_ramsey_spectroscopy.json` — full numerical results
- `c5_ramsey_spectroscopy.png` — 4-panel figure (included in QAS II)
- `c5_ibm_circuits_k5.qpy` — serialized IBM-ready circuits (1,000 circuits, k=5)

---

## Installation

```bash
# Core dependencies
pip install numpy scipy matplotlib qiskit qiskit-aer

# For manuscript tools (optional)
pip install python-docx
```

Python 3.10+ recommended.

---

## Reproducing Key Results

### Phase 5: Riemann Zero Detection (QAS II, Section 2.1)
```bash
python mertens_phase5_spectroscopy.py
```
Output: Z-scores for first 30 Riemann zeros in the Mertens power spectrum. Expects max Z-score ≈ 476 at γ₁ = 14.135.

### Phase 6: GUE Statistics (QAS II, Section 2.2)
```bash
python gue_pair_correlation.py
```
Output: KS test p-value vs GUE nearest-neighbor spacing distribution. Expects p ≈ 0.74.

### Phase 7: Explicit Formula (QAS II, Section 2.3)
```bash
python explicit_formula_phase7.py
```
Output: ψ(x) reconstruction accuracy, f(2)=f(4) near-degeneracy anomaly.

### C5 Quantum Ramsey Spectroscopy (QAS II, Section 2.4)
```bash
python simulate_c5_ramsey_spectroscopy.py
```
Output: `c5_ramsey_spectroscopy.json`, `c5_ramsey_spectroscopy.png`, `c5_ibm_circuits_k5.qpy`.  
Runtime: ~2–5 minutes (Aer simulation, no IBM account needed).

Expected Heisenberg scaling results:
| k | HWHM (measured) | Gain vs k=1 |
|---|-----------------|-------------|
| 1 | 0.956           | 1.00×       |
| 3 | 0.319           | 3.00×       |
| 5 | 0.191           | 5.00×       |

---

## IBM Hardware Note

`c5_ibm_circuits_k5.qpy` contains a batch of 1,000 serialized circuits (5 qubits, depth 12, 200 shots each ≈ 200,000 total shots, estimated ~5 min on current IBM hardware). To run on IBM hardware, load with `qiskit.qpy.load()` and submit via the IBM Quantum Platform. **All validation in the papers uses local Aer simulation.**

---

## Citation

If you use this code, please cite both papers:

```
Decker, E. (2026). Quantum Arithmetic Spectroscopy I:
Near-Degeneracies, Prime Structure, and Riemann Zero Detection.
Independent Researcher. DOI: 10.5281/zenodo.20558339

Decker, E. (2026). Quantum Arithmetic Spectroscopy II:
Mertens-Ramsey Circuits, GUE Statistics, and IBM Circuit Design.
Independent Researcher. DOI: 10.5281/zenodo.20558339

Code: https://github.com/earldecker-blip/quantum-arithmetic-spectroscopy
DOI: 10.5281/zenodo.20558338  (concept DOI — always resolves to latest version)
```

---

## License

MIT License — see `LICENSE` for details.
