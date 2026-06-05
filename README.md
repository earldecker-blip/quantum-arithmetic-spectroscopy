# Quantum Arithmetic Spectroscopy

**Earl Decker** — Independent Researcher  
earldecker@gmail.com

Code repository for:

> **Paper 1:** *Quantum Arithmetic Spectroscopy I: Near-Degeneracies, Prime Structure, and Riemann Zero Detection* (2026)
>
> **Paper 2:** *Quantum Arithmetic Spectroscopy II: Mertens-Ramsey Circuits, GUE Statistics, and IBM Circuit Design* (2026)

---

## Overview

This repository contains all Python scripts used to generate the results in both papers. The work investigates the spectral properties of f(n) = ln(n)/n and the Mertens function as tools for detecting structure related to the Riemann Hypothesis. Experiments progress from classical number-theoretic analysis (Phases 1–7) through quantum circuit design (C1–C5).

All quantum experiments use **local Aer simulation only** — no IBM hardware account required.

---

## Scripts

### Paper 1 — Classical Analysis

| Script | Description |
|--------|-------------|
| `experiment1_corrected.py` | Spectral weight f(n): near-degeneracy search |
| `experiment3_5qubit_ghz.py` | 5-qubit GHZ entanglement experiments (Aer) |
| `spectral_gap_analysis.py` | Spectral gap and level spacing statistics |
| `spectral_gap_zero_proxy.py` | Spectral gap as Riemann zero proxy |
| `zero_proxy_extended.py` | Extended zero proxy (p up to 10^9) |
| `simulate_c1c2_aer.py` | C1/C2: GHZ Ramsey zero frequency estimation |
| `simulate_c3_aer.py` | C3: Phase estimation, prime-power counting |
| `simulate_c4_aer.py` | C4: Extended C3 with error mitigation |

### Paper 2 — Quantum Spectroscopy

| Script | Phase | Result |
|--------|-------|--------|
| `mertens_phase5_spectroscopy.py` | 5 | All 30 Riemann zeros detected, Z up to 476σ |
| `gue_pair_correlation.py` | 6 | GUE KS p=0.74, level repulsion confirmed |
| `explicit_formula_phase7.py` | 7 | ψ(x) reconstructed to sub-percent accuracy |
| `simulate_c5_ramsey_spectroscopy.py` | C5 | Heisenberg scaling 5.00× at k=5 |

---

## Installation

```bash
pip install numpy scipy matplotlib qiskit qiskit-aer
```

Python 3.10+ recommended.

---

## Citation

```
Decker, E. (2026). Quantum Arithmetic Spectroscopy I. DOI: [Zenodo DOI]
Decker, E. (2026). Quantum Arithmetic Spectroscopy II. DOI: [Zenodo DOI]
Code: https://github.com/earldecker-blip/quantum-arithmetic-spectroscopy
```

---

## License

MIT — see `LICENSE` for details.
