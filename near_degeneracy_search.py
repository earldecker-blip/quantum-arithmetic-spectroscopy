"""
near_degeneracy_search.py -- Mertens Near-Degeneracy Search
============================================================
Systematic search for pairs (a,b) where |f(a)-f(b)| < epsilon in
f(n) = ln(n)/n (Mertens weight function).

MAIN RESULTS
------------
1. EXACT DEGENERACY — f(2) = f(4) is the UNIQUE exact integer solution.
   Proof: f(a) = f(b) requires a^b = b^a. The only integer solutions are
   (a,b) = (2,4) and (4,2). Verified computationally for b <= 10,000.

2. EXACT ADDITIVE RESONANCES — derived from prime-power structure.
   For any prime p:  f(p^j) = j*ln(p)/p^j
   Therefore:        f(p^j) + f(p^k) = f(p^m)  iff  j/p^j + k/p^k = m/p^m
   Known exact cases (verified):
     f(9) + f(27) = f(3)           [p=3: 2/9 + 1/9 = 3/9 = 1/3]
     f(16) + f(16) = f(2) = f(4)   [p=2: 1/4 + 1/4 = 1/2]
     f(32) + f(64) = f(16)         [p=2: 5/32 + 6/64 = 8/32 = 1/4]
     f(27) + f(27) = f(9)          [p=3: 2*1/9 = 2/9]

3. MIRROR STRUCTURE — every integer a >= 3 has a unique real mirror
   x_mirror in (1, e) satisfying f(x_mirror) = f(a).
   Only a=4 has an INTEGER mirror (x_mirror = 2).

4. ACCIDENTAL SUM RESONANCES — f(a)+f(b) ~= f(c) with no shared factors.
   Best case: f(270)+f(497) ~= f(151), gap = 1.02e-8.
   These require beat periods of ~10^8 in alpha-space — beyond current hardware.

5. QUANTUM EXPERIMENT DESIGN:
   - Exact resonances: zero-beat circuit, P(even) = const
   - Cross-branch (2,5): beat period alpha=40.5, measurable in Exp 8 window
   - Prime-power pairs: rational beat ratios (e.g., f(9)/f(27) = 2 exactly)

OUTPUTS
-------
  near_degeneracy.png         -- Figure: landscape + harmonic series + beat curves
  near_degeneracy_report.txt  -- Full table of near-degeneracies
"""

import os, math, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

script_dir = os.path.dirname(os.path.abspath(__file__))

N_EXACT   = 10000  # search range for exact degeneracy uniqueness check
N_SEARCH  = 500    # search range for sum resonances
N_PLOT    = 60     # range for landscape plot

# ── Helpers ──────────────────────────────────────────────────────────────────

def f(n):
    return math.log(n) / n if n >= 2 else 0.0

def is_prime(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(n**0.5)+1, 2):
        if n % i == 0: return False
    return True

def prime_factors(n):
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1: factors.add(n)
    return factors

def find_mirror(a):
    """Find x in (1,e) where f(x) = f(a). Only exists for a >= 3."""
    if a == 2: return 2.0
    fa = f(a)
    x = 1.5
    for _ in range(300):
        if x <= 1.001 or x >= 2.717: x = 1.5
        lx = math.log(x)
        fx = lx / x
        dfx = (1 - lx) / x**2
        if abs(dfx) < 1e-14: break
        x -= 0.5 * (fx - fa) / dfx
        x = max(1.001, min(2.710, x))
        if abs(math.log(x)/x - fa) < 1e-13: break
    return x

# ── 1. Exact degeneracy uniqueness ───────────────────────────────────────────

print("Checking exact degeneracy uniqueness for b <= {:,} ...".format(N_EXACT))
exact_deg = []
for a in range(2, 200):
    for b in range(a+1, N_EXACT+1):
        if abs(f(a) - f(b)) < 1e-15:
            exact_deg.append((a, b))
print(f"  Exact degeneracies: {exact_deg}")
print()

# ── 2. Exact additive resonances from prime powers ───────────────────────────

print("Exact additive resonances f(p^j) + f(p^k) = f(p^m):")
fv = {n: f(n) for n in range(2, N_SEARCH+1)}
exact_add = []
for p in [2, 3, 5, 7, 11, 13]:
    powers = [p**k for k in range(1, 9) if p**k <= N_SEARCH]
    for i, a in enumerate(powers):
        for b in powers[i:]:
            s = fv[a] + fv[b]
            for c in powers:
                if abs(fv[c] - s) < 1e-13:
                    ja = round(math.log(a)/math.log(p))
                    jb = round(math.log(b)/math.log(p))
                    jc = round(math.log(c)/math.log(p))
                    print(f"  f({p}^{ja}={a:>4}) + f({p}^{jb}={b:>4}) = f({p}^{jc}={c:>4})  [EXACT]")
                    exact_add.append((a, b, c, p))
print()

# ── 3. Accidental sum resonances ─────────────────────────────────────────────

print(f"Searching accidental sum resonances (no shared prime factors) n<={N_SEARCH} ...")
flist = [fv[n] for n in range(2, N_SEARCH+1)]
flist_arr = np.array(flist)

accidental = []
for a in range(2, N_SEARCH+1):
    for b in range(a, N_SEARCH+1):
        s = fv[a] + fv[b]
        if s > fv[2] + 1e-9: continue
        # Binary search for nearest c
        lo, hi = 0, N_SEARCH-2
        while lo < hi:
            mid = (lo + hi) // 2
            if flist[mid] >= s: lo = mid+1
            else: hi = mid
        for ci in [lo-1, lo, lo+1]:
            c = ci + 2
            if 2 <= c <= N_SEARCH:
                gap = abs(fv[c] - s)
                if gap < 1e-5:
                    pf_a = prime_factors(a)
                    pf_b = prime_factors(b)
                    pf_c = prime_factors(c)
                    if not (pf_a & pf_b) and not (pf_a & pf_c) and not (pf_b & pf_c):
                        accidental.append({'a':a,'b':b,'c':c,'fa':fv[a],'fb':fv[b],
                                           'fc':fv[c],'gap':gap})
accidental.sort(key=lambda x: x['gap'])
seen = set()
unique_acc = []
for h in accidental:
    key = (h['a'],h['b'],h['c'])
    if key not in seen:
        seen.add(key)
        unique_acc.append(h)

print(f"  Found {len(unique_acc)} accidental resonances with gap < 1e-5.")
print(f"  Top 10:")
print(f"    {'a':>5} {'b':>5} {'c':>5}  {'gap':>12}  {'beat_period':>14}")
print("    " + "-"*55)
for h in unique_acc[:10]:
    period = 1.0/h['gap'] if h['gap'] > 0 else float('inf')
    print(f"    {h['a']:>5} {h['b']:>5} {h['c']:>5}  {h['gap']:>12.3e}  {period:>14.0f}")
print()

# ── 4. Cross-branch near-degeneracies ────────────────────────────────────────

print("Cross-branch near-degeneracies (a=2 vs b>=3):")
for b in range(3, 31):
    gap = abs(fv[2] - fv[b])
    period = 1.0/gap if gap > 0 else float('inf')
    print(f"  |f(2)-f({b:>2})| = {gap:.6f}  beat_period = {period:8.1f}")
print()

# ── 5. Figure ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.patch.set_facecolor('white')

# Panel 1: f(n) landscape
ax = axes[0]
ax.set_facecolor('#f8f9fa')
ns = list(range(2, N_PLOT+1))
fs_vals = [fv[n] for n in ns]

colors = []
for n in ns:
    pf = prime_factors(n)
    if n in (2, 4):
        colors.append('#e74c3c')
    elif len(pf) == 1:
        pk = round(math.log(n)/math.log(list(pf)[0]))
        if pk > 1:
            colors.append('#9b59b6')
        else:
            colors.append('#2980b9')
    else:
        colors.append('#95a5a6')

ax.vlines(ns, 0, fs_vals, colors=['#dddddd'], lw=0.7, zorder=1)
ax.scatter(ns, fs_vals, c=colors, s=28, zorder=3)

x_cont = np.linspace(1.1, N_PLOT, 1000)
ax.plot(x_cont, np.log(x_cont)/x_cont, 'k--', lw=0.8, alpha=0.35)

# Highlight f(2)=f(4)
ax.annotate('', xy=(4, fv[4]+0.006), xytext=(2, fv[2]+0.006),
            arrowprops=dict(arrowstyle='<->', color='#e74c3c', lw=1.5))
ax.text(3, fv[2]+0.014, 'f(2)=f(4)\nEXACT', ha='center', va='bottom',
        fontsize=8, color='#e74c3c', fontweight='bold')

# Mirror points (green x) for a=3,5,7,9
for a in [3, 5, 7, 9]:
    xm = find_mirror(a)
    ax.plot([xm, a], [fv[a], fv[a]], 'g:', lw=0.9, alpha=0.7)
    ax.scatter([xm], [fv[a]], c='#27ae60', s=25, zorder=4, marker='x')

ax.set_xlabel('n', fontsize=11)
ax.set_ylabel('f(n) = ln(n)/n', fontsize=11)
ax.set_title('Mertens frequency landscape\nand degeneracy structure', fontsize=10)
ax.set_xlim(1, N_PLOT+1)
ax.set_ylim(-0.01, 0.40)
legend_els = [
    mpatches.Patch(color='#e74c3c', label='Exact degeneracy (2,4)'),
    mpatches.Patch(color='#9b59b6', label='Prime powers p^k'),
    mpatches.Patch(color='#2980b9', label='Primes'),
    mpatches.Patch(color='#95a5a6', label='Composites'),
    plt.Line2D([0],[0], color='#27ae60', ls=':', marker='x', ms=6,
               label='Mirror in (1,e)'),
]
ax.legend(handles=legend_els, fontsize=7, loc='upper right')

# Panel 2: Prime-power harmonic series with exact resonances
ax2 = axes[1]
ax2.set_facecolor('#f8f9fa')
configs = [
    (2, '#e74c3c', 'p=2: f(16)+f(16)=f(2)'),
    (3, '#2980b9', 'p=3: f(9)+f(27)=f(3)'),
    (5, '#27ae60', 'p=5'),
]
y_positions = [0.8, 0.5, 0.2]
for (p, col, label), ybase in zip(configs, y_positions):
    powers = [(p**k, k) for k in range(1, 7) if p**k <= 250]
    for n, k in powers:
        height = fv[n] * 3
        ax2.bar(n, height, bottom=ybase, color=col, alpha=0.7, width=max(1, n*0.04))
        ax2.text(n, ybase+height+0.01, f'p^{k}={n}', ha='center', va='bottom',
                 fontsize=7, color=col)
    ax2.text(2, ybase+0.18, label, fontsize=8, color=col, fontweight='bold')

ax2.set_xlim(0, 140)
ax2.set_ylim(0, 1.1)
ax2.set_xlabel('n = p^k', fontsize=11)
ax2.set_title('Exact algebraic resonances\nf(p^a)+f(p^b)=f(p^c) for same prime p', fontsize=10)
ax2.set_yticks([])

# Panel 3: Beat curves for key pairs
ax3 = axes[2]
ax3.set_facecolor('#f8f9fa')
alpha = np.linspace(0, 80, 8000)

pairs = [
    (2, 4,  '#e74c3c', 'solid',  2.5, '(2,4) EXACT degeneracy: flat'),
    (2, 5,  '#2980b9', 'solid',  1.8, f'(2,5): Df={abs(fv[2]-fv[5]):.4f}, T=40.5'),
    (9, 27, '#27ae60', 'solid',  1.5, f'(9,27): f(9)/f(27)=2 exactly'),
    (11,13, '#8e44ad', 'dashed', 1.2, f'(11,13): Df={abs(fv[11]-fv[13]):.4f}'),
    (17,19, '#e67e22', 'dotted', 1.2, f'(17,19): Df={abs(fv[17]-fv[19]):.4f}'),
]
for a, b, col, ls, lw, label in pairs:
    df = abs(fv[a] - fv[b])
    P = 0.5*(1 + np.cos(2*np.pi*df*alpha))
    ax3.plot(alpha, P, color=col, ls=ls, lw=lw, label=label, alpha=0.9)

ax3.set_xlabel('Sweep parameter alpha', fontsize=11)
ax3.set_ylabel('P(even parity)', fontsize=11)
ax3.set_title('Bell-pair beat curves\nfor near-degenerate pairs', fontsize=10)
ax3.legend(fontsize=7.5, loc='upper right')
ax3.set_xlim(0, 80)
ax3.set_ylim(-0.05, 1.15)
ax3.axhline(0.5, color='#aaa', lw=0.7, ls=':')
ax3.grid(True, alpha=0.25)

plt.tight_layout()
out_fig = os.path.join(script_dir, 'near_degeneracy.png')
plt.savefig(out_fig, dpi=180, bbox_inches='tight')
plt.close()
print(f"Figure saved: near_degeneracy.png")

# Save JSON report
report = {
    'exact_degeneracy': exact_deg,
    'exact_additive_resonances': [(a,b,c,p) for a,b,c,p in exact_add],
    'top_accidental_resonances': unique_acc[:50],
    'cross_branch': [{'b': b, 'gap': abs(fv[2]-fv[b]),
                      'beat_period': 1/abs(fv[2]-fv[b])} for b in range(3,31)],
}
with open(os.path.join(script_dir, 'near_degeneracy_report.json'), 'w') as fh:
    json.dump(report, fh, indent=2)
print("Report saved: near_degeneracy_report.json")
