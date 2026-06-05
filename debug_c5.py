import sys, numpy as np
sys.path.insert(0, '/sessions/peaceful-exciting-bohr/mnt/Thought Experiments')
import simulate_c5_ramsey_spectroscopy as m

# ------- synthetic test: pure cosine at gamma_1 --------
g1 = m.ZEROS_30[0]
T = 6.21
t_s = np.linspace(0.0, T, 2000)
s_syn = np.cos(2.0 * g1 * t_s)

g_grid = np.linspace(g1 - 2.0, g1 + 2.0, 2000)
C_syn = m.matched_filter(t_s, s_syn, 1, g_grid, int(1e10))

pk = np.argmax(C_syn)
print(f"SYNTHETIC s=cos(2g1 t):")
print(f"  Peak at gamma={g_grid[pk]:.4f}  (true g1={g1:.4f})")
print(f"  HWHM = {np.sum(C_syn > C_syn[pk]/2) * (g_grid[1]-g_grid[0]) / 2:.4f}")
print(f"  Theory HWHM = pi/T = {np.pi/T:.4f}")

# ------- real Mertens signal --------
t_vals, s_vals = m.mertens_signal(50_000, 100)
N_T = 2000
t_s2 = np.linspace(t_vals[0], t_vals[-1], N_T)
s_s2 = np.interp(t_s2, t_vals, s_vals)
T2 = t_s2[-1] - t_s2[0]

g_grid2 = np.linspace(g1 - 3.0, g1 + 3.0, 3000)
C_real = m.matched_filter(t_s2, s_s2, 1, g_grid2, int(1e10))

pk2 = np.argmax(C_real)
above2 = np.sum(C_real > C_real[pk2]/2)
print(f"\nREAL Mertens (N<=50000, T={T2:.2f}):")
print(f"  Peak at gamma={g_grid2[pk2]:.4f}  (true g1={g1:.4f})")
print(f"  C_peak={C_real[pk2]:.5f}  C_mean={C_real.mean():.5f}  C_std={C_real.std():.5f}")
print(f"  HWHM = {above2 * (g_grid2[1]-g_grid2[0]) / 2:.4f}")
print(f"  Theory HWHM = pi/T = {np.pi/T2:.4f}")
print(f"  Peak offset from g1 = {g_grid2[pk2]-g1:.4f}  (2pi/T={2*np.pi/T2:.4f})")
top5 = np.argsort(C_real)[-5:][::-1]
print(f"  Top 5 peaks: {[(round(g_grid2[i],3), round(C_real[i],5)) for i in top5]}")
