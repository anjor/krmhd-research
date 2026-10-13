"""c05 (independent check): the finite-k_perp rho_i advection lemmas A, A' and B, checked by other routes (direct
Fourier-mode sums with 3D wavevectors, a velocity grid), and the role of the self-consistent field: the kernel of a
conserved form is free at k_perp = 0; under gyroaveraged advection it must be local in v_perp but may couple v_par;
the H kernel is Lambda-independent because L = U Pi; W is conserved by GANDALF's right-hand side although its kernel
is not a multiplication operator in v, so "conserved only if the kernel is a k-independent multiplication in v" is
not a valid criterion; and in an electrostatic gyrokinetic toy model the field energy is conserved only through the
self-consistent field while Gamma is broken at O(k_perp^2 rho^2). It supports Section 4 (finite k_perp rho_i) of
the paper.

 A1  triad formula (as in s09 K1) by direct Fourier-mode sums (dictionary of modes, no FFT): d_t theta_k = sum_{p+q=k} (p x q)_z chi_p theta_q.
 A2  caveat to "iff K_k is k-independent": modes with k_perp = 0 are never advected and never fed, so K at k_perp = 0 is free.
     3D example: K_k = K for k_perp != 0 and K_k = K' != K at k_perp = 0 conserves Q for random chi(x,y,z), theta.
 A3  Lemma A': with gyroaveraged advection a k-independent K that is local in v_perp but couples v_par values (block diagonal)
     is conserved; one that couples v_perp values is not.  So the condition is "local in v_perp", weaker than "multiplication in v".
 B1  Lemma B via the factorisation L = U Pi (U = multiplication by u): on a velocity grid, D Pi L^{-1} = D U^{-1} = diag(w/(u F_M))
     for every Lambda, while the W and Gamma kernels D Pi and D Pi U Pi depend on Lambda.
 C1  counterexample to the criterion "a conserved quadratic form has a k-independent kernel that is a
     multiplication operator in v": W is conserved although its kernel is not a multiplication operator in v
     (KRMHD: D Pi has a rank-one off-diagonal part; conservation checked on GANDALF's krmhd_rhs, zero closure).
 C2  electrostatic gyrokinetic toy at finite k_perp rho (Boltzmann electrons, 2D, self-consistent phi_k = sum_v w J0 g_k / D_k):
     the field part E2 = sum_k |sum_v w J0 g_k|^2 / D_k (kernel k-dependent and nonlocal in v) is conserved to round-off by the
     self-consistent nonlinearity, but not when the same chi is prescribed (passive): the conservation comes from the field equation.
 C3  in the same toy, Gamma_k = [g, L_k g]_k with L_k = U Pi_k is broken by the nonlinearity at finite k rho, and dGamma/dt
     scales as rho^2 (illustration of an O(k_perp^2 rho^2) breaking in an electrostatic model; not the KRMHD/ITEF ordering).
Exit 1 on any failure.
"""
import sys
import itertools
import numpy as np
from scipy.special import j0, i0e

fails = []


def check(name, ok, info=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)
    if not ok:
        fails.append(name)


rng = np.random.default_rng(31)
cross = lambda p, q: p[0] * q[1] - p[1] * q[0]


def advect_modes(chi, th):
    """chi: {k: complex}, th: {k: vector}; returns d_t th_k = sum_{p+q=k} (p x q)_z chi_p th_q for all k reached."""
    out = {}
    for p, cp in chi.items():
        for q, tq in th.items():
            k = tuple(a + b for a, b in zip(p, q))
            out[k] = out.get(k, 0) + cross(p, q) * cp * tq
    return out


def dQdt(chi, th, Kfun, Mfun=None):
    """sum_k 2 Re th_k^H K_k d_t th_k, d_t th = -{chi_v, th_v}; Mfun(p) = multiplication operator (vector) for gyroaveraging."""
    if Mfun is None:
        d = advect_modes(chi, th)
    else:
        d = {}
        for p, cp in chi.items():
            for q, tq in th.items():
                k = tuple(a + b for a, b in zip(p, q))
                d[k] = d.get(k, 0) + cross(p, q) * cp * (Mfun(p) * tq)
    tot = 0.0
    for k, tk in th.items():
        if k in d:
            tot += 2 * np.real(np.conj(tk) @ Kfun(k) @ d[k])
    return tot


nv = 4
herm = lambda X: X + X.conj().T
print("A1  triad formula by direct mode sums")
q = (2, 1, 0); p = (1, 3, 1)
qp = tuple(a + b for a, b in zip(q, p))
neg = lambda k: tuple(-a for a in k)
a1 = rng.standard_normal(nv) + 1j * rng.standard_normal(nv); a2 = rng.standard_normal(nv) + 1j * rng.standard_normal(nv)
th = {q: a1, neg(q): a1.conj(), qp: a2, neg(qp): a2.conj()}
cp_ = 0.7 - 0.4j
chi = {p: cp_, neg(p): np.conj(cp_)}
K0 = herm(rng.standard_normal((nv, nv)) + 0j).real; K1 = herm(rng.standard_normal((nv, nv)) + 0j).real
Kb = lambda k: K0 + (k[0]**2 + k[1]**2 + 0.5 * k[2]**2) / 10 * K1
direct = dQdt(chi, th, Kb)
formula = 4 * cross(p, q) * np.real(cp_ * np.conj(a2) @ (Kb(qp) - Kb(q)) @ a1)
check("dQ/dt = 4 (p x q)_z Re[chi_p theta_{q+p}^H (K_{q+p} - K_q) theta_q] (3D wavevectors)", abs(direct - formula) < 1e-12 * abs(formula),
      f"direct {direct:.12f}, formula {formula:.12f}")


def random_real_state(kmax, nvec, with_kperp0=True):
    th = {}
    for k in itertools.product(range(-kmax, kmax + 1), repeat=3):
        if k == (0, 0, 0) or k in th:
            continue
        if not with_kperp0 and k[0] == 0 and k[1] == 0:
            continue
        v = rng.standard_normal(nvec) + 1j * rng.standard_normal(nvec)
        th[k] = v; th[neg(k)] = v.conj()
    return th


print("A2  k_perp = 0 modes: K there is unconstrained")
th = random_real_state(2, nv)
chi = random_real_state(2, 1); chi = {k: v[0] for k, v in chi.items()}
Kp = herm(rng.standard_normal((nv, nv)) + 0j).real
Kcav = lambda k: Kp if (k[0] == 0 and k[1] == 0) else K0
dq = dQdt(chi, th, Kcav)
scale = abs(dQdt(chi, th, Kb))
nk0 = sum(1 for k in th if k[0] == 0 and k[1] == 0)
check(f"K_k = K' (k_perp = 0, {nk0} modes populated), K (otherwise): dQ/dt = 0", abs(dq) < 1e-12 * scale, f"dQ/dt = {dq:.1e} (k-dependent control {scale:.2e})")
dq_kz = dQdt(chi, th, lambda k: K0 + (k[2]**2) * K1 if not (k[0] == 0 and k[1] == 0) else Kp)
check("control: K depending on k_z at k_perp != 0 is not conserved (so K must be constant on k_perp != 0)", abs(dq_kz) > 1e-6 * scale,
      f"dQ/dt = {dq_kz:.2e}")

print("A3  Lemma A': gyroaveraged advection, kernels local / nonlocal in v_perp")
vperp = np.array([0.4, 0.4, 1.3, 1.3])                    # two v_par values at each of two v_perp values
Mfun = lambda p: j0(np.hypot(p[0], p[1]) * vperp)
B = rng.standard_normal((2, 2)); Kblock = np.zeros((4, 4))
Kblock[:2, :2] = B + B.T; B = rng.standard_normal((2, 2)); Kblock[2:, 2:] = B + B.T     # couples v_par at fixed v_perp
Knonloc = Kblock.copy(); Knonloc[0, 2] = Knonloc[2, 0] = 0.7                            # couples v_perp = 0.4 and 1.3
th = random_real_state(2, 4)
d_loc = dQdt(chi, th, lambda k: Kblock, Mfun); d_non = dQdt(chi, th, lambda k: Knonloc, Mfun)
d_ref = dQdt(chi, th, lambda k: Kblock + 0.1 * (k[0]**2 + k[1]**2) * np.eye(4), Mfun)
check("block-diagonal K (local in v_perp, nonlocal in v_par): conserved", abs(d_loc) < 1e-12 * abs(d_ref), f"{d_loc:.1e} (scale {abs(d_ref):.2e})")
check("K coupling different v_perp: not conserved", abs(d_non) > 1e-6 * abs(d_ref), f"{d_non:.2e}")

print("B1  Lemma B from L = U Pi")
ug = np.linspace(-6, 6, 241)                              # 241 points: contains u = 0; drop it for U^{-1}
ug = ug[ug != 0]; w = np.full(ug.size, 12 / 240.0)
FM = np.exp(-ug**2) / np.sqrt(np.pi)
U = np.diag(ug); D = np.diag(w / FM)
ok = True; dep = []
for Lam in (np.sqrt(5.0), -np.sqrt(5.0), 3.7, -0.2, 1e3):
    Pi = np.eye(ug.size) - np.outer(FM, w) / Lam            # [f,g] = f^T D Pi g
    L = U - np.outer(ug * FM, w) / Lam                       # L G = u G - u F_M M0/Lambda (M0 = w.G)
    ok &= np.allclose(L, U @ Pi, atol=1e-14)
    Hk = D @ Pi @ np.linalg.inv(L)
    ok &= np.abs(Hk - np.diag(w / (ug * FM))).max() < 1e-8 * np.abs(np.diag(w / (ug * FM))).max()
    ok &= np.allclose(D @ Pi, (D @ Pi).T) and np.allclose(D @ Pi @ L, (D @ Pi @ L).T)
    dep.append(np.abs(D @ Pi - D).max())
check("L = U Pi; D Pi L^{-1} = diag(w/(u F_M)) for every Lambda; D Pi, D Pi L symmetric", ok)
check("W kernel D Pi depends on Lambda (rank-one part), so a k-dependent Lambda_k changes it", min(dep) > 0, f"max|D Pi - D| = {min(dep):.2e}..{max(dep):.2e}")

print("C1  W conserved although its kernel is not a multiplication operator (KRMHD, GANDALF RHS)")
try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from krmhd.physics import KRMHDState
    from krmhd.spectral import SpectralGrid3D
    from krmhd.timestepping import krmhd_rhs
    N = 16; M = 10; Lam = np.sqrt(5.0)
    grid = SpectralGrid3D.create(Nx=N, Ny=N, Nz=N)
    mask = np.asarray(grid.dealias_mask)

    def rf(amp):
        F = np.fft.rfftn(rng.standard_normal((N, N, N)), axes=(0, 1, 2)) * mask
        F[0, 0, 0] = 0
        return F / np.abs(np.fft.irfftn(F, s=(N, N, N), axes=(0, 1, 2))).max() * amp
    zp, zm = rf(20.0), rf(15.0)
    gg = np.stack([rf(1.0) for _ in range(M + 1)], -1)
    st = KRMHDState(z_plus=jnp.asarray(zp), z_minus=jnp.asarray(zm), g=jnp.asarray(gg), M=M, beta_i=1.0, v_th=1.0, nu=0.0,
                    Lambda=Lam, time=0.0, grid=grid)
    dg = np.asarray(krmhd_rhs(st, eta=0.0, v_A=1.0, closure="zero").g)
    p0 = np.ones(M + 1); p0[0] = 1 - 1 / Lam
    wt = np.ones(N // 2 + 1) * 2; wt[0] = 1; wt[-1] = 1                    # rfft Parseval weights in k_x
    dW = np.sum(wt[None, None, :, None] * p0 * np.real(np.conj(gg) * dg))
    sc = np.sum(wt[None, None, :, None] * np.abs(gg) * np.abs(dg))
    check("d/dt sum_k g^H P0 g = 0 with GANDALF's RHS (W kernel = identity minus rank one, not a multiplication)",
          abs(dW) < 1e-13 * sc, f"|dW/dt|/scale = {abs(dW) / sc:.1e}")
except Exception as e:  # pragma: no cover
    check(f"GANDALF part ran ({e})", False)

print("C2/C3  electrostatic gyrokinetic toy (2D, Boltzmann electrons)")
N = 32
k1 = np.fft.fftfreq(N, 1.0 / N)
KX, KY = np.meshgrid(k1, k1, indexing='ij'); KK = np.hypot(KX, KY)
keep = (np.abs(KX) < N / 3) & (np.abs(KY) < N / 3)
vpar = np.array([-1.6, -0.55, 0.55, 1.6]); vpe = np.array([0.35, 1.1, 2.0])
VP, VQ = np.meshgrid(vpar, vpe, indexing='ij'); VP = VP.ravel(); VQ = VQ.ravel()
F0 = np.exp(-VP**2 - VQ**2) / np.pi**1.5
wv = F0 / F0.sum()                                         # quadrature weights (exact values irrelevant)
nvv = VP.size


def bracket(a, b):
    dx = lambda f: np.real(np.fft.ifft2(1j * KX * np.fft.fft2(f)))
    dy = lambda f: np.real(np.fft.ifft2(1j * KY * np.fft.fft2(f)))
    out = dx(a) * dy(b) - dy(a) * dx(b)
    return np.real(np.fft.ifft2(np.fft.fft2(out) * keep))


def state():
    g = np.zeros((nvv, N, N))
    for i in range(nvv):
        G = np.fft.fft2(rng.standard_normal((N, N))) * keep * (KK <= 8) * (KK > 0)
        g[i] = np.real(np.fft.ifft2(G))
    return g / np.abs(g).max()


def toy_rates(g, rho, tau=1.0, passive_chi=None):
    J = np.array([j0(KK * rho * VQ[i]) for i in range(nvv)])            # J0(k rho vperp), per velocity
    b = 0.5 * (KK * rho)**2
    Dk = 1 / tau + 1 - i0e(b)                                            # 1/tau + 1 - Gamma0(b)
    gk = np.fft.fft2(g, axes=(1, 2)) * keep
    nk = np.sum(wv[:, None, None] * J * gk, 0)
    uk = np.sum((wv * VP)[:, None, None] * J * gk, 0)
    phik = nk / Dk
    chik = phik if passive_chi is None else passive_chi
    dg = np.array([-bracket(np.real(np.fft.ifft2(J[i] * chik)), g[i]) for i in range(nvv)])
    dgk = np.fft.fft2(dg, axes=(1, 2)) * keep
    dnk = np.sum(wv[:, None, None] * J * dgk, 0); duk = np.sum((wv * VP)[:, None, None] * J * dgk, 0)
    E1dot = np.sum(wv[:, None, None] / F0[:, None, None] * 2 * g * dg)
    E2dot = np.sum(2 * np.real(np.conj(nk) * dnk) / Dk)
    # Gamma = sum_x sum_v (w v_par/F0) g^2 + (1/N^2) sum_k (2/D_k) Re(n_k^* u_k)   (discrete Parseval, unnormalised FFT)
    Gdot1 = np.sum((wv * VP / F0)[:, None, None] * 2 * g * dg)
    Gdot2 = np.sum(2 * np.real(np.conj(dnk) * uk + np.conj(nk) * duk) / Dk) / (N * N)
    Gdot = Gdot1 + Gdot2
    E2 = np.sum(np.abs(nk)**2 / Dk)
    Gam = np.sum((wv * VP / F0)[:, None, None] * g * g) + np.sum(2 * np.real(np.conj(nk) * uk) / Dk) / (N * N)
    # scale: sum of the absolute values of the two field-term contributions
    Gsc = np.sum(2 * (np.abs(dnk * uk) + np.abs(nk * duk)) / Dk) / (N * N)
    return E1dot, E2dot, E2, Gdot, Gam, phik, Gsc


g = state() * 3.0
rho = 0.3
E1d, E2d, E2, Gd, Gam, phik, _ = toy_rates(g, rho)
chi_pass = np.fft.fft2(state()[0]) * keep * 3.0 * N * N / 40
E1p, E2p, E2_, Gp_, _, _, _ = toy_rates(g, rho, passive_chi=chi_pass)
e2scale = abs(E2p)
print(f"    rho = {rho}: self-consistent dE2/dt = {E2d:.2e}, dE1/dt = {E1d:.2e}; passive chi: dE2/dt = {E2p:.2e}")
check("C2 field energy (k-dependent, v-nonlocal kernel) conserved by the self-consistent nonlinearity", abs(E2d) < 1e-10 * e2scale)
check("C2 ... and not by the same advection with a prescribed chi", abs(E2p) > 1e-4 * abs(E2_))
rows = []
for rho in (0.0025, 0.005, 0.01, 0.02):                  # k rho v_perp <= 0.4 for the largest rho (asymptotic regime)
    _, _, _, Gd, Gam, _, Gsc = toy_rates(g, rho)
    rows.append((rho, abs(Gd), Gsc, Gam))
rate = np.polyfit(np.log([r[0] for r in rows]), np.log([r[1] for r in rows]), 1)[0]
print("    |dGamma/dt| vs rho: " + ", ".join(f"rho={r:.4f}: {d:.3e} (scale {sc:.2e}, Gamma {G:.3e})" for r, d, sc, G in rows) + f";  slope {rate:.3f}")
_, _, _, Gd0, Gam0, _, Gsc0 = toy_rates(g, 0.0)
for rho in (0.1, 0.3):
    _, _, _, Gdx, _, _, Gscx = toy_rates(g, rho)
    print(f"    rho={rho}: |dGamma/dt| = {abs(Gdx):.3e} = {abs(Gdx) / Gscx:.2e} of the field-term scale")
check("C3 Gamma conserved at rho = 0 (round-off) and broken at finite rho with |dGamma/dt| ~ rho^2 in this toy",
      abs(Gd0) < 1e-12 * Gsc0 and abs(rate - 2) < 0.1, f"rho=0: |dGamma/dt|/scale = {abs(Gd0) / Gsc0:.1e}")

if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("ALL c05 CHECKS PASSED")
