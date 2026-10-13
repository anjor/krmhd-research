"""d09: mirror symmetry: the map, its action on slices, parities of statistics, the balanced-turbulence statement,
the parity lemma under frozen-line dynamics, and what fourth-order statistics can still do.  The script checks that
gandalf_step commutes with the mirror map (one step for both schemes, both closures and Lambda = +-sqrt5, and a
200-step pair of forced runs with mirrored noise), that the map sends slice u to slice -u, the parity of each
statistic in the table, the action on pair forcing, the algebra of the parity lemma for random ladders, and the
fourth-order difference between correlated and uncorrelated injection.  It supports Section 7 (mirror symmetry and
parity lemma).

Mirror map M: z -> -z, v_par -> -v_par, z+ <-> z-, g_m -> (-1)^m g_m.  On GANDALF arrays (axis 0 = k_z):
  z+(k) -> z-(k~), z-(k) -> z+(k~), g(k) -> S g(k~),  k~ = (kx, ky, -kz), S = diag((-1)^m);  Phi -> Phi(-z), Psi -> -Psi(-z).
 M1 gandalf_step commutes with M (one step; IMEX and Lawson; zero and symmetric closure; Lambda = +-sqrt5;
    nu, eta, eta_z > 0), and a forced run started from M X with mirrored noise (pair forcing at phi + pi) stays the
    mirror image of the run at phi (200 steps).
 M2 slices: S v_u = +- v_{-u}; a_u(M X)(z) = +- a_{-u}(X)(-z).
 M3 parity table: even (W(m), W, Pi_m full, Pi^+-_m k_z-split, eps_nu, W(k_perp), Q_(u^2), sum_u <a_u^4>, <g_m^4>,
    E+ + E-), odd (Gamma, Gamma(m), H, Q_(u^3), sum_k sgn(k_z)|g_m|^2, E+ - E-), slice energies swapped E_u <-> E_-u.
 M4 pair forcing F = (f, r e^{i phi} f): M F_phi = F_{phi+pi}[f(k~)].
 M5 parity lemma, algebra: F(kappa) = exp[-t(i s kappa A + C)] = D R D^{-1}, R real, R(-kappa) = S R S; the
    linear-in-r part of <g_m, grad_par^q g_n> is odd in kappa (vanishes after the symmetric spectral measure of a
    real pattern) iff q + m + n is even, i.e. iff the statistic is mirror-even.  Random M, Lambda, closures, C.
 M6 parity lemma in GANDALF runs: see d09b_parity_lemma_gandalf.py.
 M7 fourth order: correlated injection (f, rho f) and uncorrelated injection with the same variances inject the
    same free energy p + rho^2 and give the same second-order statistics, but different slice fourth moments: sum_u E[a_u^4] differs by 12 sigma^4 sum_j v_j0^4 p^2 delta_j^2 (delta_j = sqrt2 rho u_j),
    even in rho (so phi = 0 and phi = pi agree, phi = pi/2 differs).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from scipy.linalg import expm  # noqa: E402

from krmhd.timestepping import gandalf_step  # noqa: E402
from dcore import (Box, Checks, SQRT5, W_m, flux_full, flux_kz, ladder, log_run, make_rhs, make_state,  # noqa: E402
                   pweights, slices, timer, Wk_perp)

chk = Checks()


def mz(a):
    return np.roll(np.flip(np.asarray(a), 0), 1, 0)


def mirror(zp, zm, g):
    M = g.shape[-1] - 1
    S = (-1.0) ** np.arange(M + 1)
    return mz(zm), mz(zp), mz(g) * S


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / max(np.abs(a).max(), np.abs(b).max(), 1e-300))


# ------------------------------------------------------------------------------------------- M1
N, M = 16, 6
box = Box(N)
rng = np.random.default_rng(4)
zp = box.random_field(rng, (1, 4), 3)
zm = box.random_field(rng, (1, 4), 3)
zp *= 2 / box.rms_grad(zp)
zm *= 1 / box.rms_grad(zm)
g = box.random_field(rng, (1, 5), 4, extra=(M + 1,))
worst = 0.0
clock = timer()
for scheme in ("imex_rk222", "lawson_rk4"):
    for closure in ("zero", "symmetric"):
        for Lam in (SQRT5, -SQRT5):
            kw = dict(eta=0.3, v_A=1.0, nu=0.7, hyper_r=2, hyper_n=4, scheme=scheme, eta_z=0.2, hyper_rz=1, closure=closure)
            st = make_state(box, zp, zm, g, Lam, 1.7, nu=0.7)
            a = gandalf_step(st, 0.003, **kw)
            stm = make_state(box, *mirror(zp, zm, g), Lam, 1.7, nu=0.7)
            b = gandalf_step(stm, 0.003, **kw)
            ma = mirror(a.z_plus, a.z_minus, a.g)
            e = max(rel(ma[0], b.z_plus), rel(ma[1], b.z_minus), rel(ma[2], b.g))
            worst = max(worst, e)
log_run("d09_mirror.py", "mirror equivariance: one gandalf_step of X and of M X for 2 schemes x 2 closures x "
        "2 Lambda (16 single steps)", N, M, 16, clock(), dict(dt=0.003))
chk("M1 gandalf_step(M X) = M gandalf_step(X): IMEX/Lawson x zero/symmetric x Lambda = +-sqrt5, nu, eta, eta_z on",
    worst < 1e-13, f"max rel mismatch {worst:.1e}")

# forced pathwise test: pair forcing at phi (run a) vs mirrored state + mirrored noise (run b, phase phi + pi)
Lam, beta, phi_f, r = SQRT5, 1.0, 0.7, 0.8
nst, dt = 200, 0.003
fm = (np.sqrt(box.KP2) >= box.k1 - 1e-9) & (np.sqrt(box.KP2) <= 2 * box.k1 + 1e-9) & (np.abs(np.abs(box.KZ) - box.k1) < 1e-9)
fm = fm & box.mask
sta = make_state(box, zp, zm, g, Lam, beta)
stb = make_state(box, *mirror(zp, zm, g), Lam, beta)
clock = timer()
growth = []
for n in range(nst):
    nz = [np.fft.rfftn(rng.standard_normal((N, N, N))) * fm / N ** 1.5 for _ in range(3)]
    Fa = np.zeros(g.shape, complex)
    Fa[..., 0] = nz[2]
    Fa[..., 1] = r * np.exp(1j * phi_f) * nz[2]
    Fb = np.zeros(g.shape, complex)
    Fb[..., 0] = mz(nz[2])
    Fb[..., 1] = r * np.exp(1j * (phi_f + np.pi)) * mz(nz[2])
    amp = 0.3 * np.sqrt(dt)
    sta = make_state(box, sta.z_plus + amp * nz[0], sta.z_minus + amp * nz[1], sta.g + amp * Fa, Lam, beta)
    stb = make_state(box, stb.z_plus + amp * mz(nz[1]), stb.z_minus + amp * mz(nz[0]), stb.g + amp * Fb, Lam, beta)
    sta = gandalf_step(sta, dt, eta=0.5, v_A=1.0, nu=1.0, hyper_r=2, hyper_n=4, scheme="imex_rk222")
    stb = gandalf_step(stb, dt, eta=0.5, v_A=1.0, nu=1.0, hyper_r=2, hyper_n=4, scheme="imex_rk222")
    if (n + 1) % 50 == 0:
        ma = mirror(sta.z_plus, sta.z_minus, sta.g)
        growth.append(max(rel(ma[0], stb.z_plus), rel(ma[1], stb.z_minus), rel(ma[2], stb.g)))
log_run("d09_mirror.py", "pathwise mirror test: pair forcing at phi vs mirrored run at phi+pi (two runs)", N, M, 2 * nst,
        clock(), dict(dt=dt))
Wa, Wb = W_m(box, sta.g), W_m(box, stb.g)
pa = pweights(M, Lam)
A = ladder(M, Lam)
Ga = sum(2 * pa[m] * A[m, m + 1] * box.inner(np.asarray(sta.g)[..., m], np.asarray(sta.g)[..., m + 1]) for m in range(M))
Gb = sum(2 * pa[m] * A[m, m + 1] * box.inner(np.asarray(stb.g)[..., m], np.asarray(stb.g)[..., m + 1]) for m in range(M))
print(f"    forced pathwise test: mismatch |M X_a - X_b| / |X| after 50, 100, 150, 200 steps: "
      + ", ".join(f"{x:.1e}" for x in growth) + f";  W: {rel(Wa, Wb):.1e},  Gamma_a = {Ga:.4e}, Gamma_b = {Gb:.4e}")
chk("M1 forced run from M X with mirrored noise and phase phi + pi stays the mirror image", growth[-1] < 1e-10,
    f"{growth[-1]:.1e}")
chk("M1 ... W(m) equal and Gamma opposite in the two runs", rel(Wa, Wb) < 1e-10 and abs(Ga + Gb) < 1e-10 * abs(Ga),
    f"W {rel(Wa, Wb):.1e}, Gamma_a + Gamma_b = {Ga + Gb:.1e}")

# ------------------------------------------------------------------------------------------- M2
for Mm in (8, 9, 128):
    u, V, WL = slices(Mm, SQRT5)
    S = (-1.0) ** np.arange(Mm + 1)
    jm = [int(np.argmin(np.abs(u + uj))) for uj in u]
    e = max(min(np.abs(S * V[:, j] - V[:, k]).max(), np.abs(S * V[:, j] + V[:, k]).max()) for j, k in enumerate(jm))
    chk(f"M2 M={Mm}: spectrum symmetric and S v_u = +- v_(-u)", e < 1e-9 and np.abs(np.sort(u) + np.sort(u)[::-1]).max() < 1e-9,
        f"{e:.1e}")
M = 9
u, V, WL = slices(M, SQRT5)
S = (-1.0) ** np.arange(M + 1)
zp = box.random_field(rng, (1, 4), 3)
zm = box.random_field(rng, (1, 4), 3)
zp *= 2 / box.rms_grad(zp)
zm *= 0.7 / box.rms_grad(zm)
g = box.random_field(rng, (1, 5), 4, extra=(M + 1,))
zpm, zmm, gmir = mirror(zp, zm, g)
a = g @ WL.T
am = gmir @ WL.T
jm = [int(np.argmin(np.abs(u + uj))) for uj in u]
e = 0.0
for j in range(M + 1):
    sgn = np.sign(np.real(np.vdot(S * V[:, j], V[:, jm[j]])))
    e = max(e, rel(am[..., j], sgn * mz(a[..., jm[j]])))
chk("M2 slices of M X are the mirrored slices -u of X", e < 1e-13, f"{e:.1e}")

# ------------------------------------------------------------------------------------------- M3
p = pweights(M, SQRT5)
A = ladder(M, SQRT5)
psi, psim = (zp - zm) / 2, (zpm - zmm) / 2


def stats(zp_, zm_, g_, psi_):
    a_ = g_ @ WL.T
    E = np.array([box.norm2(a_[..., j]) for j in range(M + 1)])
    Pk = flux_kz(box, g_, 1.0, A, p, per_mode=True) * box.w[..., None]
    Gm = np.array([2 * p[m] * A[m, m + 1] * box.inner(g_[..., m], g_[..., m + 1]) for m in range(M)])
    q4 = [float(np.mean(box.to_real(g_[..., m], 2) ** 4)) for m in (0, 1, 2)]
    a4 = sum(float(np.mean(box.to_real(a_[..., j], 2) ** 4)) for j in range(M + 1))
    rates = np.where(np.arange(M + 1) >= 2, (np.arange(M + 1) / M) ** 4, 0.0)
    Ep, Em = box.rms_grad(zp_) ** 2 / 4, box.rms_grad(zm_) ** 2 / 4
    even = dict(W_m=W_m(box, g_), W=np.sum(E), Pi_full=flux_full(box, g_, psi_, 1.0, A, p),
                Pi_plus=np.where(Pk > 0, Pk, 0).sum(axis=(0, 1, 2)), Pi_minus=np.where(Pk < 0, Pk, 0).sum(axis=(0, 1, 2)),
                eps_nu=np.sum(rates * p * W_m(box, g_)), Wkperp=Wk_perp(box, g_, p), Q_u2=np.sum(u ** 2 * E),
                slice_quartic_sum=a4, g_quartic=np.array(q4), E_plus_plus_E_minus=Ep + Em)
    odd = dict(Gamma=np.sum(u * E), Gamma_m=Gm, H=np.sum(E / u), Q_u3=np.sum(u ** 3 * E),
               W_kz_odd=np.einsum("zyxm->m", (box.w * np.sign(box.KZ))[..., None] * np.abs(g_) ** 2),
               E_plus_minus_E_minus=Ep - Em)
    return even, odd, E


ev, od, E = stats(zp, zm, g, psi)
evm, odm, Em_ = stats(zpm, zmm, gmir, psim)
for k in ev:
    e = rel(ev[k], evm[k])
    chk(f"M3 even: {k}", e < 1e-11, f"{e:.1e}")
for k in od:
    e = rel(od[k], -np.asarray(odm[k]))
    chk(f"M3 odd: {k}", e < 1e-11 and np.abs(od[k]).max() > 0, f"{e:.1e}")
chk("M3 slice energies swap E_u <-> E_-u", rel(Em_, E[jm]) < 1e-12, f"{rel(Em_, E[jm]):.1e}")

# ------------------------------------------------------------------------------------------- M4
f0 = box.random_field(rng, (1, 2), 1)
for phi_ in (0.0, 0.7, np.pi / 2):
    F = np.zeros(f0.shape + (M + 1,), complex)
    F[..., 0] = f0
    F[..., 1] = 0.8 * np.exp(1j * phi_) * f0
    MF = mz(F) * S
    G = np.zeros_like(F)
    G[..., 0] = mz(f0)
    G[..., 1] = 0.8 * np.exp(1j * (phi_ + np.pi)) * mz(f0)
    chk(f"M4 M F_phi = F_(phi + pi)[f(k~)] at phi = {phi_:.2f}", rel(MF, G) < 1e-15)

# ------------------------------------------------------------------------------------------- M5
w_even, w_odd = 0.0, 0.0
for trial in range(300):
    Mm = int(rng.integers(3, 40))
    Lam = [SQRT5, -SQRT5, rng.uniform(1.2, 9), -rng.uniform(0.2, 9)][trial % 4]
    Am = ladder(Mm, Lam, "symmetric" if trial % 2 else "zero")
    Cm = np.diag(rng.uniform(0, 3, Mm + 1) * (np.arange(Mm + 1) >= 2))
    tt, s, kap = rng.uniform(0.01, 3), rng.uniform(0.3, 2), rng.normal() * 5
    Fp = expm(-tt * (1j * s * kap * Am + Cm))
    Fn = expm(-tt * (-1j * s * kap * Am + Cm))
    D = np.diag((-1j) ** np.arange(Mm + 1))
    R = np.linalg.solve(D, Fp @ D)
    Sm = np.diag((-1.0) ** np.arange(Mm + 1))
    Rn = np.linalg.solve(D, Fn @ D)
    sc = np.abs(Fp).max() ** 2
    w_even = max(w_even, np.abs(np.imag(R)).max() / np.abs(R).max(), np.abs(Rn - Sm @ R @ Sm).max() / np.abs(R).max())
    for q in range(3):
        for m_ in range(Mm + 1):
            for n_ in range(Mm + 1):
                h1 = lambda F_, k_: np.conj(F_[m_, 0]) * (1j * k_) ** q * F_[n_, 1]  # noqa: E731
                h2 = lambda F_, k_: np.conj(F_[m_, 1]) * (1j * k_) ** q * F_[n_, 0]  # noqa: E731
                tot = h1(Fp, kap) + h1(Fn, -kap) + h2(Fp, kap) + h2(Fn, -kap)
                if (q + m_ + n_) % 2 == 0:
                    w_even = max(w_even, abs(tot) / sc / max(1, abs(kap)) ** q)
                else:
                    w_odd = max(w_odd, abs(tot) / sc / max(1, abs(kap)) ** q)
chk("M5 R real, R(-kappa) = S R S, and the linear-in-r part of <g_m, grad_par^q g_n> is kappa-odd for q+m+n even",
    w_even < 1e-11, f"max residual {w_even:.1e} (300 random M, Lambda, closures, C, t, s, kappa)")
chk("M5 ... while for q+m+n odd (mirror-odd statistics, e.g. Gamma(m)) it is not", w_odd > 1e-3, f"max {w_odd:.2f}")

# M6 (parity lemma in GANDALF runs) lives in d09b_parity_lemma_gandalf.py
# ------------------------------------------------------------------------------------------- M7
for Mm in (8, 64):
    u, V, WL = slices(Mm, SQRT5)
    p0 = 1 - 1 / SQRT5
    v0 = V[0]
    chk(f"M7 M={Mm}: v_j(1) = sqrt2 u_j v_j(0)", np.abs(V[1] - np.sqrt(2) * u * v0).max() < 1e-12)
    for rho in (0.5, 1.0):
        c = v0 * (p0 + np.sqrt(2) * rho * u)               # correlated: a_j = c_j f
        var_unc = v0 ** 2 * (p0 ** 2 + 2 * rho ** 2 * u ** 2)   # uncorrelated g_1 forcing with the same variance
        second = (np.sum(c ** 2), np.sum(var_unc))
        fourth = (3 * np.sum(c ** 4), 3 * np.sum(var_unc ** 2))
        pred = 3 * np.sum(v0 ** 4 * 4 * p0 ** 2 * 2 * rho ** 2 * u ** 2)
        c_m = v0 * (p0 - np.sqrt(2) * rho * u)
        chk(f"M7 M={Mm} rho={rho}: same injected free energy p + rho^2 (2nd order identical)",
            abs(second[0] - second[1]) < 1e-12 and abs(second[0] - (p0 + rho ** 2)) < 1e-12)
        chk(f"M7 M={Mm} rho={rho}: sum_u E[a_u^4] differs by 12 sum v0^4 p^2 delta^2 (sigma = 1); even in rho",
            abs(fourth[0] - fourth[1] - pred) < 1e-12 * fourth[0] and abs(3 * np.sum(c_m ** 4) - fourth[0]) < 1e-12 * fourth[0],
            f"correlated {fourth[0]:.5f}, uncorrelated {fourth[1]:.5f}, relative difference {(fourth[0] - fourth[1]) / fourth[1]:.3f}")
chk.finish()
