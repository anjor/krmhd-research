"""d06: the k_z-based phase-mixing / unmixing decomposition is not invariant under field-line tilt.  For linear
streaming along a uniformly tilted field the script checks that free streaming (and Landau damping from g_0) is pure
phase mixing, the per-mode relations between the field-aligned and k_z-based fluxes and splits through GANDALF's
diagnostics, and the Gaussian-ensemble results: the misclassified weight alpha/pi and the backward share against
Monte Carlo sums, the late-time limit 1/2 of the share, and the cos^2(alpha) law and the closed form of the true net
flux from the one-dimensional ensemble integrals (../review/dynamics/c04_tilt_landau.py checks these two by direct 2D
quadrature).  It supports Section 5 and Appendix E (the k_z-based decomposition under field-line tilt).

Model: linear streaming along a uniformly tilted static field, grad_par = d_z + b0 d_x (no flow, collisionless).
Every Fourier mode evolves with k_par = k_z + b0 k_x:  g(k, t) = exp(-i s k_par t A) g(k, 0).
True (field-aligned) flux through m:   Pi_m(k)   = -2 s k_par p_m A_{m,m+1} Im(g_m^* g_{m+1});
k_z-based flux (GANDALF hermite_flux): Pi^z_m(k) = (k_z / k_par) Pi_m(k)                     (exact, per mode;
hermite_flux itself omits the factor s p_m, which is 1 in T2).
Phase-mixing split  g~_m = (i sgn kappa)^m g_m,  g^{+-}_m = (g~_m +- g~_{m+1})/2,  kappa = k_par or k_z:
  |g^+|^2 - |g^-|^2 = -sgn(kappa) Im(g_m^* g_{m+1}),  and the k_z split swaps + and - on modes with k_z k_par < 0.

Exact results checked here
 T1 free streaming from g_0 (Lambda -> infinity): |g_m|^2 = |c|^2 e^-lam lam^m/m!, lam = (s k_par t)^2/2, and
    Pi_m = |c|^2 (2 lam/t) e^-lam lam^m/m! >= 0 for every m, t (pure phase mixing); truncated GANDALF ladder agrees
    before recurrence.  Same positivity checked for Lambda = sqrt5 (Landau) with the GANDALF ladder.
 T2 per-mode identities above, through krmhd.diagnostics.hermite_flux / phase_unmixing_energy on an exact solution.
 T3 Gaussian spectrum S(k) ~ exp(-k_z^2/2 sz^2 - k_x^2/2 sx^2), tan(alpha) = b0 sx / sz:
    (a) the k_z-reported net flux is exactly cos^2(alpha) x the true net flux, for every m and t;
    (b) the spectral weight with k_z k_par < 0 is alpha/pi;
    (c) the k_z-reported backward flux (true backward flux = 0) is
        B_m(t) = int dy phi(y) [cos a sin a phi(|y| cot a)/|y| - cos^2 a Phi(-|y| cot a)] Pi_m(k_par = sigma_par y, t),
        with Pi_m(k_par, t) the free-streaming flux; its share B/(F + B) tends to 1/2 at late times (fixed m);
    (b) and (c) are checked against Monte Carlo sums over sampled modes; (a) holds identically in the
    one-dimensional integrals used here (c04 in ../review/dynamics checks it by 2D quadrature).
 T4 exact true net flux for the Gaussian ensemble: t Pi_m = 2 (tau^2/2)^{m+1} (2m+1)!! / (m! (1 + tau^2)^{m+3/2}),
    tau = s sigma_par t.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import mpmath as mp  # noqa: E402
from scipy.linalg import expm  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from krmhd.diagnostics import hermite_flux, phase_mixing_energy, phase_unmixing_energy  # noqa: E402
from dcore import Box, Checks, SQRT5, ladder, make_state, pweights  # noqa: E402

chk = Checks()
s = 1.0
BIG = 1e300                      # Lambda -> infinity (free streaming), GANDALF's ladder becomes symmetric


def pi_free(kpar: float, t: float, m: int) -> float:
    """Free-streaming flux through m per unit |g_0(k,0)|^2."""
    lam = 0.5 * (s * kpar * t) ** 2
    return float((2 * lam / t) * mp.e ** (-lam) * mp.mpf(lam) ** m / mp.factorial(m))


# ---------------------------------------------------------------------------------- T1
print("T1  free streaming is pure phase mixing; truncated GANDALF ladder vs the Poisson formula")
M = 256
Af = ladder(M, BIG)
p_f = pweights(M, BIG)
worst_formula, worst_neg = 0.0, 0.0
for kpar in (-7.3, 2.1, 11.0):
    for t in (0.1, 0.5, 1.0, 1.5):
        lam = 0.5 * (s * kpar * t) ** 2
        if lam > 0.3 * M:
            continue
        g = expm(-1j * s * kpar * t * Af)[:, 0]
        Wn = np.abs(g) ** 2
        pois = np.array([float(mp.e ** (-lam) * mp.mpf(lam) ** m / mp.factorial(m)) for m in range(M + 1)])
        worst_formula = max(worst_formula, np.abs(Wn - pois)[: M // 2].max())
        Pi = np.array([-2 * s * kpar * p_f[m] * Af[m, m + 1] * np.imag(np.conj(g[m]) * g[m + 1]) for m in range(M)])
        Pi_pred = np.array([pi_free(kpar, t, m) for m in range(M)])
        worst_formula = max(worst_formula, np.abs(Pi - Pi_pred)[: M // 2].max())
        worst_neg = min(worst_neg, Pi[: M // 2].min())
chk("T1 |g_m|^2 Poisson and Pi_m = (2 lam/t) e^-lam lam^m/m! (M = 256 ladder, before recurrence)",
    worst_formula < 1e-12, f"max abs diff {worst_formula:.1e}")
chk("T1 free-streaming flux non-negative at every m (pure phase mixing)", worst_neg > -1e-14, f"min {worst_neg:.1e}")
# Landau case Lambda = sqrt5 with g_0 initial: is the true flux still non-negative before recurrence?
A5 = ladder(M, SQRT5)
p5 = pweights(M, SQRT5)
neg5 = 0.0
for kpar in (1.0, 5.0):
    for t in np.linspace(0.05, 3.0, 40):
        if 0.5 * (kpar * t) ** 2 > 0.3 * M:
            continue
        g = expm(-1j * s * kpar * t * A5)[:, 0]
        Pi = np.array([-2 * s * kpar * p5[m] * A5[m, m + 1] * np.imag(np.conj(g[m]) * g[m + 1]) for m in range(M)])
        neg5 = min(neg5, Pi[: M // 2].min() / np.abs(Pi).max())
print(f"    Lambda = sqrt5, g_0 initial: min_m,t Pi_m / max|Pi| = {neg5:.2e} (k_par = 1, 5; t <= 3)")
chk("T1 Landau case (Lambda = sqrt5) from g_0 is also pure phase mixing (flux >= 0)", neg5 > -1e-10, f"{neg5:.1e}")

# ---------------------------------------------------------------------------------- T2
print("T2  per-mode identities through GANDALF's diagnostics on an exact tilted-field solution")
N, Mb = 32, 256
box = Box(N)
rng = np.random.default_rng(3)
sz, sx = 2 * np.pi * 1.5, 2 * np.pi * 3.0
b0 = 0.5                                             # tan(alpha) = b0 sx / sz = 1, alpha = 45 deg
A = ladder(Mb, BIG)
pw = pweights(Mb, BIG)
KX, KZ = box.KX * np.ones_like(box.KZ) * np.ones_like(box.KY), box.KZ * np.ones_like(box.KX) * np.ones_like(box.KY)
KPAR = KZ + b0 * KX
spar = np.sqrt(sz ** 2 + b0 ** 2 * sx ** 2)
t_late = 6.0 / (s * spar)                            # tau = s sigma_par t = 6
kcut = np.sqrt(2 * Mb / 3) / (s * t_late)            # retained modes have lam(t_late) < M/3: no recurrence
S_k = np.exp(-KZ ** 2 / (2 * sz ** 2) - KX ** 2 / (2 * sx ** 2) - box.KY ** 2 / (2 * sx ** 2)) * box.mask
S_k = S_k * (np.abs(KPAR) < kcut)
c = (rng.standard_normal(S_k.shape) + 1j * rng.standard_normal(S_k.shape)) * np.sqrt(S_k)
c[0, 0, 0] = 0
evals, R = np.linalg.eigh(A)
coef = R.T[:, 0]                                       # e_0 in the eigenbasis (A symmetric for Lambda -> infinity)
opp = (KZ * KPAR < 0) & box.mask & (S_k > 0)
same = (KZ * KPAR > 0) & box.mask & (S_k > 0)
wfrac = (box.w * S_k)[opp].sum() / (box.w * S_k)[box.mask].sum()
print(f"    N = {N}, M = {Mb}, b0 = {b0} (alpha = 45 deg), spectral cut |k_par| < {kcut:.1f}; "
      f"weight on modes with k_z k_par < 0: {wfrac:.3f}")
shares = {}
for t in (0.25 * t_late, t_late):
    ph = np.exp(-1j * s * KPAR[..., None] * t * evals[None, None, None, :])
    g = (ph * coef) @ R.T * c[..., None]               # g(k,t) = R exp(-i s k_par t diag(u)) R^T e_0 c_k
    st = make_state(box, np.zeros_like(c), np.zeros_like(c), g, BIG, 1.0)
    flux_gandalf = np.asarray(hermite_flux(st))        # -k_z sqrt(2(m+1)) Im[g_{m+1} g_m^*]
    true = np.zeros_like(flux_gandalf)
    for m in range(Mb):
        true[..., m] = -2 * s * KPAR * pw[m] * A[m, m + 1] * np.imag(np.conj(g[..., m]) * g[..., m + 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        pred = np.where(np.abs(KPAR[..., None]) > 0, (KZ / KPAR)[..., None] * true, 0.0)
    e_id = np.abs(flux_gandalf - pred).max() / np.abs(flux_gandalf).max()
    tau = s * spar * t
    chk(f"T2 tau={tau:.1f}: hermite_flux = (k_z/k_par) x field-aligned flux, every mode and m", e_id < 1e-12, f"{e_id:.1e}")
    chk(f"T2 tau={tau:.1f}: field-aligned flux >= 0 on every mode (pure phase mixing)",
        true.min() > -1e-13 * np.abs(true).max(), f"min/max = {true.min() / np.abs(true).max():.1e}")
    m = 3
    for kap, name in ((KPAR, "k_par"), (KZ, "k_z")):
        sg = np.sign(kap)
        gt_m = (1j * sg) ** m * g[..., m]
        gt_m1 = (1j * sg) ** (m + 1) * g[..., m + 1]
        if name == "k_par":
            gp_true, gm_true = np.abs(gt_m + gt_m1) ** 2 / 4, np.abs(gt_m - gt_m1) ** 2 / 4
            diff_id = np.abs((gp_true - gm_true) + sg * np.imag(np.conj(g[..., m]) * g[..., m + 1])).max()
        else:
            gp_z, gm_z = np.abs(gt_m + gt_m1) ** 2 / 4, np.abs(gt_m - gt_m1) ** 2 / 4
    e_swap = max(np.abs(gp_z - gm_true)[opp].max(), np.abs(gm_z - gp_true)[opp].max(),
                 np.abs(gp_z - gp_true)[same].max(), np.abs(gm_z - gm_true)[same].max())
    scale = max(np.abs(gp_true).max(), 1e-300)
    chk(f"T2 tau={tau:.1f}: |g+|^2 - |g-|^2 = -sgn(kappa) Im(g_m^* g_m+1)", diff_id < 1e-14 * max(1, scale), f"{diff_id:.1e}")
    chk(f"T2 tau={tau:.1f}: k_z split swaps + and - exactly on modes with k_z k_par < 0", e_swap < 1e-14 * max(1, scale),
        f"{e_swap:.1e}")
    line = []
    for mm in (1, 2, 4, 8, 16):
        unmix = phase_unmixing_energy(st, mm)
        mix = phase_mixing_energy(st, mm)
        shares[(round(tau, 1), mm)] = unmix / (mix + unmix)
        tot = np.abs(true[..., mm] * box.w).sum()
        bz = -np.where(flux_gandalf[..., mm] < 0, flux_gandalf[..., mm], 0) * box.w
        fz = np.where(flux_gandalf[..., mm] > 0, flux_gandalf[..., mm], 0) * box.w
        line.append(f"m={mm}: unmixing energy share {unmix / (mix + unmix):.3f}, backward flux share "
                    f"{bz.sum() / (bz.sum() + fz.sum()):.3f}")
    print(f"    tau = {tau:.1f} (t = {t:.4f}); true backward flux = 0; GANDALF k_z diagnostics: " + "; ".join(line))
chk("T2 at tau = 6 GANDALF reports a large unmixing share at low m although the dynamics is pure phase mixing",
    max(shares[(6.0, 1)], shares[(6.0, 2)]) > 0.1, f"m=1: {shares[(6.0, 1)]:.3f}, m=2: {shares[(6.0, 2)]:.3f}")

# ---------------------------------------------------------------------------------- T3, T4
print("T3/T4  Gaussian ensemble: exact formulas vs Monte Carlo over sampled modes")
mp.mp.dps = 30


def ens_formulas(alpha: float, tau: float, m: int) -> dict:
    """Exact ensemble fluxes (units: sigma_par = 1, s = 1, t = tau; per unit spectral weight)."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    cot = ca / sa
    tt = tau
    phi = lambda y: mp.e ** (-y * y / 2) / mp.sqrt(2 * mp.pi)  # noqa: E731
    Phi = lambda y: (1 + mp.erf(y / mp.sqrt(2))) / 2  # noqa: E731

    def flux(y):
        lam = (tt * y) ** 2 / 2
        return (2 * lam / tt) * mp.e ** (-lam) * lam ** m / mp.factorial(m)

    true = 2 * mp.quad(lambda y: phi(y) * flux(y), [0, mp.inf])
    B = 2 * mp.quad(lambda y: phi(y) * (ca * sa * phi(y * cot) / y - ca ** 2 * Phi(-y * cot)) * flux(y), [0, mp.inf])
    F = 2 * mp.quad(lambda y: phi(y) * (ca * sa * phi(y * cot) / y + ca ** 2 * Phi(y * cot)) * flux(y), [0, mp.inf])
    closed = 2 * (mp.mpf(tt) ** 2 / 2) ** (m + 1) * mp.fac2(2 * m + 1) / (mp.factorial(m) * (1 + mp.mpf(tt) ** 2) ** (m + 1.5)) / tt
    return dict(true=float(true), B=float(B), F=float(F), closed=float(closed))


def ens_mc(alpha: float, tau: float, m: int, n: int = 2_000_000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    sz_ = 1.0
    sx_ = 1.0
    b = math.tan(alpha) * sz_ / sx_
    kz = rng.standard_normal(n) * sz_
    kx = rng.standard_normal(n) * sx_
    kp = kz + b * kx
    sp_ = math.sqrt(sz_ ** 2 + b ** 2 * sx_ ** 2)
    tt = tau / sp_                     # tau = s sigma_par t
    lam = (kp * tt) ** 2 / 2
    logf = np.log(2 * lam / tt) - lam + m * np.log(lam) - math.lgamma(m + 1)
    fl = np.exp(logf)
    rep = kz / kp * fl
    return dict(true=fl.mean() * sp_, F=np.where(rep > 0, rep, 0).mean() * sp_, B=-np.where(rep < 0, rep, 0).mean() * sp_,
                net=rep.mean() * sp_, opp=np.mean(kz * kp < 0))


rows = []
for alpha_deg in (14.04, 26.57, 45.0, 63.43):
    alpha = math.radians(alpha_deg)
    for tau in (0.5, 2.0, 8.0):
        for m in (1, 4, 16):
            ex = ens_formulas(alpha, tau, m)
            mc = ens_mc(alpha, tau, m)
            rows.append((alpha_deg, tau, m, ex, mc))
print("    alpha  tau   m | true net  | k_z: net/true  cos^2a | backward share B/(F+B)  [MC] | (F+B)/true | opp. weight [alpha/pi]")
worst_c2, worst_B, worst_cl, worst_opp = 0.0, 0.0, 0.0, 0.0
for alpha_deg, tau, m, ex, mc in rows:
    a = math.radians(alpha_deg)
    c2 = (ex["F"] - ex["B"]) / ex["true"]
    worst_c2 = max(worst_c2, abs(c2 - math.cos(a) ** 2))
    worst_cl = max(worst_cl, abs(ex["closed"] - ex["true"]) / ex["true"])
    share = ex["B"] / (ex["F"] + ex["B"])
    share_mc = mc["B"] / (mc["F"] + mc["B"])
    worst_B = max(worst_B, abs(share - share_mc))
    worst_opp = max(worst_opp, abs(mc["opp"] - a / math.pi))
    print(f"    {alpha_deg:5.1f} {tau:4.1f} {m:3d} | {ex['true']:.3e} | {c2:.4f}  {math.cos(a) ** 2:.4f} | "
          f"{share:.4f} [{share_mc:.4f}] | {(ex['F'] + ex['B']) / ex['true']:7.3f} | {mc['opp']:.4f} [{a / math.pi:.4f}]")
chk("T3a k_z-reported net flux = cos^2(alpha) x true net flux", worst_c2 < 1e-12, f"{worst_c2:.1e}")
chk("T4 closed form of the true net flux", worst_cl < 1e-12, f"{worst_cl:.1e}")
chk("T3b weight on modes with k_z k_par < 0 = alpha/pi (Monte Carlo, 2e6 modes)", worst_opp < 2e-3, f"{worst_opp:.1e}")
chk("T3c backward share formula agrees with Monte Carlo", worst_B < 5e-3, f"{worst_B:.1e}")
# late-time limit of the backward share
lim = [ens_formulas(math.radians(45.0), tau, 4) for tau in (50.0, 200.0, 800.0)]
shares = [x["B"] / (x["F"] + x["B"]) for x in lim]
print(f"    alpha = 45 deg, m = 4: backward share at tau = 50, 200, 800: {', '.join(f'{x:.4f}' for x in shares)}")
chk("T3c backward share -> 1/2 at late times", abs(shares[-1] - 0.5) < abs(shares[0] - 0.5) and abs(shares[-1] - 0.5) < 0.01,
    f"{shares[-1]:.4f}")
chk.finish()
