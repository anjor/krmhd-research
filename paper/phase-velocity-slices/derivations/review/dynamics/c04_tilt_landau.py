"""c04: the field-line-tilt results of d06, re-derived by routes d06 does not use.  With a large-M Hermite ladder
(M = 5000) the script checks that the field-aligned flux from g_0 is non-negative for Lambda = sqrt5 and for free
streaming (and shows that it is not for Lambda = -sqrt5 or for g_1 data), and it recomputes the Gaussian-ensemble
results by direct 2D quadrature: the cos^2(alpha) law (also for a Landau-plus-hypercollision flux), the misclassified
weight alpha/pi, the one-dimensional backward-flux formula and its late-time limit 1/2, and the closed form of the true
net flux.  It supports Section 5 and Appendix E (the k_z-based decomposition under field-line tilt).

L1 Landau case: is the field-aligned Hermite flux from g_0 non-negative at every m and t for Lambda = sqrt5 (d06
   checked s k t <= 12.4 with M = 256)?  Here M = 5000 (GANDALF's ladder, eigen-decomposition of the symmetrised
   tridiagonal matrix), s k t up to 60, m < 2600 (before the truncation is felt).  Also Lambda = -sqrt5 and g_1 data,
   which the statement does not cover, for comparison.  (M = 5000, m < 2600: the free energy has not reached m = 2600.)
Q  Gaussian ensemble S(k) ~ exp(-k_z^2/2 - k_x^2/2) (sigma_z = sigma_x = 1, tan(alpha) = b0): direct 2D quadrature in
   (k_par, k_z) coordinates (d06 uses 1D formulas in which k_z was integrated analytically) of
   Q1 the reported net flux / true net flux = cos^2(alpha), for the free-streaming flux and for an arbitrary per-mode
      flux (Landau Lambda = sqrt5 with hypercollisions, M = 64), since the law is stated for any m, t, Lambda and
      collision operator;
   Q2 misclassified weight alpha/pi;
   Q3 the one-dimensional backward-flux formula of d06 (T3c) and the late-time share -> 1/2;
   Q4 the closed form of the true net flux (d06, T4).
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from scipy.linalg import eigh_tridiagonal, expm  # noqa: E402
from scipy.special import erf, gammaln  # noqa: E402

from c00_core import Checks, SQRT5, ladder, pw  # noqa: E402

chk = Checks()

# ------------------------------------------------------------------------------------------------ L1
print("L1  field-aligned flux from g_0 (or g_1), single mode, s k = 1, large-M GANDALF ladder")


def evolve_setup(M, Lam):
    A = ladder(M, Lam)
    p = pw(M, Lam)
    ph = np.sqrt(p)
    d = np.zeros(M + 1)
    e = np.array([ph[m] * A[m, m + 1] / ph[m + 1] for m in range(M)])     # B = P^1/2 A P^-1/2 (symmetric)
    e_low = np.array([ph[m + 1] * A[m + 1, m] / ph[m] for m in range(M)])
    assert np.allclose(e, e_low)
    u, Q = eigh_tridiagonal(d, e)
    return A, p, ph, u, Q


def flux_curve(M, Lam, init, thetas, mmax):
    A, p, ph, u, Q = evolve_setup(M, Lam)
    c = Q.T @ (ph * init)                                       # initial data in the eigenbasis of B
    coup = np.array([2 * p[m] * A[m, m + 1] for m in range(mmax)])
    worst, tail, where = 0.0, 0.0, None
    for chunk in np.array_split(thetas, max(1, len(thetas) // 200)):
        E = np.exp(-1j * u[:, None] * chunk[None, :]) * c[:, None]
        Y = Q @ E.real + 1j * (Q @ E.imag)                       # P^1/2 g, one column per theta
        G = Y / ph[:, None]
        Pi = -coup[:, None] * np.imag(np.conj(G[:mmax]) * G[1:mmax + 1])    # per unit k: Pi_m / (s k)
        rel = Pi / np.abs(Pi).max(axis=0)[None, :]
        if rel.min() < worst:
            worst = float(rel.min())
            im, it = np.unravel_index(np.argmin(rel), rel.shape)
            where = (int(im), float(chunk[it]))
        tail = max(tail, float((np.sum(p[mmax:, None] * np.abs(G[mmax:]) ** 2, axis=0)
                                / np.sum(p[:, None] * np.abs(G) ** 2, axis=0)).max()))
    return worst, tail, where


thetas = np.linspace(0.02, 60.0, 1200)
M, MMAX = 5000, 2600
e0 = np.zeros(M + 1)
e0[0] = 1.0
e1 = np.zeros(M + 1)
e1[1] = 1.0
rows = {}
for lab, Lam, init in (("Lambda=+sqrt5, g_0 data", SQRT5, e0), ("Lambda=inf (free streaming), g_0 data", 1e300, e0),
                       ("Lambda=-sqrt5, g_0 data", -SQRT5, e0), ("Lambda=+sqrt5, g_1 data", SQRT5, e1)):
    w, tl, wh = flux_curve(M, Lam, init, thetas, MMAX)
    rows[lab] = (w, tl)
    print(f"    {lab:40s}: min over m < {MMAX}, s k t <= 60 of Pi_m / max_m|Pi_m| = {w:+.2e}"
          + (f" (at m = {wh[0]}, s k t = {wh[1]:.2f})" if wh and w < -1e-12 else "")
          + f";  free energy beyond m = {MMAX}: {tl:.1e}")
chk("L1 Lambda = sqrt5 from g_0: flux >= 0 at every m for s k t <= 60 (M = 5000, before the truncation is felt)",
    rows["Lambda=+sqrt5, g_0 data"][0] > -1e-10 and rows["Lambda=+sqrt5, g_0 data"][1] < 1e-12,
    f"min {rows['Lambda=+sqrt5, g_0 data'][0]:+.1e}")
chk("L1 free streaming from g_0: flux >= 0 (Poisson)", rows["Lambda=inf (free streaming), g_0 data"][0] > -1e-10,
    f"min {rows['Lambda=inf (free streaming), g_0 data'][0]:+.1e}")

# ------------------------------------------------------------------------------------------------ Q
print("Q   Gaussian ensemble by 2D quadrature over (k_par, k_z); sigma_z = sigma_x = 1, b0 = tan(alpha), s = 1")


def pi_free(kp, t, m):
    lam = 0.5 * (kp * t) ** 2
    with np.errstate(divide="ignore"):
        lg = np.log(2 * lam / t) - lam + m * np.log(lam) - gammaln(m + 1)
    return np.where(lam > 0, np.exp(lg), 0.0)


def ensemble(alpha, tau, m, pi_fun=None, nk=4001, nz=1601):
    """Returns true net, reported net, reported forward, reported backward, misclassified weight."""
    b0 = math.tan(alpha)
    spar = math.sqrt(1 + b0 ** 2)
    t = tau / spar
    # k_par grid: dense near 0 (the late-time flux lives at |k_par| ~ sqrt(2m)/t)
    kc = max(8.0 * math.sqrt(2 * m + 2) / t, 1e-6)
    kmax = 9.0 * spar
    x = np.linspace(-1, 1, nk)
    kp = np.sign(x) * np.where(np.abs(x) < 0.5, 2 * np.abs(x) * kc, kc + (2 * np.abs(x) - 1) * (kmax - kc))
    kp = np.unique(np.concatenate([kp, np.linspace(-kmax, kmax, nk)]))
    kz = np.linspace(-9.0, 9.0, nz)
    KP, KZ = np.meshgrid(kp, kz, indexing="ij")
    KX = (KP - KZ) / b0
    S = np.exp(-KZ ** 2 / 2 - KX ** 2 / 2) / (2 * np.pi) / b0          # density in (k_par, k_z), unit total weight
    pi = pi_fun(kp) if pi_fun is not None else pi_free(kp, t, m)
    true_k = pi[:, None] * np.ones_like(KZ)
    with np.errstate(divide="ignore", invalid="ignore"):
        rep = np.where(np.abs(KP) > 0, KZ / KP, 0.0) * true_k
    integ = lambda F: np.trapezoid(np.trapezoid(S * F, kz, axis=1), kp)  # noqa: E731
    return dict(true=integ(true_k), net=integ(rep), F=integ(np.where(rep > 0, rep, 0)), B=-integ(np.where(rep < 0, rep, 0)),
                opp=integ(np.where(KZ * KP < 0, 1.0, 0.0)), total=integ(np.ones_like(KZ)))


def formula_1d(alpha, tau, m, which="B"):
    """The 1D backward-flux integral of d06 (and its forward analogue), evaluated by quadrature in y."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    cot = ca / sa
    y = np.linspace(1e-9, 12, 200001)
    phi = np.exp(-y ** 2 / 2) / math.sqrt(2 * math.pi)
    Phi = lambda v: 0.5 * (1 + erf(v / math.sqrt(2)))  # noqa: E731
    t = tau                                         # sigma_par = 1 units, as in the 1D formula of d06
    fl = pi_free(y, t, m)
    if which == "B":
        integ = phi * (ca * sa * np.exp(-(y * cot) ** 2 / 2) / math.sqrt(2 * math.pi) / y - ca ** 2 * Phi(-y * cot)) * fl
    else:
        integ = phi * (ca * sa * np.exp(-(y * cot) ** 2 / 2) / math.sqrt(2 * math.pi) / y + ca ** 2 * Phi(y * cot)) * fl
    return 2 * np.trapezoid(integ, y)


def closed_form(tau, m):
    lg = (m + 1) * math.log(tau ** 2 / 2) + sum(math.log(k) for k in range(1, 2 * m + 2, 2)) - math.lgamma(m + 1) \
        - (m + 1.5) * math.log(1 + tau ** 2)
    return 2 * math.exp(lg)


worst_c2, worst_opp, worst_B, worst_cl = 0.0, 0.0, 0.0, 0.0
print("    alpha   tau   m |  net/true  cos^2a |  B/(F+B) 2D   1D     | t*Pi true 2D  exact | opp weight  alpha/pi")
for adeg in (26.57, 45.0, 63.43):
    a = math.radians(adeg)
    spar = math.sqrt(1 + math.tan(a) ** 2)
    for tau in (0.5, 2.0, 8.0):
        for m in (1, 4, 16):
            q = ensemble(a, tau, m)
            c2 = q["net"] / q["true"]
            sh = q["B"] / (q["F"] + q["B"])
            # the 1D formula is per unit spectral weight in units sigma_par = 1, with Pi_m(sigma_par y, t):
            # here k_par = sigma_par y and t = tau / sigma_par, so the flux values coincide; F, B in the same units
            B1d = formula_1d(a, tau, m, "B")
            F1d = formula_1d(a, tau, m, "F")
            sh1d = B1d / (B1d + F1d)
            tpi = q["true"] * tau / spar
            cl = closed_form(tau, m)
            worst_c2 = max(worst_c2, abs(c2 - math.cos(a) ** 2))
            worst_opp = max(worst_opp, abs(q["opp"] / q["total"] - a / math.pi))
            worst_B = max(worst_B, abs(sh - sh1d))
            worst_cl = max(worst_cl, abs(tpi - cl) / cl)
            print(f"    {adeg:5.2f} {tau:5.1f} {m:3d} | {c2:.5f}  {math.cos(a) ** 2:.5f} | {sh:.5f}  {sh1d:.5f} | "
                  f"{tpi:.5e} {cl:.5e} | {q['opp'] / q['total']:.5f}  {a / math.pi:.5f}")
chk("Q1 reported net / true net = cos^2(alpha) (2D quadrature, free streaming)", worst_c2 < 1e-4, f"{worst_c2:.1e}")
# Q2 by an exact route: for sigma_z = sigma_x the spectrum is isotropic in (k_z, k_x), so the weight of the region
# k_z k_par < 0 is its angular measure / (2 pi); the region is the double wedge between the lines k_z = 0 and
# k_z = -b0 k_x, of opening arctan(b0) = alpha.  Measured on 2e6 equally spaced angles (no sampling noise):
ang = (np.arange(2_000_000) + 0.5) * 2 * np.pi / 2_000_000
worst_ang = 0.0
for adeg in (14.04, 26.57, 45.0, 63.43, 80.0):
    a = math.radians(adeg)
    kz_, kx_ = np.cos(ang), np.sin(ang)
    frac = float(np.mean(kz_ * (kz_ + math.tan(a) * kx_) < 0))
    worst_ang = max(worst_ang, abs(frac - a / math.pi))
print(f"    misclassified weight by angular measure vs alpha/pi: max difference {worst_ang:.1e} "
      f"(the trapezoid column above is O(h)-inaccurate for this discontinuous integrand and is not used)")
chk("Q2 misclassified weight = alpha/pi (exact angular measure, isotropic case)", worst_ang < 1e-6, f"{worst_ang:.1e}")
chk("Q3 backward share from 2D quadrature = 1D formula of d06", worst_B < 1e-4, f"{worst_B:.1e}")
chk("Q4 closed form of the true net flux = 2D quadrature", worst_cl < 1e-4, f"{worst_cl:.1e}")

# Q1 with an arbitrary per-mode flux: Landau (Lambda = sqrt5), hypercollisions, M = 64, from g_0
Mm, nu = 64, 2.0
A = ladder(Mm, SQRT5)
p = pw(Mm, SQRT5)
C = np.diag([nu * (m / Mm) ** 4 if m >= 2 else 0.0 for m in range(Mm + 1)])
t_l = 1.3
mm = 3
kgrid = np.linspace(-9 * math.sqrt(2), 9 * math.sqrt(2), 2001)
vals = []
for kp_ in kgrid:
    g = expm(-t_l * (1j * kp_ * A + C))[:, 0]
    vals.append(-2 * kp_ * p[mm] * A[mm, mm + 1] * np.imag(np.conj(g[mm]) * g[mm + 1]))
vals = np.array(vals)
pi_L = lambda kp_: np.interp(kp_, kgrid, vals)  # noqa: E731
q = ensemble(math.radians(45.0), 1.0, mm, pi_fun=pi_L, nk=2001, nz=1201)
neg = vals.min() / np.abs(vals).max()
print(f"    Landau + hypercollisions (M = 64, nu = {nu}, t = {t_l}, m = {mm}), alpha = 45 deg: reported net / true net = "
      f"{q['net'] / q['true']:.5f} (cos^2 = 0.5); per-mode true flux min/max = {neg:+.2e}")
chk("Q1 cos^2(alpha) also for an arbitrary per-mode flux (Landau + collisions)", abs(q["net"] / q["true"] - 0.5) < 1e-4,
    f"{q['net'] / q['true']:.5f}")

# late-time share
lates = []
for tau in (50.0, 200.0, 800.0):
    q = ensemble(math.radians(45.0), tau, 4, nk=8001, nz=2401)
    lates.append(q["B"] / (q["F"] + q["B"]))
print("    alpha = 45 deg, m = 4: backward share at tau = 50, 200, 800 (2D quadrature): " + ", ".join(f"{x:.4f}" for x in lates))
chk("Q3 late-time backward share -> 1/2 (2D quadrature)", lates[0] < lates[1] < lates[2] < 0.5 and 0.5 - lates[2] < 0.01,
    ", ".join(f"{x:.4f}" for x in lates))
chk.finish()
