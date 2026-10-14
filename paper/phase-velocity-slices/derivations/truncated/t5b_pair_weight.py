"""t5b_pair_weight: checks the sine-kernel form of the continuum Christoffel-Darboux kernel
K_m(u,u') = sum_{n<=m} phi_n(u) phi_n(u') and of the pair weight (u - u') K_m(u,u') that enters the Hermite flux.
At Lambda = inf the kernel approaches sin[k_m (u-u')]/[pi (u-u')], k_m = sqrt(2m+1), for |u|, |u'| << sqrt(2m), and the
pair weight approaches sin[k_m (u-u')]/pi. At finite Lambda the relative deviation of the kernel still falls with m,
because max K_m ~ k_m/pi grows, but the absolute deviation of the pair weight does not: it stays of order 1/pi as m
grows. Two routes: (A) the continuum three-term recurrence for r_n(u) with sqrt(rho) folded in, on a grid of u, and
(B) the right eigenvectors of GANDALF's ladder matrix (krmhd.hermite.compute_streaming_matrix, read-only) at M = 1500,
which give r_n(u_j) at the Gauss nodes. Supports Section 5 (the sine-kernel form of the flux and its failure at finite
Lambda) of the paper; t5_flux_cd.py (C1) and ../review/truncated/c05_flux.py (e, f) check the kernel at Lambda = inf.

Notation: phi_n = sqrt(p_n rho) r_n, p_0 = 1 - 1/Lambda, p_n = 1 (n >= 1), rho = F_M/|eps(u+i0)|^2,
eps(u+i0) = 1 - [1 + u Z(u+i0)]/Lambda, Z(x) = i sqrt(pi) w(x); r_0 = 1, u r_n = A_{n,n+1} r_{n+1} + A_{n,n-1} r_{n-1}.
Kernel deviation: max |K_m - sin(k_m d)/(pi d)| / max K_m on |u|, |u'| <= 1.5 (d = u - u');
pair-weight deviation: max |d K_m - sin(k_m d)/pi|, quoted in units of 1/pi.
P1  Lambda = inf: kernel deviation below 1% for m >= 64 and falling with m; pair-weight deviation falling with m
P2  Lambda = +-sqrt5: kernel deviation falls with m, but the pair-weight deviation stays between 0.5/pi and 1/pi for
    m = 64 to 1024 and falls by less than 15% over that range
P3  route B (GANDALF's ladder eigenvectors, M = 1500) reproduces K_m of route A at the nodes |u_j| <= 1.5 to round-off,
    and gives the same pair-weight deviations at the nodes
Run from the repository root:
    uv run python paper/phase-velocity-slices/derivations/truncated/t5b_pair_weight.py
Exit status 1 if any check fails.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import wofz

from tcore import Checker, section, slices, weight, SQRT5

C = Checker("t5b_pair_weight")
MS = (64, 128, 256, 1024)
CASES = ((np.inf, "inf"), (SQRT5, "+sqrt5"), (-SQRT5, "-sqrt5"))


def rho(x: np.ndarray, Lam: float) -> np.ndarray:
    """Spectral density F_M/|eps(x+i0)|^2 (= F_M at Lambda = inf)."""
    x = np.asarray(x, dtype=float)
    FM = np.exp(-x ** 2) / math.sqrt(math.pi)
    if np.isinf(Lam):
        return FM
    Z = 1j * math.sqrt(math.pi) * wofz(x.astype(complex))
    return FM / np.abs(1.0 - (1.0 + x * Z) / Lam) ** 2


def phi_route_A(u: np.ndarray, mmax: int, Lam: float) -> np.ndarray:
    """phi_n(u), n = 0..mmax, from the three-term recurrence with sqrt(rho) folded in (shape (mmax+1, len(u)))."""
    u = np.asarray(u, dtype=float)
    A10 = (1.0 - (0.0 if np.isinf(Lam) else 1.0 / Lam)) / math.sqrt(2.0)
    q = np.empty((mmax + 1, u.size))
    q[0] = np.sqrt(rho(u, Lam))
    q[1] = u * q[0] / math.sqrt(0.5)                      # A_{01} = sqrt(1/2)
    for n in range(1, mmax):
        down = A10 if n == 1 else math.sqrt(n / 2.0)       # A_{n,n-1}
        q[n + 1] = (u * q[n] - down * q[n - 1]) / math.sqrt((n + 1) / 2.0)
    return np.sqrt(weight(mmax, Lam))[:, None] * q


def deviations(K: np.ndarray, D: np.ndarray, m: int) -> tuple[float, float]:
    """(kernel deviation / max K, pair-weight deviation in units of 1/pi)."""
    kap = math.sqrt(2 * m + 1)
    off = np.abs(D) > 1e-12
    S = np.where(off, np.sin(kap * D) / (np.pi * np.where(off, D, 1.0)), kap / np.pi)
    return float(np.max(np.abs(K - S)) / np.max(K)), float(np.pi * np.max(np.abs(D * K - np.sin(kap * D) / np.pi)))


section("A. Continuum recurrence on |u|, |u'| <= 1.5")
u = np.linspace(-1.5, 1.5, 481)
D = u[:, None] - u[None, :]
devA = {}
for Lam, lab in CASES:
    P = phi_route_A(u, max(MS), Lam)
    for m in MS:
        K = P[: m + 1].T @ P[: m + 1]
        kd, pd = deviations(K, D, m)
        devA[(lab, m)] = (kd, pd)
        print(f"   Lam={lab:7s} m={m:5d}: kernel deviation / max K = {100 * kd:6.2f}%;  max K pi/k_m = "
              f"{np.max(K) * np.pi / math.sqrt(2 * m + 1):.4f};  pair-weight deviation = {pd:.3f}/pi")

kd_inf = [devA[("inf", m)][0] for m in MS]
pd_inf = [devA[("inf", m)][1] for m in MS]
C.check("P1 Lam = inf: kernel -> sine kernel (deviation < 1% for m >= 64, falling) and pair weight -> sin(k_m d)/pi",
        max(kd_inf) < 0.01 and all(np.diff(kd_inf) < 0) and all(np.diff(pd_inf) < 0),
        f"kernel {', '.join(f'{100 * v:.2f}%' for v in kd_inf)}; pair weight {', '.join(f'{v:.3f}/pi' for v in pd_inf)}"
        f" at m = {MS}")
for lab in ("+sqrt5", "-sqrt5"):
    kd = [devA[(lab, m)][0] for m in MS]
    pd = [devA[(lab, m)][1] for m in MS]
    C.check(f"P2 Lam = {lab}: relative kernel deviation falls with m, but the pair-weight deviation stays O(1/pi)",
            all(np.diff(kd) < 0) and min(pd) > 0.5 and max(pd) < 1.0 and pd[-1] / pd[0] > 0.85,
            f"kernel {', '.join(f'{100 * v:.1f}%' for v in kd)}; pair weight "
            f"{', '.join(f'{v:.3f}/pi' for v in pd)} at m = {MS}")

section("B. GANDALF's ladder eigenvectors at M = 1500 (Gauss nodes |u_j| <= 1.5)")
M = 1500
okB = True
for Lam, lab in CASES:
    uj_all, R, _, _ = slices(M, Lam, "zero")
    sel = np.abs(uj_all) <= 1.5
    uj = uj_all[sel]
    r = R[:, sel] / R[0, sel][None, :]                      # r_n(u_j), normalised so that r_0 = 1
    PhiB = np.sqrt(weight(M, Lam))[:, None] * np.sqrt(rho(uj, Lam))[None, :] * r
    PhiA = phi_route_A(uj, 256, Lam)
    Dn = uj[:, None] - uj[None, :]
    out = []
    for m in (64, 128, 256):
        KB = PhiB[: m + 1].T @ PhiB[: m + 1]
        KA = PhiA[: m + 1].T @ PhiA[: m + 1]
        diff = float(np.max(np.abs(KB - KA)) / np.max(KA))
        pdB = deviations(KB, Dn, m)[1]
        out.append(f"m={m}: |K_B - K_A|/max K = {diff:.1e}, pair weight {pdB:.3f}/pi")
        okB = okB and diff < 1e-11 and (pdB < 0.25 if np.isinf(Lam) else 0.5 < pdB < 1.0)
    print(f"   Lam={lab:7s} ({sel.sum()} nodes): " + "; ".join(out))
C.check("P3 GANDALF's ladder eigenvectors give the continuum kernel at the nodes; same pair-weight deviations", okB)

C.finish()
