"""c08_spectrum: independent checks of the phase velocities of the truncated hierarchy: u_max for both closures by
root-finding without an eigenvalue solver, the size of the Lam-shift of u_max in 60-digit arithmetic, the three-term
u_max asymptotics against independent roots up to M = 2000, the characteristic-polynomial identity and the sum rules
in 50-digit arithmetic, and the spectral-shift law out to M = 8192. Supports Section 8 and Appendix C of the paper.

Independent routes:

a. u_max at Lam = inf, zero closure, from scipy.special.roots_hermite (Golub-Welsch + Newton, asymptotic method for
   large n) and from mpmath root-finding of H_{M+1}; copy closure from mpmath root-finding of psi_{M+1}(u) - psi_{M-1}(u)
   (no eigenvalue solver), M = 16, 64, 128. Compared with the u_max table of t2_eigenvalues.py and with GANDALF's
   matrix.
b. Size of the Lam-shift of u_max: mpmath 60-digit eigenvalues at M = 12, 20, 32, 48 compared with -lambda_max x_max/Lam,
   and lambda_max compared with exp(-(2M+3)) and with exp(-x_max^2).
c. The three-term u_max asymptotic formulas (zero and copy closures) against roots_hermite / mpmath roots up to
   M = 2000 (fit-free residual scaling).
d. The identity det(z - A) = det(z - J) + det(z - J^(2))/(2 Lam) in 50-digit arithmetic at M = 30, 31 for complex z and both
   closures; sum rules from 50-digit eigenvalues.
e. Spectral-shift law continued to M = 8192 (does the deviation keep falling like M^-1/2?).
"""
from __future__ import annotations

import numpy as np
import mpmath as mp
from scipy.special import roots_hermite, ai_zeros, wofz
from scipy.linalg import eigh_tridiagonal

from cc import Checks, hdr, SQRT5

C = Checks("c08_spectrum")
from krmhd.hermite import compute_streaming_matrix

# reference values: the u_max table printed by t2_eigenvalues.py
table = {(16, "zero"): 4.87135, (64, "zero"): 10.61602, (128, "zero"): 15.35505,
         (16, "symmetric"): 5.30215, (64, "symmetric"): 11.00116, (128, "symmetric"): 15.71228}

hdr("a. u_max without an eigenvalue solver")
mp.mp.dps = 40


def psi_list(M, t):
    """Orthonormal Hermite functions psi_0..psi_{M+1} at t (mpmath, stable forward recurrence)."""
    t = mp.mpf(t)
    pm1, p0 = mp.mpf(0), mp.pi ** mp.mpf(-0.25) * mp.exp(-t * t / 2)
    vals = [p0]
    for m in range(0, M + 1):
        p1 = mp.sqrt(mp.mpf(2) / (m + 1)) * t * p0 - mp.sqrt(mp.mpf(m) / (m + 1)) * pm1
        pm1, p0 = p0, p1
        vals.append(p0)
    return vals


okA = True
for M in (16, 64, 128):
    x, _ = roots_hermite(M + 1)
    xr = float(x.max())
    fz = lambda t: psi_list(M, t)[M + 1]
    fc = lambda t: psi_list(M, t)[M + 1] - psi_list(M, t)[M - 1]
    xm = mp.findroot(fz, (xr - 0.05, xr + 0.05), solver="anderson")
    tc = table[(M, "symmetric")]
    xc = mp.findroot(fc, (tc - 0.05, tc + 0.05), solver="anderson")
    # make sure each is the LARGEST root: no sign change beyond it
    largest = True
    for f_, r_ in ((fz, xm), (fc, xc)):
        tt = np.linspace(float(r_) + 1e-4, float(r_) + 4, 300)
        vals = [float(f_(t)) for t in tt]
        largest = largest and all(np.sign(v) == np.sign(vals[0]) for v in vals)
    ug = {clo: np.abs(np.linalg.eigvals(np.asarray(compute_streaming_matrix(M, SQRT5, clo)))).max() for clo in ("zero", "symmetric")}
    d1 = abs(float(xm) - table[(M, "zero")])
    d2 = abs(float(xc) - table[(M, "symmetric")])
    okA = okA and d1 < 6e-6 and d2 < 6e-6 and largest and abs(ug["zero"] - float(xm)) < 1e-9 and abs(ug["symmetric"] - float(xc)) < 1e-9
    print(f"   M={M}: zero closure: largest zero of psi_(M+1) = {mp.nstr(xm, 12)} (scipy roots_hermite {xr:.10f}; table "
          f"{table[(M, 'zero')]}; GANDALF Lam=sqrt5 {ug['zero']:.10f}); copy closure largest root of psi_(M+1) - psi_(M-1) "
          f"= {mp.nstr(xc, 12)} (table {table[(M, 'symmetric')]}; GANDALF {ug['symmetric']:.10f}); largest roots: {largest}")
C("c08.a table values (both closures) reproduced by root-finding, and equal GANDALF's matrix at Lam = sqrt5", okA)

hdr("b. Size of the Lam-shift of u_max (60 digits)")
mp.mp.dps = 60
rows = []
for M in (12, 20, 32, 48):
    def mat(LL):
        A = mp.matrix(M + 1, M + 1)
        for m in range(M):
            A[m, m + 1] = mp.sqrt(mp.mpf(m + 1) / 2); A[m + 1, m] = mp.sqrt(mp.mpf(m + 1) / 2)
        if LL is not None:
            A[1, 0] = (1 - 1 / LL) / mp.sqrt(2)
        return A
    eJ = sorted([mp.re(e) for e in mp.eig(mat(None), left=False, right=False)])
    eA = sorted([mp.re(e) for e in mp.eig(mat(mp.sqrt(5)), left=False, right=False)])
    xm = eJ[-1]
    p = [mp.mpf(1), mp.sqrt(2) * xm]
    for m in range(1, M):
        p.append((xm * p[m] - mp.sqrt(mp.mpf(m) / 2) * p[m - 1]) / mp.sqrt(mp.mpf(m + 1) / 2))
    lam_max = 1 / mp.fsum(q * q for q in p)
    shift = eA[-1] - xm
    first = -lam_max * xm / mp.sqrt(5)
    rows.append((M, shift, first, lam_max, xm))
    print(f"   M={M}: u_max(sqrt5) - x_max = {mp.nstr(shift, 6)}; -lambda_max x_max/Lam = {mp.nstr(first, 6)}; "
          f"ln lambda_max = {mp.nstr(mp.log(lam_max), 6)}; -(2M+3) = {-(2 * M + 3)}; -x_max^2 = {mp.nstr(-xm ** 2, 6)}; "
          f"ln lambda_max + x_max^2 = {mp.nstr(mp.log(lam_max) + xm ** 2, 5)}")
okB1 = all(abs(r[1] / r[2] - 1) < 0.02 for r in rows)
# is lambda_max ~ exp(-(2M+3))? Compare ln(lambda_max) + (2M+3) (should stay O(1) if that scaling held) with
# ln(lambda_max) + x_max^2
dev_2m3 = [float(mp.log(r[3]) + (2 * r[0] + 3)) for r in rows]
dev_xmax = [float(mp.log(r[3]) + r[4] ** 2) for r in rows]
print(f"   ln(lambda_max) + (2M+3) = {np.round(dev_2m3, 2).tolist()} (grows like 2^(2/3)|a_1| nu^(1/3) = "
      f"{np.round([2 ** (2 / 3) * 2.33811 * (2 * r[0] + 3) ** (1 / 3) for r in rows], 2).tolist()}); "
      f"ln(lambda_max) + x_max^2 = {np.round(dev_xmax, 2).tolist()} (slowly varying)")
C("c08.b the shift is -lambda_max x_max/Lam to O(1/(Lam x_max^2)) (60 digits)", okB1)
C("c08.b' lambda_max is NOT ~ exp(-(2M+3)): ln(lambda_max) + (2M+3) grows like 3.71 nu^(1/3) (lambda_max ~ "
  "exp(-x_max^2) times an algebraic factor)", dev_2m3[-1] > 15 and
  abs(dev_xmax[-1] - dev_xmax[0]) < 0.5 * abs(dev_2m3[-1] - dev_2m3[0]))

hdr("c. u_max asymptotics (both closures) against independent roots up to M = 2000")
a1 = ai_zeros(1)[0][0]
a1p = ai_zeros(1)[1][0]
okC = True
for clo in ("zero", "symmetric"):
    res = []
    for M in (100, 250, 500, 1000, 2000):
        if clo == "zero":
            x, _ = roots_hermite(M + 1)
            um = float(x.max())
            nu = 2 * M + 3
            pred = np.sqrt(nu) + 2 ** (-1 / 3) * a1 * nu ** (-1 / 6) - 2 ** (-2 / 3) * a1 ** 2 / 10 * nu ** (-5 / 6)
        else:
            nu = 2 * M + 1
            pred = np.sqrt(nu) + 2 ** (-1 / 3) * a1p * nu ** (-1 / 6) - 2 ** (-5 / 3) * (4 / (5 * a1p) + a1p ** 2 / 5) * nu ** (-5 / 6)
            # psi_M' + kappa psi_M = 0 with kappa = u/(sqrt M + sqrt(M+1))^2, solved by mpmath (log-scaled Hermite functions)
            mp.mp.dps = 30
            def f(t):
                v = psi_list(M, t)
                return v[M + 1] - v[M - 1]
            um = float(mp.findroot(f, (pred - 0.02, pred + 0.02), solver="anderson"))
        res.append((M, (um - pred) * nu ** 1.5))
    print(f"   {clo} closure: (u_max - three-term formula) nu^(3/2) at M = " + ", ".join(f"{M}: {r:+.4f}" for M, r in res))
    rr = np.array([r for _, r in res])
    okC = okC and np.abs(np.diff(rr)).max() < 0.02 and np.abs(rr).max() < 0.25
C("c08.c the three-term u_max formulas (both closures) leave an O(nu^-3/2) remainder (independent roots to M = 2000)", okC)

hdr("d. Characteristic-polynomial identity and sum rules, 50 digits")
mp.mp.dps = 50
okD = True
for M in (30, 31):
    for clo in ("zero", "symmetric"):
        def mat(LL):
            A = mp.matrix(M + 1, M + 1)
            for m in range(M):
                A[m, m + 1] = mp.sqrt(mp.mpf(m + 1) / 2); A[m + 1, m] = mp.sqrt(mp.mpf(m + 1) / 2)
            if LL is not None:
                A[1, 0] = (1 - 1 / LL) / mp.sqrt(2)
            if clo == "symmetric":
                A[M, M - 1] += mp.sqrt(mp.mpf(M + 1) / 2)
            return A
        Lm = -mp.sqrt(5)
        A, J = mat(Lm), mat(None)
        J2 = mp.matrix(M - 1, M - 1)
        for i in range(M - 1):
            for j in range(M - 1):
                J2[i, j] = J[i + 2, j + 2]
        worst = mp.mpf(0)
        for z in (mp.mpc(0.3, 0.5), mp.mpc(-2.1, 0.01), mp.mpc(4.0, -1.0)):
            lhs = mp.det(z * mp.eye(M + 1) - A)
            rhs = mp.det(z * mp.eye(M + 1) - J) + mp.det(z * mp.eye(M - 1) - J2) / (2 * Lm)
            worst = max(worst, abs(lhs - rhs) / abs(lhs))
        ev = mp.eig(A, left=False, right=False)
        s1 = abs(mp.fsum(ev))
        s2 = mp.fsum(e * e for e in ev) - (mp.mpf(M * (M + 1)) / 2 - 1 / Lm + (mp.sqrt(M * (M + 1)) if clo == "symmetric" else 0))
        okD = okD and worst < mp.mpf(10) ** -40 and s1 < mp.mpf(10) ** -35 and abs(s2) < mp.mpf(10) ** -35
        print(f"   M={M} {clo:9s} Lam=-sqrt5: max rel. |det identity| = {mp.nstr(worst, 3)}; |sum u| = {mp.nstr(s1, 3)}; "
              f"sum u^2 - rule = {mp.nstr(abs(s2), 3)}")
C("c08.d the determinant identity and the sum rules hold to 50 digits at M = 30, 31, both closures", okD)

hdr("e. Spectral-shift law out to M = 8192")


def eigs(M, Lam):
    p0 = 1 - (0 if np.isinf(Lam) else 1 / Lam)
    off = np.sqrt(np.arange(1, M + 1) / 2.0)
    off[0] = np.sqrt(p0 / 2.0)
    return eigh_tridiagonal(np.zeros(M + 1), off, eigvals_only=True)


okE = True
for Lam in (SQRT5, -SQRT5):
    devs = []
    for M in (512, 2048, 8192):
        x = eigs(M, np.inf)
        uu = eigs(M, Lam)
        D = np.gradient(x)
        Z = 1j * np.sqrt(np.pi) * wofz(x)
        epsr = 1 - (1 + x * Z) / Lam
        sel = np.abs(x) < 3
        devs.append(np.abs((uu - x)[sel] / D[sel] - np.angle(epsr[sel]) / np.pi).max())
    rate = np.log(devs[0] / devs[-1]) / np.log(8192 / 512)
    print(f"   Lam={Lam:+.4f}: max deviation from the shift law over |x| < 3 at M = 512, 2048, 8192: "
          + ", ".join(f"{d:.2e}" for d in devs) + f"; fitted rate M^-{rate:.3f}")
    okE = okE and devs[-1] < devs[0] / 3
C("c08.e spectral-shift law: deviation keeps falling (about M^-1/2) to M = 8192", okE)
C.done()
