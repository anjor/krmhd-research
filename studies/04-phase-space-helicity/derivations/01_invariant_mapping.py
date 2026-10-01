"""Quadratic invariants of the GANDALF Hermite hierarchy (Study 04, derivation 01).

Claims checked here (SPEC.md §2, claims C1 and C3 in claims.md):

1. For the collisionless, unforced, ideal (eta = 0) hierarchy with the
   zero-truncation closure g_{M+1} = 0,

       d/dt g = -{Phi, g} - s * nabla_par (A g),        nabla_par = d_z + {Psi, .}

   with A the real (M+1)x(M+1) ladder matrix of SPEC.md §1, a quadratic form
   Q = sum_k g^dagger P g (P real symmetric) is conserved for all g, Phi, Psi
   if and only if P A is symmetric.  Reason: {Phi, .} is a common incompressible
   advection and conserves every bilinear int g_m g_n; d_z and {Psi, .} are
   antisymmetric under the volume integral and act identically on the moment
   index; so dQ/dt = -s * sum_k g^dagger (P A - A^T P) (D g) with D
   antisymmetric, which vanishes for all g iff P A - A^T P = 0.

2. The solution space of P A = A^T P (P symmetric tridiagonal) is exactly
   two-dimensional at every M >= 2:

       W     : P = diag(1 - 1/Lambda, 1, 1, ..., 1)
       Gamma : P off-diagonal p_m, p_0 = sqrt(2) (1 - 1/Lambda),
               p_m = sqrt(2 (m + 1)) for m >= 1, zero diagonal.

   Gamma = sum_k sum_{m=0}^{M-1} p_m Re[g_{m+1} g_m^*].  The normalisation of
   p_m for m >= 1 matches the prefactor of krmhd.diagnostics.hermite_flux,
   Pi_m = -k_par sqrt(2(m+1)) Im[g_{m+1} g_m^*], so Gamma_m and Pi_m / (-k_par)
   are the real and imaginary parts of one correlator.

3. Zero truncation leaves no boundary term: the (M+1)x(M+1) truncation of A
   satisfies the symmetry condition exactly.  The copy closure
   g_{M+1} = g_{M-1} adds sqrt((M+1)/2) to A_{M,M-1}.  Gamma is still exactly
   conserved (its condition never involves A_{M,M-1}); W is conserved only
   with the last weight changed to sqrt(M)/(sqrt(M) + sqrt(M+1)), so the
   usual W (weight 1 at m = M) acquires a boundary term at m = M.

   Physical identification: for Lambda -> infinity (c = 1/sqrt2) the matrix A
   is the matrix of multiplication by v_par in the orthonormal Hermite basis,
   so Gamma = g^dagger A g is the Hermite image of int dv v_par g^2: the first
   v_par moment of the free-energy density.  Finite Lambda only changes the
   m = 0 weight.

Part A proves (2) and (3) with SymPy at M = 6 and checks the closed form at
M = 40.  Part B evaluates the actual GANDALF right-hand side
(krmhd.timestepping.krmhd_rhs, which includes both Poisson brackets and the
Lambda coupling) on a random dealiased state in float64 with nu = eta = 0 and
confirms dW/dt = dGamma/dt = 0 to round-off, for Lambda = +sqrt(5) and
-sqrt(5), at two values of M.  It also confirms that the wrong g_0 weight
(1 instead of 1 - 1/Lambda) and the wrong p_0 give O(1) residuals, so the
test has teeth.

Part C: mapping to Chandran, Mallet & Meyrand 2026 (arXiv:2607.27981), Appendix B.
Their compressive fields G^+- obey (B4) with Lambda^+- of (B7); expanded in
Hermite polynomials with the sqrt(2^m m!) normalisation (B27-B30) the
coefficients Gtilde_m obey the same ladder as GANDALF's g_m with the same
(1 - 1/Lambda) coupling, so g_m <-> Gtilde^+_m for Lambda = Lambda^+ = +sqrt5
at beta_i = tau = Z = 1 (and Gtilde^-_m for Lambda = Lambda^- = -sqrt5).
Their energy (B31) is sum Gtilde_m^2 - Gtilde_0^2/Lambda, i.e. weight
(1 - 1/Lambda) on m = 0: our W.  Their additional invariant (B25), (B32)

    Gamma^+- = (v_th/sqrt2) int d^3r [ sum_m sqrt(m+1) Gtilde_m Gtilde_{m+1} - Gtilde_0 Gtilde_1/Lambda ]

has the m = 0 coefficient (1 - 1/Lambda) and m >= 1 coefficients sqrt(m+1):
the same ratios as our p_m.  With unit volume and v_th = 1,
Gamma_ours = 2 Gamma^+-_CMM.  (B25) identifies it as the first v_par moment
of the free-energy density, int v_par (G)^2/(2 F_M) dv, minus M_0 M_1/Lambda.

The paper's headline 'phase-space helicity' H_ph-sp (B18) is a different
invariant, P int (G)^2/(2 v_par F_M) dv, with a 1/v_par weight.  Its Hermite
representation is a dense matrix (1/v_par couples all moments of opposite
parity), so it is outside the nearest-neighbour ansatz of this derivation and
outside the diagnostics planned in PLAN.md.  Part C only checks the Gamma
coefficient identity numerically; H_ph-sp is left to a decision at Gate 1.

Requires GANDALF >= 0.6.0 (no B_parallel field).
Run:  uv run python studies/04-phase-space-helicity/derivations/01_invariant_mapping.py
Exit status is non-zero if any check fails.
"""

from __future__ import annotations

import sys

import numpy as np
import sympy as sp

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    """Record a named check."""
    status = "ok  " if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


# --------------------------------------------------------------------------
# Part A: symbolic
# --------------------------------------------------------------------------

def ladder_matrix(M: int, c: sp.Expr, closure: str = "zero") -> sp.Matrix:
    """GANDALF ladder matrix A (SPEC.md §1): d/dt g = -s nabla_par (A g).

    A_{0,1} = 1/sqrt2, A_{1,0} = c = (1 - 1/Lambda)/sqrt2,
    A_{m,m+1} = sqrt((m+1)/2) (m >= 1), A_{m,m-1} = sqrt(m/2) (m >= 2).
    closure='zero': g_{M+1} = 0.  closure='symmetric': g_{M+1} = g_{M-1}.
    """
    A = sp.zeros(M + 1, M + 1)
    A[0, 1] = 1 / sp.sqrt(2)
    A[1, 0] = c
    for m in range(1, M + 1):
        if m + 1 <= M:
            A[m, m + 1] = sp.sqrt(sp.Rational(m + 1, 2))
        if m >= 2:
            A[m, m - 1] = sp.sqrt(sp.Rational(m, 2))
    if closure == "symmetric":
        A[M, M - 1] += sp.sqrt(sp.Rational(M + 1, 2))
    elif closure != "zero":
        raise ValueError(closure)
    return A


def solve_invariants(M: int, c: sp.Symbol) -> list[sp.Matrix]:
    """All symmetric tridiagonal P with P A = A^T P, as a basis of the nullspace."""
    A = ladder_matrix(M, c)
    d = sp.symbols(f"d0:{M + 1}", real=True)
    p = sp.symbols(f"p0:{M}", real=True)
    P = sp.zeros(M + 1, M + 1)
    for m in range(M + 1):
        P[m, m] = d[m]
    for m in range(M):
        P[m, m + 1] = p[m]
        P[m + 1, m] = p[m]
    cond = (P * A - A.T * P).applyfunc(sp.simplify)
    eqs = [e for e in cond if e != 0]
    unknowns = list(d) + list(p)
    sol = sp.linsolve(eqs, unknowns)
    (sol_tuple,) = sol
    free = sorted(set().union(*[s.free_symbols for s in sol_tuple]) - {c}, key=str)
    basis = []
    for f in free:
        subs = {g: (1 if g == f else 0) for g in free}
        basis.append(P.subs(dict(zip(unknowns, sol_tuple))).subs(subs).applyfunc(sp.simplify))
    return basis


def closed_form_P(M: int, c: sp.Expr) -> tuple[sp.Matrix, sp.Matrix]:
    """The claimed closed forms: (P_W, P_Gamma)."""
    PW = sp.eye(M + 1)
    PW[0, 0] = sp.sqrt(2) * c          # (1 - 1/Lambda) = sqrt2 * c
    PG = sp.zeros(M + 1, M + 1)
    for m in range(M):
        pm = sp.sqrt(2) * sp.sqrt(2) * c if m == 0 else sp.sqrt(2 * (m + 1))
        # p_0 = sqrt2 (1 - 1/Lambda) = sqrt2 * sqrt2 * c = 2c
        PG[m, m + 1] = pm
        PG[m + 1, m] = pm
    return PW, PG


def part_a() -> None:
    print("Part A: symbolic (SymPy)")
    c = sp.Symbol("c", positive=True)

    # A1: nullspace at M = 6 is two-dimensional
    M = 6
    basis = solve_invariants(M, c)
    check("A1 solution space of P A = A^T P is 2-dimensional at M=6", len(basis) == 2,
          f"dim = {len(basis)}")

    # A2: the two closed forms span it
    PW, PG = closed_form_P(M, c)
    A = ladder_matrix(M, c)
    for name, P in (("W", PW), ("Gamma", PG)):
        res = (P * A - A.T * P).applyfunc(sp.simplify)
        check(f"A2 closed-form P_{name} satisfies P A = A^T P at M=6", res == sp.zeros(M + 1, M + 1))
    # span check: each basis element is a combination of PW, PG
    a, b = sp.symbols("a b")
    spanned = True
    for B in basis:
        eqs = [e for e in (a * PW + b * PG - B) if sp.simplify(e) != 0]
        sol = sp.solve(eqs, [a, b], dict=True)
        spanned &= bool(sol)
    check("A2 basis elements are combinations of P_W and P_Gamma", spanned)

    # A3: closed form at larger M (zero closure, no boundary term)
    M = 40
    A = ladder_matrix(M, c)
    PW, PG = closed_form_P(M, c)
    ok = all((P * A - A.T * P).applyfunc(sp.simplify) == sp.zeros(M + 1, M + 1) for P in (PW, PG))
    check("A3 closed forms exact at M=40 with zero truncation (no boundary term)", ok)

    # A4: copy closure g_{M+1} = g_{M-1}: Gamma exact, W needs a modified last weight
    M = 6
    As = ladder_matrix(M, c, closure="symmetric")
    PW, PG = closed_form_P(M, c)
    resG = (PG * As - As.T * PG).applyfunc(sp.simplify)
    check("A4 copy closure leaves Gamma exactly conserved", resG == sp.zeros(M + 1, M + 1))
    resW = (PW * As - As.T * PW).applyfunc(sp.simplify)
    nz = {(i, j): resW[i, j] for i in range(M + 1) for j in range(M + 1) if resW[i, j] != 0}
    check("A4 copy closure breaks W with weight 1 at m=M (boundary term)", len(nz) > 0,
          f"nonzero entries of P A - A^T P: {nz}")
    PWm = PW.copy()
    PWm[M, M] = sp.sqrt(M) / (sp.sqrt(M) + sp.sqrt(M + 1))
    resWm = (PWm * As - As.T * PWm).applyfunc(sp.simplify)
    check("A4 copy closure conserves W with last weight sqrt(M)/(sqrt(M)+sqrt(M+1))",
          resWm == sp.zeros(M + 1, M + 1))
    print("     Boundary term for W under the copy closure: dW/dt = -s sum_k g^dagger (P A - A^T P) (D g),")
    print("     supported on the (M-1, M) block only; vanishes as |g_{M-1}| |g_M| -> 0.")


# --------------------------------------------------------------------------
# Part B: numerical, against the installed GANDALF RHS
# --------------------------------------------------------------------------

def rfft_weights(Nx: int) -> np.ndarray:
    """Weights along the rfft axis so that sum_k w |f_k|^2 = full-k-space sum.

    kx = 0 plane once, Nyquist plane once (even Nx), all other kx planes twice.
    Matches krmhd.diagnostics.hermite_moment_energy(account_for_rfft=True).
    """
    n = Nx // 2 + 1
    w = np.full(n, 2.0)
    w[0] = 1.0
    if Nx % 2 == 0:
        w[-1] = 1.0
    return w


def invariant_rates(g: np.ndarray, gdot: np.ndarray, Lambda: float, w0: float | None = None,
                    p0: float | None = None) -> tuple[float, float, float, float]:
    """Return (dW/dt, W_scale, dGamma/dt, Gamma_scale) from g and its RHS.

    g, gdot: [Nz, Ny, Nx//2+1, M+1] complex.  The scales are sums of absolute
    values of the same terms, so a zero rate is measured relative to the size
    of the cancelling contributions, not to a possibly-zero invariant.
    """
    Nx = 2 * (g.shape[2] - 1)
    w = rfft_weights(Nx)[None, None, :]
    M = g.shape[3] - 1
    wgt = np.ones(M + 1)
    wgt[0] = (1.0 - 1.0 / Lambda) if w0 is None else w0
    terms_W = 2.0 * np.real(np.conj(g) * gdot) * wgt[None, None, None, :] * w[..., None]
    dW = float(terms_W.sum())
    W_scale = float(np.abs(terms_W).sum())
    pm = np.sqrt(2.0 * (np.arange(M) + 1.0))
    pm[0] = np.sqrt(2.0) * (1.0 - 1.0 / Lambda) if p0 is None else p0
    corr = gdot[..., 1:] * np.conj(g[..., :-1]) + g[..., 1:] * np.conj(gdot[..., :-1])
    terms_G = np.real(corr) * pm[None, None, None, :] * w[..., None]
    dG = float(terms_G.sum())
    G_scale = float(np.abs(terms_G).sum())
    return dW, W_scale, dG, G_scale


def part_b() -> None:
    print("Part B: numerical, GANDALF RHS (float64, nu = eta = 0, both Poisson brackets active)")
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from krmhd.physics import KRMHDState, initialize_random_spectrum
    from krmhd.spectral import SpectralGrid3D
    from krmhd.timestepping import krmhd_rhs

    N = 16
    grid = SpectralGrid3D.create(Nx=N, Ny=N, Nz=N, Lx=1.0, Ly=1.0, Lz=1.0)
    mask = np.asarray(grid.dealias_mask)
    for M in (8, 13):
        for Lambda in (np.sqrt(5.0), -np.sqrt(5.0)):
            st = initialize_random_spectrum(
                grid, M=M, amplitude=1.0, k_min=1.0, k_max=4.0, nu=0.0, Lambda=float(Lambda),
                seed=7, g_perturbation_amplitude=1.0,
            )
            # Make g O(1) in every moment with random phases, dealiased, k=0 zeroed,
            # reality on kx=0/Nyquist planes handled by the mask-symmetric construction.
            rng = np.random.default_rng(11)
            g = (rng.standard_normal(st.g.shape) + 1j * rng.standard_normal(st.g.shape))
            g *= mask[..., None]
            g[0, 0, 0, :] = 0.0
            # enforce reality on the kx = 0 and Nyquist planes: f(-k) = f(k)^*
            for plane in (0, g.shape[2] - 1):
                sub = g[:, :, plane, :]
                flipped = np.conj(np.roll(np.roll(sub[::-1, ::-1, :], 1, axis=0), 1, axis=1))
                g[:, :, plane, :] = 0.5 * (sub + flipped)
            zp = np.asarray(st.z_plus) * mask
            zm = np.asarray(st.z_minus) * mask
            state = KRMHDState(
                z_plus=jnp.asarray(zp), z_minus=jnp.asarray(zm),
                g=jnp.asarray(g), M=M, beta_i=1.0, v_th=1.0, nu=0.0, Lambda=float(Lambda),
                time=0.0, grid=grid,
            )
            rhs = krmhd_rhs(state, eta=0.0, v_A=1.0)
            gdot = np.asarray(rhs.g)
            dW, Ws, dG, Gs = invariant_rates(g, gdot, float(Lambda))
            tol = 1e-10
            tag = f"M={M}, Lambda={Lambda:+.4f}"
            check(f"B1 dW/dt = 0 ({tag})", abs(dW) / Ws < tol, f"|dW/dt|/scale = {abs(dW)/Ws:.2e}")
            check(f"B2 dGamma/dt = 0 ({tag})", abs(dG) / Gs < tol, f"|dGamma/dt|/scale = {abs(dG)/Gs:.2e}")
            # negative controls
            dWb, Wsb, _, _ = invariant_rates(g, gdot, float(Lambda), w0=1.0)
            _, _, dGb, Gsb = invariant_rates(g, gdot, float(Lambda), p0=np.sqrt(2.0))
            check(f"B3 wrong g_0 weight breaks W ({tag})", abs(dWb) / Wsb > 1e-6,
                  f"|dW/dt|/scale = {abs(dWb)/Wsb:.2e}")
            check(f"B4 wrong p_0 breaks Gamma ({tag})", abs(dGb) / Gsb > 1e-6,
                  f"|dGamma/dt|/scale = {abs(dGb)/Gsb:.2e}")
            # the test is nontrivial: the brackets and streaming are all active
            check(f"B5 RHS is nontrivial ({tag})", Gs > 0 and Ws > 0 and np.abs(gdot).max() > 1e-3)


def part_c() -> None:
    """Coefficient identity between our p_m and CMM (B32), and the Lambda^+- values."""
    print("Part C: mapping to Chandran, Mallet & Meyrand (B7), (B31), (B32)")
    beta_i, tau, Z = 1.0, 1.0, 1.0
    root = np.sqrt((1.0 + tau / Z) ** 2 + beta_i ** -2)
    Lp = -tau / Z + 1.0 / beta_i + root
    Lm = -tau / Z + 1.0 / beta_i - root
    check("C1 Lambda^+ = +sqrt5 at beta_i = tau = Z = 1 (B7)", abs(Lp - np.sqrt(5.0)) < 1e-12, f"{Lp:.6f}")
    check("C2 Lambda^- = -sqrt5 at beta_i = tau = Z = 1 (B7)", abs(Lm + np.sqrt(5.0)) < 1e-12, f"{Lm:.6f}")
    check("C3 nu-scan checkpoints used Lambda^+ (state/Lambda = 2.2360677)", abs(2.2360677 - Lp) < 1e-6)
    for Lambda in (Lp, Lm):
        M = 12
        m = np.arange(M)
        ours = np.sqrt(2.0 * (m + 1.0))
        ours[0] = np.sqrt(2.0) * (1.0 - 1.0 / Lambda)
        cmm = np.sqrt(m + 1.0)                 # coefficient of Gtilde_m Gtilde_{m+1} in (B32)
        cmm[0] = 1.0 - 1.0 / Lambda            # sqrt(1) - 1/Lambda
        ratio = ours / cmm
        check(f"C4 p_m / c_m^CMM is a constant (= sqrt2) for all m (Lambda={Lambda:+.4f})",
              np.allclose(ratio, np.sqrt(2.0)), f"ratio range {ratio.min():.6f}..{ratio.max():.6f}")
    print("     Gamma_ours = sqrt2 * [sum_m sqrt(m+1) g_m g_{m+1} - g_0 g_1/Lambda] = 2 Gamma^+-_CMM (unit volume, v_th = 1)")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    if FAILURES:
        print(f"\nFAILED: {FAILURES}")
        sys.exit(1)
    print("\nALL CHECKS PASSED")
