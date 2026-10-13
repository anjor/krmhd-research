"""t7b_imex_order: checks why GANDALF's IMEX-RK222 step couples g to z+- only at first order in dt: the order
conditions of the ARS(2,2,2) tableau with the stage-2 Elsasser fields taken at t + dt/2, the first-order convergence
of g(T) through gandalf_step against a Lawson-RK4 reference, a re-implementation of the step from krmhd's own
functions that reproduces gandalf_step and recovers second order when the stage time is gamma dt, and the
second-order drift of the invariants. Supports Appendix D of the paper (summarised in Section 8).

A  (SymPy) ARS(2,2,2) explicit tableau with GANDALF's gamma = 1 - 1/sqrt2, delta = 1 - 1/(2 gamma): weights
   b~ = (delta, 1 - delta, 0), stage time c~_2 = gamma, sum b~ c~ = (1 - delta) gamma = 1/2 (second order). GANDALF
   evaluates the stage-2 explicit term with z+- at t + dt/2 (the Elsasser midpoint predictor), so the time-dependence
   of N(g, z(t)) is sampled with sum b~ c_z = (1 - delta)/2 = 1/(4 gamma) = 1/2 + sqrt2/4: the local error contains
   (sqrt2/4) dt^2 (dN/dz) dz/dt, a first-order global error.
B  GANDALF measurement (float64, 8^3, M = 6, Lam = sqrt5): self-convergence of g(T) for gandalf_step IMEX against a
   Lawson-RK4 reference: order 1 for a single propagating z+ (z- = 0, the Elsasser update is then exact) and for
   general nonlinear z+-; Lawson-RK4 is order 4 in the first case.
C  Mechanism: a re-implementation of one IMEX step from krmhd's own pieces (_krmhd_rhs_jit, build_implicit_operator,
   factor_imex_operator, imex_solve) reproduces gandalf_step to round-off when the stage-2 z+- is taken at t + dt/2;
   taking it at t + gamma dt instead restores second order.
D  The first-order defect is a pure phase error (it advects every slice with an incompressible field taken at the wrong
   time), so it does not change the invariants at leading order. For non-autonomous advection both the GANDALF step and
   the consistent variant change |g|^2 by O(dt^3) per step (SymPy, scalar oscillator with omega(t) = omega0 + omega1 t:
   |R|^2 = 1 + c' omega0 omega1 dt^3 + O(dt^4)), so W and Gamma drift at second order in dt asymptotically, with a
   third-order part that dominates at coarse dt; the trajectory error of g is first order.
"""
from __future__ import annotations

import time

import numpy as np
import sympy as sp

from tcore import (Checker, section, ladder, weight, make_grid, random_field, make_state, ksum_weights, qform, SQRT5,
                   jnp, maxgrad, dz)
from krmhd.timestepping import (gandalf_step, _krmhd_rhs_jit, _fields_from_state, KRMHDFields, _IMEX_GAMMA,
                                _IMEX_DELTA)
from krmhd.hermite import build_implicit_operator, factor_imex_operator, imex_solve

C = Checker("t7b_imex_order")

section("A. Order conditions")
g_ = 1 - 1 / sp.sqrt(2)
d_ = 1 - 1 / (2 * g_)
b = [d_, 1 - d_, 0]
c = [0, g_, 1]
s1 = sp.simplify(sum(b))
s2 = sp.simplify(sum(bi * ci for bi, ci in zip(b, c)))
s2z = sp.simplify(b[1] * sp.Rational(1, 2))
defect = sp.nsimplify(sp.simplify(s2z - sp.Rational(1, 2)))
print(f"   gamma = {sp.nsimplify(g_)} ({float(g_)!r} vs GANDALF {_IMEX_GAMMA!r}), delta = {sp.nsimplify(sp.simplify(d_))} "
      f"({float(d_)!r} vs {_IMEX_DELTA!r})")
print(f"   sum b~ = {s1}; sum b~ c~ (stage time gamma) = {s2}; with z at t + dt/2: (1 - delta)/2 = {sp.simplify(s2z)} "
      f"= 1/2 + {defect}")
C.check("A1 consistent stage times give sum b~ c~ = 1/2; GANDALF's z at t + dt/2 gives 1/(4 gamma) = 1/2 + sqrt2/4",
        s1 == 1 and s2 == sp.Rational(1, 2) and sp.simplify(defect - sp.sqrt(2) / 4) == 0
        and abs(_IMEX_GAMMA - float(g_)) < 1e-16 and abs(_IMEX_DELTA - float(d_)) < 1e-15)

dt, w0, w1, cc = sp.symbols("dt omega0 omega1 c", real=True)
wfun = lambda t: w0 + w1 * t
gs1 = 1 + dt * g_ * sp.I * wfun(0)
outA2 = {}
for cval, lab in ((g_, "gamma"), (sp.Rational(1, 2), "1/2")):
    gn = 1 + dt * d_ * sp.I * wfun(0) + dt * (1 - d_) * sp.I * wfun(cval * dt) * gs1
    mod2 = sp.series(sp.expand(gn * sp.conjugate(gn)), dt, 0, 4).removeO()
    lerr = sp.series(gn - sp.exp(sp.I * (w0 * dt + w1 * dt ** 2 / 2)), dt, 0, 3).removeO()
    outA2[lab] = (sp.simplify(sp.expand(mod2 - 1)), sp.simplify(sp.expand(lerr)))
    print(f"   oscillator d_t g = i omega(t) g, stage time {lab}: |R|^2 - 1 = {outA2[lab][0]} + O(dt^4); "
          f"local error up to dt^2: {outA2[lab][1]}")
okA2 = (sp.simplify(outA2["gamma"][1]) == 0 and sp.simplify(outA2["1/2"][1] - sp.sqrt(2) * sp.I * w1 * dt ** 2 / 4) == 0
        and sp.simplify(outA2["gamma"][0] - sp.sqrt(2) * w0 * w1 * dt ** 3 / 2) == 0
        and sp.simplify(outA2["1/2"][0] - (1 + sp.sqrt(2)) * w0 * w1 * dt ** 3 / 2) == 0)
C.check("A2 non-autonomous oscillator: stage time 1/2 adds the local error i (sqrt2/4) omega1 dt^2 (a phase error); both "
        "stage times change |g|^2 by O(dt^3) per step", okA2)

section("B/C. Convergence of g(T): gandalf_step (IMEX, Lawson) and the re-implemented IMEX step")
N = 8
grid = make_grid(N)
M = 6
Lam = SQRT5
mask = jnp.asarray(grid.dealias_mask)


def imex_variant(state, dt: float, cz: float):
    """One ARS(2,2,2) step assembled from krmhd's own functions exactly as _gandalf_step_imex222_jit does, except that
    the stage-2 explicit term uses z+- predicted to t + cz dt (GANDALF: cz = 1/2). eta = eta_z = nu = 0."""
    f = _fields_from_state(state)
    kx, ky, kz = grid.kx, grid.ky, grid.kz
    kz3 = kz[:, None, None]
    kz0 = jnp.zeros_like(kz)
    L = build_implicit_operator(kz, state.beta_i, 0.0, state.M, state.Lambda, 1, "zero")
    lu, piv = factor_imex_operator(L, dt, _IMEX_GAMMA)

    def rhs(ff, kzz):
        return _krmhd_rhs_jit(ff, kx, ky, kzz, mask, 0.0, 1.0, state.beta_i, 0.0, state.Lambda, state.M,
                              grid.Nz, grid.Ny, grid.Nx, closure="zero")

    r0 = rhs(f, kz)
    nlp0 = r0.z_plus - 1j * kz3 * f.z_plus
    nlm0 = r0.z_minus + 1j * kz3 * f.z_minus
    zph = jnp.exp(1j * kz3 * dt / 2) * (f.z_plus + dt / 2 * nlp0)
    zmh = jnp.exp(-1j * kz3 * dt / 2) * (f.z_minus + dt / 2 * nlm0)
    N1 = rhs(f, kz0).g
    g1 = imex_solve(lu, piv, f.g + dt * _IMEX_GAMMA * N1)
    zpc = jnp.exp(1j * kz3 * cz * dt) * (f.z_plus + cz * dt * nlp0)
    zmc = jnp.exp(-1j * kz3 * cz * dt) * (f.z_minus + cz * dt * nlm0)
    N2 = rhs(KRMHDFields(z_plus=zpc, z_minus=zmc, g=g1, time=f.time + cz * dt), kz0).g
    rh = rhs(KRMHDFields(z_plus=zph, z_minus=zmh, g=g1, time=f.time + dt / 2), kz)
    nlph = rh.z_plus - 1j * kz3 * zph
    nlmh = rh.z_minus + 1j * kz3 * zmh
    zpn = jnp.exp(1j * kz3 * dt) * f.z_plus + jnp.exp(1j * kz3 * dt / 2) * dt * nlph
    zmn = jnp.exp(-1j * kz3 * dt) * f.z_minus + jnp.exp(-1j * kz3 * dt / 2) * dt * nlmh
    Lg1 = jnp.einsum("zij,zyxj->zyxi", L, g1)
    gn = imex_solve(lu, piv, f.g + dt * (1 - _IMEX_GAMMA) * Lg1 + dt * _IMEX_DELTA * N1 + dt * (1 - _IMEX_DELTA) * N2)
    gn = gn * mask[..., None]
    return state.model_copy(update={"z_plus": zpn, "z_minus": zmn, "g": gn, "time": state.time + dt})


def integrate(st, T: float, nsteps: int, how: str):
    dt = T / nsteps
    for _ in range(nsteps):
        if how == "imex":
            st = gandalf_step(st, dt, eta=0.0, v_A=1.0, nu=0.0, scheme="imex_rk222")
        elif how == "lawson":
            st = gandalf_step(st, dt, eta=0.0, v_A=1.0, nu=0.0, scheme="lawson_rk4")
        elif how == "var_half":
            st = imex_variant(st, dt, 0.5)
        elif how == "var_gamma":
            st = imex_variant(st, dt, _IMEX_GAMMA)
    return st


runs = []
okB = True
okC = True
okD = True
A = ladder(M, Lam)
Pi = np.diag(weight(M, Lam))
ZGRAD = 2.0
for case in ("single z+ (z- = 0)", "general z+-"):
    rng = np.random.default_rng(21)
    zp = random_field(grid, rng, amp=1.0)
    zm = np.zeros_like(zp) if case.startswith("single") else random_field(grid, rng, amp=1.0)
    scale = ZGRAD / max(maxgrad(grid, zp), maxgrad(grid, zm))      # real-space max|grad_perp z+-| = ZGRAD
    zp = zp * scale
    zm = zm * scale
    g = random_field(grid, rng, slope=-1.0, extra_shape=(M + 1,))
    st0 = make_state(grid, M, Lam, zp, zm, g, beta_i=1.0)
    kz0 = jnp.zeros_like(grid.kz)
    Nfull = np.asarray(_krmhd_rhs_jit(_fields_from_state(st0), grid.kx, grid.ky, kz0, mask, 0.0, 1.0, 1.0, 0.0, Lam, M,
                                      grid.Nz, grid.Ny, grid.Nx, closure="zero").g)
    stream = np.stack([dz(grid, np.einsum("mn,zyxn->zyxm", A, np.asarray(g))[..., m]) for m in range(M + 1)], -1)
    print(f"   case: {case}; max|grad_perp z+| = {maxgrad(grid, zp):.3f}, max|grad_perp z-| = {maxgrad(grid, zm):.3f}; "
          f"|N(g)| / |s d_z(A g)| = {np.linalg.norm(Nfull) / np.linalg.norm(stream):.3f}")
    # C1: the variant with cz = 1/2 reproduces gandalf_step
    t0 = time.time()
    a = gandalf_step(st0, 0.01, eta=0.0, v_A=1.0, nu=0.0, scheme="imex_rk222")
    np.asarray(a.g)
    runs.append(("imex single step (C1)", case, 1, 0.01, time.time() - t0))
    t0 = time.time()
    bb = imex_variant(st0, 0.01, 0.5)
    np.asarray(bb.g)
    runs.append(("var_half single step (C1)", case, 1, 0.01, time.time() - t0))
    rep = max(float(np.abs(np.asarray(a.g) - np.asarray(bb.g)).max() / np.abs(np.asarray(a.g)).max()),
              float(np.abs(np.asarray(a.z_plus) - np.asarray(bb.z_plus)).max() / np.abs(np.asarray(a.z_plus)).max()))
    print(f"     re-implemented step (z at t + dt/2) vs gandalf_step: max relative difference {rep:.1e}")
    okC = okC and rep < 1e-13
    T = 0.2
    t0 = time.time()
    ref = integrate(st0, T, 2048, "lawson")
    runs.append(("lawson_rk4 reference", case, 2048, T / 2048, time.time() - t0))
    gref = np.asarray(ref.g)
    W0 = qform(grid, np.asarray(st0.g), Pi)
    G0 = qform(grid, np.asarray(st0.g), Pi @ A) / 2
    res = {}
    for how in ("imex", "var_half", "var_gamma", "lawson"):
        errs, dW, dG = [], [], []
        nlist = (16, 32, 64, 128, 256) if how != "lawson" else (16, 32, 64, 128)
        t0 = time.time()
        for ns in nlist:
            st = integrate(st0, T, ns, how)
            gg = np.asarray(st.g)
            errs.append(np.linalg.norm(gg - gref) / np.linalg.norm(gref))
            dW.append(abs(qform(grid, gg, Pi) - W0) / W0)
            dG.append(abs(qform(grid, gg, Pi @ A) / 2 - G0) / abs(G0))
        wall = time.time() - t0
        runs.append((f"{how} convergence", case, sum(nlist), None, wall))
        orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
        oW = np.log2(np.array(dW[:-1]) / np.array(dW[1:]))
        res[how] = (orders, oW, errs, dW)
        print(f"     {how:9s}: rel. error of g(T) for dt = T/{list(nlist)}: " + " ".join(f"{e:.2e}" for e in errs)
              + f"; observed orders {np.round(orders, 2).tolist()}")
        if how != "lawson":
            print(f"                |W(T)-W(0)|/W(0): " + " ".join(f"{e:.2e}" for e in dW) + f"; orders {np.round(oW, 2).tolist()};"
                  f" |Gamma(T)-Gamma(0)|/|Gamma(0)|: " + " ".join(f"{e:.2e}" for e in dG))
    okB = okB and abs(res["imex"][0][-1] - 1.0) < 0.1 and abs(res["var_half"][0][-1] - 1.0) < 0.1
    okC = okC and abs(res["var_gamma"][0][-1] - 2.0) < 0.15
    if case.startswith("single"):
        okB = okB and res["lawson"][0][-1] > 3.5
    if case.startswith("single"):
        oW_i, oW_g = res["imex"][1], res["var_gamma"][1]
        dWr = np.array(res["imex"][3]) / np.array(res["var_gamma"][3])
        print(f"     W-drift ratio GANDALF/consistent variant at each dt: {np.round(dWr, 3).tolist()}")
        okD = okD and np.all(oW_i > 1.9) and np.all(np.diff(oW_i) < 0) and abs(oW_i[-1] - oW_g[-1]) < 0.05 \
            and np.all(np.abs(dWr - 1) < 0.35)
C.check("B1 gandalf_step IMEX: g(T) converges at first order (z- = 0 and general z+-); Lawson-RK4 at fourth order "
        "for z- = 0", okB)
C.check("C1 the re-implemented step reproduces gandalf_step; moving the stage-2 z+- from t + dt/2 to t + gamma dt "
        "restores second order", okC)
C.check("D1 single-wave case: the W drift converges at local orders 2.6 -> 2.1 (second order asymptotically) for GANDALF and the consistent variant alike, while g(T) converges at first order", okD)
print("GANDALF integrations (what, case, steps, dt, wall s), grid 8^3, M = 6, Lam = sqrt5, T = 0.2:")
for r in runs:
    print(f"   {r[0]} | {r[1]} | steps={r[2]} | dt={r[3] if r[3] is None else f'{r[3]:.2e}'} | wall={r[4]:.1f}s")
C.finish()
