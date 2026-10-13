"""d04: the forced version of the frozen-line theorem, tested in Phi-forced ideal turbulence.  Three impulsive g
injections are followed in one Phi-forced ideal Alfvenic history (RK4 on GANDALF's right-hand side, N = 24, 32, 48),
and the script checks linearity, that each single response follows the frozen-line prediction from its own injection
time, that averaging over the signs of the injections removes the pathwise cross terms (white forcing), and that for
temporally correlated injections the cross term depends on the relabelling between the injection times.  It supports
Section 6 (corollary for white forcing).

Setting: z+- start from random fields and are forced through Phi only (the same white-noise increment added to z+ and
z- every step, as krmhd.forcing.force_alfven_modes does), eta = 0, so the induction equation stays ideal.  g receives
K = 3 impulsive injections F_n at t_n = 0, 0.04, 0.08 (random g_0, g_1 patterns); collisionless, M = 8.
All g's share the same Alfvenic history (classical RK4 on GANDALF's RHS; g is passive, so the z+- stages are
computed once and reused).

Checks
 F1 linearity: the run with all injections equals the sum of the single-injection runs (round-off).
 F2 each single-injection response obeys the frozen-line theorem from its own injection time:
    W_n(m, t) = W(m) of  exp[-(t - t_n) s A grad_par(t_n)] F_n  (field lines frozen at t_n, no flow);
    the difference must shrink with N.
 F3 pathwise, W(m, t) of the forced run is NOT the sum of the single responses (cross terms between injections);
    the average over the 2^K sign patterns of the injections IS the sum (white-in-time forcing).
 F4 temporally correlated forcing (the same pattern F_0 injected at 0 and again at t_1 = gap): the sign average keeps
    the cross term 2<y_A, y_B>, which involves the relabelling between 0 and t_1; the cross term of the two frozen
    responses without relabelling misses it by an amount that grows with the gap (eddy time ~ 0.1 here).
Usage: python d04_forcing_superposition.py [N list, default 24,32,48]
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from dcore import (Box, Checks, SQRT5, W_m, ladder, log_run, make_rhs, make_rk4, pweights, relerr, timer)  # noqa: E402

NS = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [24, 32, 48]
M, LAM, BETA, DT, N0 = 8, SQRT5, 1.0, 0.002, 8
S = np.sqrt(BETA)
P = pweights(M, LAM)
T_INJ = (0.0, 0.04, 0.08)
T_END = 0.16
GAPS = (0.04, 0.08, 0.12)        # correlated forcing: the pattern F_0 injected again after these gaps
FAMP = 3.0


def make_rk4_multi(rhs_full, rhs_g, dt):
    """RK4 for (z+, z-) and a list of passive g's that share the z+- stages."""

    @jax.jit
    def zstages(zp, zm):
        g0 = jnp.zeros(zp.shape + (M + 1,), complex)
        k1 = rhs_full(zp, zm, g0)[:2]
        z2 = (zp + 0.5 * dt * k1[0], zm + 0.5 * dt * k1[1])
        k2 = rhs_full(*z2, g0)[:2]
        z3 = (zp + 0.5 * dt * k2[0], zm + 0.5 * dt * k2[1])
        k3 = rhs_full(*z3, g0)[:2]
        z4 = (zp + dt * k3[0], zm + dt * k3[1])
        k4 = rhs_full(*z4, g0)[:2]
        znew = (zp + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
                zm + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))
        return (zp, zm), z2, z3, z4, znew

    @jax.jit
    def gstep(z1, z2, z3, z4, g):
        k1 = rhs_g(*z1, g)
        k2 = rhs_g(*z2, g + 0.5 * dt * k1)
        k3 = rhs_g(*z3, g + 0.5 * dt * k2)
        k4 = rhs_g(*z4, g + dt * k3)
        return g + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return zstages, gstep


def fields8(seed: int):
    b0 = Box(N0)
    rng = np.random.default_rng(seed)
    phi = b0.random_field(rng, (1, 2.5), 2)
    psi = b0.random_field(rng, (1, 2.5), 2)
    inj = []
    for _ in T_INJ:
        F = np.zeros((N0, N0, N0 // 2 + 1, M + 1), complex)
        F[..., 0] = b0.random_field(rng, (1, 2.5), 2)
        F[..., 1] = 0.7 * b0.random_field(rng, (1, 2.5), 2)
        inj.append(F)
    forc = [b0.random_field(rng, (1, 2), 1) for _ in range(int(round(T_END / DT)) + 1)]
    return phi, psi, inj, forc


chk = Checks()
results = {}
for N in NS:
    box = Box(N)
    phi8, psi8, inj8, forc8 = fields8(7)
    phi = box.embed(phi8, N0)
    psi = box.embed(psi8, N0)
    phi *= 1.0 / box.rms_grad(phi)
    psi *= 1.0 / box.rms_grad(psi)
    inj = []
    for F8 in inj8:
        F = box.embed(F8, N0)
        inj.append(F / np.sqrt(np.sum(W_m(box, F) * P)))
    forc = []
    for f8 in forc8:
        f = box.embed(f8, N0)
        forc.append(f / box.rms_grad(f))
    rhs_full, rhs_g = make_rhs(box, M, LAM, BETA)
    zstages, gstep = make_rk4_multi(rhs_full, rhs_g, DT)
    _, step_frozen = make_rk4(rhs_full, rhs_g, np.zeros(M + 1), DT)
    zp, zm = jnp.asarray(phi + psi), jnp.asarray(phi - psi)
    K = len(T_INJ)
    nsteps = int(round(T_END / DT))
    inj_steps = [int(round(t / DT)) for t in T_INJ]
    # g's: single responses y_n (n = 0..K-1), the combined run (all +1), and the correlated pair (F_1 at t_0 and t_1)
    gs = [jnp.zeros((N, N, N // 2 + 1, M + 1), complex) for _ in range(K + 1 + len(GAPS))]
    gap_steps = [int(round(gp / DT)) for gp in GAPS]
    psi_at = {}
    clock = timer()
    rms_phi = []
    for n in range(nsteps + 1):
        for k, ns_ in enumerate(inj_steps):
            if n == ns_:
                gs[k] = gs[k] + jnp.asarray(inj[k])
                gs[K] = gs[K] + jnp.asarray(inj[k])
                psi_at[k] = np.asarray(zp - zm) / 2
        for ig, ns_ in enumerate(gap_steps):
            if n == ns_:
                gs[K + 1 + ig] = gs[K + 1 + ig] + jnp.asarray(inj[0])   # F_0 injected again after the gap
                psi_at[("gap", ig)] = np.asarray(zp - zm) / 2
        if n == nsteps:
            break
        z1, z2, z3, z4, znew = zstages(zp, zm)
        gs = [gstep(z1, z2, z3, z4, g) for g in gs]
        dphi = FAMP * np.sqrt(DT) * forc[n]                       # Phi-only white forcing
        zp, zm = znew[0] + dphi, znew[1] + dphi
        if n % 10 == 0:
            rms_phi.append(box.rms_grad(np.asarray(zp + zm) / 2))
    wall_nl = clock()
    # frozen-line responses: from t_n with Psi(t_n) frozen, no flow
    fr = []
    for k in range(K):
        g = jnp.asarray(inj[k])
        zpF, zmF = jnp.asarray(psi_at[k]), jnp.asarray(-psi_at[k])
        for _ in range(nsteps - inj_steps[k]):
            g = step_frozen(zpF, zmF, g)
        fr.append(np.asarray(g))
    # correlated pairs: frozen responses of F_0 injected at t_1 = gap (field lines frozen at t_1)
    frB = []
    for ig, ns_ in enumerate(gap_steps):
        gB = jnp.asarray(inj[0])
        zpF, zmF = jnp.asarray(psi_at[("gap", ig)]), jnp.asarray(-psi_at[("gap", ig)])
        for _ in range(nsteps - ns_):
            gB = step_frozen(zpF, zmF, gB)
        frB.append(np.asarray(gB))
    wall = clock()
    log_run("d04_forcing_superposition.py", "forced frozen-line test: Phi-forced ideal turbulence, 3 g injections",
            N, M, nsteps * (K + 1 + len(GAPS)) + sum(nsteps - s for s in inj_steps) + sum(nsteps - s for s in gap_steps), wall, dict(dt=DT, T=T_END, famp=FAMP))
    y = [np.asarray(g) for g in gs]
    print(f"N = {N}: rms|grad Phi| during run {min(rms_phi):.2f} .. {max(rms_phi):.2f}; wall {wall:.0f}s", flush=True)
    # F1 linearity
    lin = np.abs(y[K] - sum(y[:K])).max() / np.abs(y[K]).max()
    chk(f"N={N} F1 combined forced run = sum of single responses", lin < 1e-12, f"{lin:.1e}")
    # F2 per-injection frozen-line agreement
    e2 = []
    for k in range(K):
        e = relerr(W_m(box, y[k]), W_m(box, fr[k]))
        e2.append(e)
        print(f"  injection at t={T_INJ[k]:.2f}: W_n(m, t={T_END}) NL vs frozen-at-t_n: rel. diff {e:.2e}", flush=True)
    results[N] = dict(e2=e2)
    # F3 cross terms and sign average
    Wsum = sum(W_m(box, y[k]) for k in range(K))
    Wfr = sum(W_m(box, fr[k]) for k in range(K))
    cross = []
    avg = np.zeros(M + 1)
    for eps in itertools.product((1, -1), repeat=K):
        ye = sum(e * y[k] for k, e in enumerate(eps))
        We = W_m(box, ye)
        avg += We / 2 ** K
        cross.append(relerr(We, Wsum))
    print(f"  pathwise: W(m) of a forced run differs from sum of single responses by {min(cross):.2f} .. "
          f"{max(cross):.2f} (sum |diff| / sum W) over the 2^K sign patterns", flush=True)
    e_avg = relerr(avg, Wsum)
    e_avg_fr = relerr(avg, Wfr)
    print(f"  sign-averaged W(m) vs sum of single NL responses: {e_avg:.1e};  vs sum of frozen-line responses: "
          f"{e_avg_fr:.2e}", flush=True)
    results[N].update(cross=(min(cross), max(cross)), e_avg_fr=e_avg_fr)
    chk(f"N={N} F3 pathwise cross terms are not small", max(cross) > 0.05, f"max {max(cross):.2f}")
    chk(f"N={N} F3 sign average = sum of single responses (exact algebra)", e_avg < 1e-12, f"{e_avg:.1e}")
    # F4 correlated pairs
    yA = y[0]                                   # F_0 injected at 0 (NL)
    mism_list = []
    for ig, gp in enumerate(GAPS):
        yB = y[K + 1 + ig]                      # F_0 injected at t_1 = gap (NL)
        WA, WB = W_m(box, yA), W_m(box, yB)
        cross_nl = W_m(box, yA + yB) - WA - WB  # = 2 <y_A, y_B>_m, survives the average over a common sign
        cross_fr = W_m(box, fr[0] + frB[ig]) - W_m(box, fr[0]) - W_m(box, frB[ig])
        eB = relerr(WB, W_m(box, frB[ig]))
        ratio = np.abs(cross_nl).sum() / (WA + WB).sum()
        mism = np.abs(cross_nl - cross_fr).sum() / np.abs(cross_nl).sum()
        mism_list.append(mism)
        print(f"  correlated forcing, gap {gp:.2f}: response to the second injection vs frozen-at-t_1 {eB:.1e}; "
              f"cross term sum|2<y_A,y_B>_m| = {ratio:.2f} of sum(W_A + W_B); frozen responses without relabelling "
              f"miss it by {mism:.2f} (relative)", flush=True)
        if ig == 0:
            chk(f"N={N} F4 correlated forcing: the cross term survives the average and is O(1)", ratio > 0.05, f"{ratio:.2f}")
    results[N].update(mism=mism_list)
    chk(f"N={N} F4 the cross term needs the relabelling: the no-relabelling estimate gets worse with the gap and is "
        f"off by > 10% at gap {GAPS[-1]}", mism_list[-1] > 0.1 and mism_list[-1] > mism_list[0],
        ", ".join(f"{x:.2f}" for x in mism_list))

# convergence of F2 with N
for k in range(len(T_INJ)):
    e = [results[N]["e2"][k] for N in NS]
    chk(f"F2 injection {k} (t_n = {T_INJ[k]}): NL vs frozen-at-t_n decreases with N and < 2e-3 at N={NS[-1]}",
        all(e[i + 1] < e[i] for i in range(len(e) - 1)) and e[-1] < 2e-3, " -> ".join(f"{x:.1e}" for x in e))
chk.finish()
