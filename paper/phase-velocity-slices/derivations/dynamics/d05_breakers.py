"""d05: what breaks the frozen-line theorem, and what does not.  Nonlinear runs with one added term each (viscosity
or white forcing of Phi, resistivity or white forcing of Psi, k_perp^4 or k_z^2 damping of g; RK4 on GANDALF's
right-hand side at N = 16, 24, 32) are compared with the frozen-line prediction, and the script checks that the
difference converges with N when the hypotheses hold and stays at an N-independent level when one is violated.  It
supports Section 6 (hypotheses of the theorem and what breaks it).

The theorem needs only (i) an ideal induction equation, d_t Psi + {Phi, Psi} - d_z Phi = 0, and (ii) no operator on g
that fails to commute with the E x B relabelling (spatial dissipation of g).  The momentum (Phi) equation is
unrestricted.  Test: NL vs the frozen-line prediction FR (field lines frozen at t = 0, no flow; FR carries the same
g-dissipation as NL where NL has one), at N = 16, 24, 32.  A case that satisfies the hypotheses must converge
(NL - FR decreasing with N); a case that violates them must not.
  ideal       : control
  visc_phi    : viscosity on Phi only (d_t Phi += nu_v lap Phi)       -> hypotheses hold
  force_phi   : white-noise forcing of Phi only (z+ and z- kicked equally, as krmhd.forcing.force_alfven_modes)
  resist_psi  : resistivity on Psi only (d_t Psi += eta lap Psi)       -> (i) violated
  force_psi   : white-noise forcing of Psi only (z+ and z- kicked oppositely, as force_alfven_modes_balanced does in part)
  damp_g_perp : -eta_g k_perp^4 g on every g_m, included in FR too   -> (ii) violated
  damp_g_par  : -eta_z k_z^2 g on every g_m, included in FR too      -> (ii) violated
Same physical initial fields as d02 (M = 8, Lambda = sqrt5, s = 1, rms|grad Phi| = rms|grad Psi| = 1), RK4 on
GANDALF's RHS plus the extra terms, dt = 0.00125, outputs at t = 0.125, 0.25.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from dcore import Box, Checks, SQRT5, W_m, log_run, make_rhs, pweights, relerr, timer  # noqa: E402

NS = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [16, 24, 32]
M, LAM, BETA, DT, N0 = 8, SQRT5, 1.0, 0.00125, 8
P = pweights(M, LAM)
TOUT = (0.125, 0.25)
CASES = ("ideal", "visc_phi", "force_phi", "resist_psi", "force_psi", "damp_g_perp", "damp_g_par")
NU_V, ETA_R, FAMP, ETA_G, ETA_Z = 0.01, 0.01, 3.0, 4e-6, 0.02


def fields8(seed: int = 1):
    b0 = Box(N0)
    rng = np.random.default_rng(seed)
    phi = b0.random_field(rng, (1, 2.5), 2)
    psi = b0.random_field(rng, (1, 2.5), 2)
    g = np.zeros((N0, N0, N0 // 2 + 1, M + 1), complex)
    g[..., 0] = b0.random_field(rng, (1, 2.5), 2)
    g[..., 1] = 0.5 * b0.random_field(rng, (1, 2.5), 2)
    rng2 = np.random.default_rng(99)
    forc = [b0.random_field(rng2, (1, 2), 1) for _ in range(int(round(TOUT[-1] / DT)) + 1)]
    return phi, psi, g, forc


def run_case(box: Box, case: str, phi, psi, g0, forc) -> dict:
    rhs_full, rhs_g = make_rhs(box, M, LAM, BETA)
    lap = jnp.asarray(-box.KP2)
    dg = jnp.zeros(box.KP2.shape)
    if case == "damp_g_perp":
        dg = jnp.asarray(ETA_G * box.KP2 ** 2) * jnp.ones((box.N, 1, 1))
    if case == "damp_g_par":
        dg = jnp.asarray(ETA_Z * box.KZ ** 2) * jnp.ones((1, box.N, box.N // 2 + 1))
    dg4 = dg[..., None]

    def f_full(zp, zm, g):
        a, b, c = rhs_full(zp, zm, g)
        if case == "visc_phi":
            d = NU_V * lap * (zp + zm) / 2
            a, b = a + d, b + d
        if case == "resist_psi":
            d = ETA_R * lap * (zp - zm) / 2
            a, b = a + d, b - d
        return a, b, c - dg4 * g

    def f_fr(zp, zm, g):
        return rhs_g(zp, zm, g) - dg4 * g

    @jax.jit
    def step(zp, zm, g):
        k1 = f_full(zp, zm, g)
        k2 = f_full(zp + 0.5 * DT * k1[0], zm + 0.5 * DT * k1[1], g + 0.5 * DT * k1[2])
        k3 = f_full(zp + 0.5 * DT * k2[0], zm + 0.5 * DT * k2[1], g + 0.5 * DT * k2[2])
        k4 = f_full(zp + DT * k3[0], zm + DT * k3[1], g + DT * k3[2])
        return (zp + DT / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
                zm + DT / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
                g + DT / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]))

    @jax.jit
    def step_fr(zp, zm, g):
        k1 = f_fr(zp, zm, g)
        k2 = f_fr(zp, zm, g + 0.5 * DT * k1)
        k3 = f_fr(zp, zm, g + 0.5 * DT * k2)
        k4 = f_fr(zp, zm, g + DT * k3)
        return g + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    zp, zm, gN = jnp.asarray(phi + psi), jnp.asarray(phi - psi), jnp.asarray(g0)
    zpF, zmF, gF = jnp.asarray(psi), jnp.asarray(-psi), jnp.asarray(g0)
    out = {}
    nsteps = int(round(TOUT[-1] / DT))
    for n in range(nsteps + 1):
        t = round(n * DT, 10)
        for tt in TOUT:
            if abs(t - tt) < 1e-9:
                out[tt] = (W_m(box, gN), W_m(box, gF))
        if n == nsteps:
            break
        zp, zm, gN = step(zp, zm, gN)
        gF = step_fr(zpF, zmF, gF)
        if case in ("force_phi", "force_psi"):
            d = jnp.asarray(FAMP * np.sqrt(DT) * forc[n])
            zp, zm = (zp + d, zm + d) if case == "force_phi" else (zp + d, zm - d)
    out["rms_psi_end"] = box.rms_grad(np.asarray(zp - zm) / 2)
    return out


chk = Checks()
res = {}
for N in NS:
    box = Box(N)
    phi8, psi8, g8, forc8 = fields8()
    phi = box.embed(phi8, N0)
    psi = box.embed(psi8, N0)
    phi /= box.rms_grad(phi)
    psi /= box.rms_grad(psi)
    g0 = box.embed(g8, N0)
    g0 /= np.sqrt(np.sum(W_m(box, g0) * P))
    forc = []
    for f8 in forc8:
        f = box.embed(f8, N0)
        forc.append(f / box.rms_grad(f))
    for case in CASES:
        clock = timer()
        out = run_case(box, case, phi, psi, g0, forc)
        log_run("d05_breakers.py", f"frozen-line hypotheses test: {case}", N, M, int(round(TOUT[-1] / DT)), clock(),
                dict(dt=DT, T=TOUT[-1]))
        res[(N, case)] = {t: relerr(out[t][0], out[t][1]) for t in TOUT}
        res[(N, case)]["rms_psi_end"] = out["rms_psi_end"]
print(f"NL vs frozen-line prediction, sum_m |W_NL - W_FR| / sum_m W_FR  (nu_v = {NU_V}, eta = {ETA_R}, "
      f"forcing amplitude {FAMP}, eta_g = {ETA_G}, eta_z = {ETA_Z})")
for case in CASES:
    for t in TOUT:
        e = [res[(N, case)][t] for N in NS]
        print(f"  {case:12s} t={t:5.3f}: " + "  ".join(f"N={N}: {x:.2e}" for N, x in zip(NS, e))
              + f"   (rms|grad Psi| at end, N={NS[-1]}: {res[(NS[-1], case)]['rms_psi_end']:.2f})")
for case in CASES:
    e = [res[(N, case)][0.125] for N in NS]
    conv = all(e[i + 1] < e[i] for i in range(len(e) - 1)) and e[-1] < 0.25 * e[0]
    if case in ("ideal", "visc_phi", "force_phi"):
        chk(f"{case}: hypotheses hold -> NL - FR converges with N (t = 0.125)", conv, " -> ".join(f"{x:.1e}" for x in e))
    else:
        plateau = e[-1] > 0.5 * e[0] and e[-1] > 20 * res[(NS[-1], 'ideal')][0.125]
        chk(f"{case}: hypotheses violated -> NL - FR does not converge (t = 0.125)", plateau,
            " -> ".join(f"{x:.1e}" for x in e) + f"; ideal control at N={NS[-1]}: {res[(NS[-1], 'ideal')][0.125]:.1e}")
chk.finish()
