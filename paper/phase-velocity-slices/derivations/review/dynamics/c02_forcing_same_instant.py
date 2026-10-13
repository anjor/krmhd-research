"""c02: the forced corollary of the frozen-line theorem when the g forcing is correlated with the Alfvenic forcing at
the same instant, a case d04 does not test.  Two injections with random signs coincide with Phi kicks that carry the
same signs and change the subsequent flow at O(1) (RK4 on GANDALF's right-hand side; run_all.sh uses N = 16, 24, 32,
48), and the script checks that the sign-averaged W(m) equals the averaged sum of frozen-line responses with an error
that falls with N, while an anticipating control fails at an N-independent level.  It supports Section 6 (corollary
for white forcing, and a convergent test of forcing of Phi).

The corollary states that for white, zero-mean, non-anticipating g forcing, E W(m, t) is the sum of the frozen-line
responses of the injections, and that a same-instant correlation with the Alfvenic forcing is allowed.  d04 only uses
injections that are independent of the flow (the Alfvenic history is the same for all sign patterns).

Test (exact, finite ensemble).  Two injections F_0, F_1 at t_0 = 0 and t_1 = 0.06 with random signs eps_0, eps_1; at
the same instants Phi receives a kick eps_k * phi_k (z+ and z- kicked equally, Psi unforced, induction ideal), strong
enough to change the subsequent flow at O(1).  So the flow after t_k depends on eps_k: the g forcing is perfectly
correlated with the Alfvenic forcing at the same instant.  For each of the 4 sign patterns the single-injection
responses y_0, y_1 are integrated in that pattern's flow (RK4 on GANDALF's right-hand side, eta = 0).  F_0 and F_1
have the same spatial pattern (random signs make the forcing white all the same), so the cross term is O(1).
  A  mean over the 4 patterns of W(m, T) of the forced run  =  mean over patterns of the sum of the frozen-line
     responses (each from its injection time, field lines frozen at that time), up to an error that falls with N;
  B  the cross term <y_0, y_1>_m does not depend on eps_1 (although the Eulerian fields after t_1 do), which is why
     the sign average removes it;
  C  negative control ("anticipating" forcing): the kick that carries eps_1 is applied at t = 0 < t_1, so the flow
     before the injection depends on its sign; then <y_0, y_1> depends on eps_1 and the mean formula fails at an
     N-independent level.
Usage: python c02_forcing_same_instant.py [N list, default 16,24,32]
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from c00_core import Checks, Grid, SQRT5, clock, log_run, make_rhs, mode_field, pw, random_modes  # noqa: E402

NS = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [16, 24, 32]
M, LAM, BETA = 8, SQRT5, 1.0
P = pw(M, LAM)
DT = 0.0025
T_INJ = (0.0, 0.06)
T_END = 0.15
KICK = 1.0
rng = np.random.default_rng(7)
md_phi, md_psi = random_modes(rng), random_modes(rng)
md_kick = [random_modes(rng) for _ in T_INJ]
md_F0 = (random_modes(rng), random_modes(rng))
md_F = [md_F0 for _ in T_INJ]      # the same spatial pattern at both injections (independent signs: still white),
                                   # so that the cross term <y0, y1> is O(1) and the test is sharp
PROTOCOLS = {"same_instant": {0: 0.0, 1: 0.06}, "anticipating": {0: 0.0, 1: 0.0}}   # time of the kick carrying eps_k


def Wm(grid, g):
    return np.einsum("zyxm->m", grid.W[..., None] * np.abs(np.asarray(g)) ** 2) / grid.N ** 6


def cross(grid, a, b):
    return np.einsum("zyxm->m", grid.W[..., None] * np.real(np.conj(np.asarray(a)) * np.asarray(b))) / grid.N ** 6


chk = Checks()
summary = {}
for N in NS:
    grid = Grid(N)
    phi0 = mode_field(grid, *md_phi)
    psi0 = mode_field(grid, *md_psi)
    phi0 *= 1.0 / grid.rms_grad(phi0)
    psi0 *= 1.0 / grid.rms_grad(psi0)
    kicks = []
    for md in md_kick:
        f = mode_field(grid, *md)
        kicks.append(KICK * f / grid.rms_grad(f))
    Fs = []
    for mda, mdb in md_F:
        F = np.zeros((N, N, N // 2 + 1, M + 1), complex)
        F[..., 0] = mode_field(grid, *mda)
        F[..., 1] = 0.7 * mode_field(grid, *mdb)
        Fs.append(F / np.sqrt(np.sum(P * Wm(grid, F))))
    rhs = make_rhs(grid, M, LAM, BETA)

    @jax.jit
    def stages(zp, zm):
        g0 = jnp.zeros(zp.shape + (M + 1,), complex)
        k1 = rhs(zp, zm, g0)[:2]
        z2 = (zp + 0.5 * DT * k1[0], zm + 0.5 * DT * k1[1])
        k2 = rhs(*z2, g0)[:2]
        z3 = (zp + 0.5 * DT * k2[0], zm + 0.5 * DT * k2[1])
        k3 = rhs(*z3, g0)[:2]
        z4 = (zp + DT * k3[0], zm + DT * k3[1])
        k4 = rhs(*z4, g0)[:2]
        znew = (zp + DT / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), zm + DT / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))
        return (zp, zm), z2, z3, z4, znew

    @jax.jit
    def gstep(z1, z2, z3, z4, g):
        k1 = rhs(*z1, g)[2]
        k2 = rhs(*z2, g + 0.5 * DT * k1)[2]
        k3 = rhs(*z3, g + 0.5 * DT * k2)[2]
        k4 = rhs(*z4, g + DT * k3)[2]
        return g + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    @jax.jit
    def frozen_step(psi, h):
        z1 = (psi, -psi)
        return gstep(z1, z1, z1, z1, h)

    nsteps = int(round(T_END / DT))
    inj_steps = [int(round(t / DT)) for t in T_INJ]
    for proto, kick_times in PROTOCOLS.items():
        kick_steps = {k: int(round(t / DT)) for k, t in kick_times.items()}
        data = {}
        tick = clock()
        nrhs = 0
        for eps in itertools.product((1, -1), repeat=len(T_INJ)):
            zp, zm = jnp.asarray(phi0 + psi0), jnp.asarray(phi0 - psi0)
            ys = [None] * len(T_INJ)
            psi_at = {}
            for n in range(nsteps + 1):
                for k in range(len(T_INJ)):
                    if n == kick_steps[k]:                         # Phi kick carrying eps_k (Psi unchanged)
                        d = jnp.asarray(eps[k] * kicks[k])
                        zp, zm = zp + d, zm + d
                for k, ns_ in enumerate(inj_steps):
                    if n == ns_:
                        ys[k] = jnp.asarray(Fs[k])
                        psi_at[k] = np.asarray(zp - zm) / 2
                if n == nsteps:
                    break
                z1, z2, z3, z4, znew = stages(zp, zm)
                ys = [gstep(z1, z2, z3, z4, y) if y is not None else None for y in ys]
                zp, zm = znew
                nrhs += 4 * (1 + sum(y is not None for y in ys))
            fr = []
            for k, ns_ in enumerate(inj_steps):
                h = jnp.asarray(Fs[k])
                ps = jnp.asarray(psi_at[k])
                for _ in range(nsteps - ns_):
                    h = frozen_step(ps, h)
                    nrhs += 4
                fr.append(np.asarray(h))
            data[eps] = dict(y=[np.asarray(y) for y in ys], fr=fr, psi1=psi_at[1])
        wall = tick()
        log_run("c02_forcing_same_instant.py", f"forced corollary with sign-dependent Phi kicks ({proto}), 4 sign patterns",
                N, M, nsteps * 4, wall, dt=DT, T=T_END, rhs_calls=nrhs)
        pats = list(data)
        Wtot = {e: Wm(grid, sum(e[k] * data[e]["y"][k] for k in range(len(T_INJ)))) for e in pats}
        mean_W = sum(Wtot.values()) / len(pats)
        pred = sum(sum(Wm(grid, f) for f in data[e]["fr"]) for e in pats) / len(pats)
        errA = float(np.sum(P * np.abs(mean_W - pred)) / np.sum(P * pred))
        c01 = {e: cross(grid, data[e]["y"][0], data[e]["y"][1]) for e in pats}
        cscale = np.mean([np.sum(P * np.abs(c)) for c in c01.values()])
        dB = max(np.sum(P * np.abs(c01[(e0, 1)] - c01[(e0, -1)])) for e0 in (1, -1)) / cscale
        # how different are the Eulerian responses between eps_1 = +1 and -1 (flow after t_1 differs)?
        ydiff = max(np.sqrt(np.sum(P * Wm(grid, data[(e0, 1)]["y"][0] - data[(e0, -1)]["y"][0])))
                    / np.sqrt(np.sum(P * Wm(grid, data[(e0, 1)]["y"][0]))) for e0 in (1, -1))
        psidiff = max(np.sqrt(grid.avg(data[(e0, 1)]["psi1"] - data[(e0, -1)]["psi1"], data[(e0, 1)]["psi1"] - data[(e0, -1)]["psi1"]))
                      / np.sqrt(grid.avg(data[(e0, 1)]["psi1"], data[(e0, 1)]["psi1"])) for e0 in (1, -1))
        pathwise = max(float(np.sum(P * np.abs(Wtot[e] - sum(Wm(grid, y) for y in data[e]["y"]))))
                       / float(np.sum(P * sum(Wm(grid, y) for y in data[e]["y"]))) for e in pats)
        summary[(proto, N)] = dict(errA=errA, dB=dB, ydiff=ydiff, psidiff=psidiff, pathwise=pathwise, cscale=cscale)
        print(f"N = {N:2d} {proto:13s} (wall {wall:.0f}s): |mean_eps W - mean_eps sum frozen| / W = {errA:.2e};  "
              f"cross term <y0,y1> change under eps_1 -> -eps_1: {dB:.2e} of its size;  Eulerian y_0 change under "
              f"eps_1 -> -eps_1: {ydiff:.2f};  Psi(t_1) change: {psidiff:.2f};  pathwise cross terms 2|<y0,y1>| = {pathwise:.2f} "
              f"of W(y0) + W(y1)",
              flush=True)

for proto in PROTOCOLS:
    eA = [summary[(proto, N)]["errA"] for N in NS]
    eB = [summary[(proto, N)]["dB"] for N in NS]
    if proto == "same_instant":
        chk("A same-instant correlated forcing: sign-averaged W(m) = sum of frozen-line responses (error falls with N)",
            all(eA[i + 1] < eA[i] for i in range(len(eA) - 1)) and eA[-1] < eA[0] / 5 and eA[-1] < 5e-3,
            " -> ".join(f"{x:.1e}" for x in eA))
        chk("B cross term <y0, y1> independent of eps_1 (falls with N) although the Eulerian fields change at O(1)",
            all(eB[i + 1] < eB[i] for i in range(len(eB) - 1)) and eB[-1] < 1e-2
            and min(summary[(proto, N)]["ydiff"] for N in NS) > 0.05,
            " -> ".join(f"{x:.1e}" for x in eB) + f"; Eulerian change {summary[(proto, NS[-1])]['ydiff']:.2f}")
        chk("   pathwise cross terms are O(1) (the average is what removes them)", summary[(proto, NS[-1])]["pathwise"] > 0.02,
            f"{summary[(proto, NS[-1])]['pathwise']:.2f}")
    else:
        chk("C anticipating control: Psi(t_1) depends on eps_1", min(summary[(proto, N)]["psidiff"] for N in NS) > 0.05,
            f"{summary[(proto, NS[-1])]['psidiff']:.2f}")
        eB_si = summary[("same_instant", NS[-1])]["dB"]
        eA_si = summary[("same_instant", NS[-1])]["errA"]
        chk("C anticipating control: the cross term depends on eps_1 and the mean formula fails, both at an N-independent "
            "level far above the same-instant residuals",
            max(eB) / min(eB) < 2 and min(eB) > 30 * eB_si and max(eA) / min(eA) < 2 and min(eA) > 30 * eA_si,
            "cross-term change " + " -> ".join(f"{x:.1e}" for x in eB) + f" (same-instant {eB_si:.1e}); mean-formula error "
            + " -> ".join(f"{x:.1e}" for x in eA) + f" (same-instant {eA_si:.1e})")
chk.finish()
