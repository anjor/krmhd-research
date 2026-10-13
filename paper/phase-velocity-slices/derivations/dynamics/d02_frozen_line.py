"""d02: numerical test of the frozen-line theorem with GANDALF's own operators.  Ideal nonlinear KRMHD and the
frozen-line prediction (field lines frozen at t = 0, no flow) are integrated from the same physical initial fields at
N = 16, 24, 32, 48, without and with hypercollisions, and the script checks that volume-integrated single-point
statistics of g converge to the prediction as N grows while W(k_perp), W(|k_z|) and the k_z-only Hermite flux do not.
It supports Section 6 (frozen-line theorem and its corollary for single-point statistics).  The table of results is
saved to paper/phase-velocity-slices/data/derivations/dynamics/d02_results.npy (git-ignored).

Theorem (continuum): if the induction equation is ideal, d_t Psi + {Phi, Psi} = d_z Phi, and g obeys
    d_t g + {Phi, g} + s grad_par(A g) = -C g     (C m-diagonal, local in x; no spatial dissipation of g),
then g(t) = U(t) exp[-t (s A grad_par(0) + C)] g(0), with U(t) the E x B relabelling (volume preserving).
Corollary: every volume-integrated single-point statistic of g equals that of the LINEAR problem
    d_t h = -s grad_par(0) (A h) - C h        (field lines frozen at t = 0, no flow).

Runs (all with GANDALF's _krmhd_rhs_jit, classical RK4, dt = 0.002, eta = 0):
  NL : full ideal KRMHD (z+- evolve nonlinearly, g passive);
  FR : frozen-line prediction, z+- held at (Psi0, -Psi0), i.e. Phi = 0, Psi = Psi(0);
  LZ : the usual k_z-only linear prediction (Phi = Psi = 0).
Initial fields are the same physical fields at every N (generated on an 8^3 grid and embedded), so the NL-FR
difference must converge to zero with N for single-point statistics, and must not for the others
(W(k_perp), the k_z spectrum, the k_z-based Hermite flux).
Cases: nu = 0 and nu = 20 with GANDALF's (m/M)^4 hypercollisions.

Usage: python d02_frozen_line.py [N list, default 16,24,32,48] [T, default 0.25]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from dcore import (Box, Checks, SQRT5, W_m, coll_rates, corr_matrix, flux_full, flux_kz, ladder, log_run,  # noqa: E402
                   make_rhs, make_rk4, maxrel, pweights, quartic_and_quantiles, relerr, slice_amps, slices,
                   timer, Wk_perp, Wk_z, data_dir)

NS = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [16, 24, 32, 48]
T = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
M, LAM, BETA, DT, AMP = 8, SQRT5, 1.0, 0.002, 1.0
S = np.sqrt(BETA)
A = ladder(M, LAM)
P = pweights(M, LAM)
U, V, WL = slices(M, LAM)
N0 = 8
TOUT = [0.062, 0.124, 0.25, 0.5]           # exact multiples of DT
TOUT = [t for t in TOUT if t <= T + 1e-12]


def initial_fields(seed: int = 1):
    """Fields on an 8^3 grid (n_perp in [1, 2.5], |n_z| <= 2), the same physical fields for every N."""
    b0 = Box(N0)
    rng = np.random.default_rng(seed)
    phi = b0.random_field(rng, (1, 2.5), 2)
    psi = b0.random_field(rng, (1, 2.5), 2)
    g = np.zeros((N0, N0, N0 // 2 + 1, M + 1), complex)
    g[..., 0] = b0.random_field(rng, (1, 2.5), 2)
    g[..., 1] = 0.5 * b0.random_field(rng, (1, 2.5), 2)
    return phi, psi, g


def stats(box: Box, g, psi) -> dict:
    g = np.asarray(g)
    a = slice_amps(g, WL)
    q4, qq = quartic_and_quantiles(box, g, ms=(0, 1, 2), pad=3)
    return dict(W=W_m(box, g), C=corr_matrix(box, a), Pi=flux_full(box, g, psi, S, A, P), q4=q4, qq=qq,
                Wkp=Wk_perp(box, g, P), Wkz=Wk_z(box, g, P), Pz=flux_kz(box, g, S, A, P))


SP_KEYS = ("W", "C", "Pi", "q4", "qq")      # single-point statistics (must converge NL -> FR)
NSP_KEYS = ("Wkp", "Wkz", "Pz")           # not single-point (must not)


def run(N: int, nu: float) -> dict:
    box = Box(N)
    phi8, psi8, g8 = initial_fields()
    phi = box.embed(phi8, N0)
    psi = box.embed(psi8, N0)
    phi *= AMP / box.rms_grad(phi)
    psi *= AMP / box.rms_grad(psi)
    g = box.embed(g8, N0)
    g /= np.sqrt(np.sum(W_m(box, g) * P))
    rhs_full, rhs_g = make_rhs(box, M, LAM, BETA)
    coll = coll_rates(M, nu, 4)
    step_full, step_frozen = make_rk4(rhs_full, rhs_g, coll, DT)
    zp, zm, gN = jnp.asarray(phi + psi), jnp.asarray(phi - psi), jnp.asarray(g)
    zpF, zmF = jnp.asarray(psi), jnp.asarray(-psi)
    zero = jnp.zeros_like(zp)
    gF = jnp.asarray(g)
    gZ = jnp.asarray(g)
    nsteps = int(round(T / DT))
    outs = {int(round(t / DT)): t for t in TOUT}
    res = {}
    clock = timer()
    for n in range(nsteps + 1):
        if n in outs and abs(outs[n] - 0.062) < 1e-12 and N == NS[-1] and nu == 0.0:
            padtest = []
            for pad in (2, 3, 4):
                _, qn = quartic_and_quantiles(box, gN, ms=(0, 1, 2), pad=pad)
                _, qf = quartic_and_quantiles(box, gF, ms=(0, 1, 2), pad=pad)
                padtest.append((pad, float(np.abs(qn - qf).max() / np.abs(qf).max())))
            res["padtest"] = padtest
        if n in outs:
            psiN = np.asarray(zp - zm) / 2
            res[outs[n]] = dict(NL=stats(box, gN, psiN), FR=stats(box, gF, psi), LZ=stats(box, gZ, 0 * psi),
                                eps=dict(NL=2 * float(np.sum(coll * P * W_m(box, gN))),
                                         FR=2 * float(np.sum(coll * P * W_m(box, gF))),
                                         LZ=2 * float(np.sum(coll * P * W_m(box, gZ)))))
        if n == nsteps:
            break
        zp, zm, gN = step_full(zp, zm, gN)
        gF = step_frozen(zpF, zmF, gF)
        gZ = step_frozen(zero, zero, gZ)
    wall = clock()
    log_run("d02_frozen_line.py", f"frozen-line theorem NL vs frozen (RK4 on GANDALF RHS), nu={nu}", N, M, nsteps,
            wall, dict(dt=DT, T=T, amp=AMP))
    # sanity: the Alfvenic flow is strong: chi-like ratio ||{Phi,g}|| / ||s d_z A g|| at t = 0
    br = box.bracket(phi, g[..., 0])
    res["chi"] = np.sqrt(box.norm2(br) / box.norm2(S * 1j * box.KZ * (g @ A.T)[..., 0]))
    res["wall"] = wall
    return res


def err(key: str, a: dict, b: dict) -> float:
    if key in ("Pi", "Pz"):
        return maxrel(a[key], b[key])
    if key == "C":
        return float(np.abs(a[key] - b[key]).max() / np.abs(b[key]).max())
    if key == "qq":
        return float(np.abs(a[key] - b[key]).max() / np.abs(b[key]).max())
    return relerr(a[key], b[key])


chk = Checks()
allres = {}
for nu in (0.0, 20.0):
    print(f"=== nu = {nu} (hyper_n = 4), M = {M}, Lambda = sqrt5, s = 1, rms|grad Phi| = rms|grad Psi| = {AMP}, "
          f"dt = {DT}, T = {T}", flush=True)
    for N in NS:
        allres[(nu, N)] = run(N, nu)
        r = allres[(nu, N)]
        print(f"  N = {N}: wall {r['wall']:.0f}s; ||{{Phi,g0}}||/||s d_z (Ag)_0|| at t=0: {r['chi']:.2f}", flush=True)
    if "padtest" in allres[(nu, NS[-1])]:
        pt = allres[(nu, NS[-1])]["padtest"]
        print(f"  quantile difference NL vs FR at N={NS[-1]}, t=0.062 vs sampling refinement: "
              + ", ".join(f"pad {p_}: {e_:.1e}" for p_, e_ in pt), flush=True)
        chk("quantile difference shrinks as the sampling grid is refined (sampling-limited, not dynamical)",
            pt[-1][1] < pt[0][1], ", ".join(f"{e_:.1e}" for _, e_ in pt))
    keys = SP_KEYS + ("eps",) + NSP_KEYS
    for t in TOUT:
        print(f"  t = {t}:  relative difference NL vs FR (frozen-line prediction) [NL vs LZ in brackets]", flush=True)
        for key in keys:
            row = []
            for N in NS:
                r = allres[(nu, N)][t]
                if key == "eps":
                    if nu == 0:
                        continue
                    e1 = abs(r["eps"]["NL"] - r["eps"]["FR"]) / r["eps"]["FR"]
                    e2 = abs(r["eps"]["NL"] - r["eps"]["LZ"]) / r["eps"]["LZ"]
                else:
                    e1 = err(key, r["NL"], r["FR"])
                    e2 = err(key, r["NL"], r["LZ"])
                row.append((N, e1, e2))
            if row:
                lab = {"W": "W(m)", "C": "C(j,j') slice corr.", "Pi": "Pi_m (full grad_par)", "q4": "<g_m^4>",
                       "qq": "quantiles g_m(x)", "eps": "eps_nu", "Wkp": "W(k_perp)  [not 1-pt]",
                       "Wkz": "W(|k_z|)   [not 1-pt]", "Pz": "Pi_m (k_z only) [not 1-pt]"}[key]
                print(f"    {lab:28s} " + "  ".join(f"N={N:2d}: {e1:8.1e} [{e2:7.1e}]" for N, e1, e2 in row), flush=True)
                allres[(nu, "tab", t, key)] = row
    # checks: convergence for single-point statistics at t <= 0.125; non-convergence for the others
    for t in [tt for tt in TOUT if tt <= 0.13]:
        for key in SP_KEYS + (("eps",) if nu > 0 else ()):
            row = allres[(nu, "tab", t, key)]
            e = [e1 for _, e1, _ in row]
            if key == "qq":
                # grid-sampled quantiles converge only algebraically (sampling of a continuous distribution on a
                # finite grid; see the pad test below), so require a decrease, not spectral accuracy
                ok = e[-1] < e[0] / 4 and e[-1] < e[-2]
                chk(f"nu={nu} t={t} quantiles: NL - FR decreases with N (sampling-limited)", ok,
                    " -> ".join(f"{x:.1e}" for x in e))
                continue
            ok = e[-1] < e[0] / 10 and e[-1] < (1e-3 if t > 0.07 else 1e-4)
            chk(f"nu={nu} t={t} {key}: NL - FR falls >10x from N={NS[0]} to N={NS[-1]} and is small", ok,
                " -> ".join(f"{x:.1e}" for x in e))
            lz = [e2 for _, _, e2 in row]
            if key in ("W", "C", "Pi"):
                chk(f"nu={nu} t={t} {key}: k_z-only linear prediction is far off", min(lz) > 30 * e[-1],
                    f"NL-LZ = {min(lz):.1e} .. {max(lz):.1e}")
        for key in NSP_KEYS:
            row = allres[(nu, "tab", t, key)]
            e = [e1 for _, e1, _ in row]
            chk(f"nu={nu} t={t} {key}: NL - FR does NOT vanish (not a single-point statistic)", e[-1] > 1e-2,
                " -> ".join(f"{x:.1e}" for x in e))
np.save(os.path.join(data_dir(), "d02_results.npy"), {str(k): v for k, v in allres.items()}, allow_pickle=True)
chk.finish()
