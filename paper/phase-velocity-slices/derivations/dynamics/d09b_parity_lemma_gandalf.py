"""d09b: the parity lemma in GANDALF runs.  For the initial conditions g(0) = f (e_0 + r e_1) and f (e_0 - r e_1),
the script checks that W(m, t) and the field-aligned flux Pi_m(t) are identical for +r and -r in the frozen-field
problem (exactly) and in ideal nonlinear runs (RK4 on GANDALF's right-hand side, converging with N = 16, 24, 32, 48),
and that the lemma does not extend to the mirror-odd Gamma, to a k_z-odd correlation, to runs with spatial damping
of g, or to fourth-order statistics.  It supports Section 7 (parity lemma).

Lemma: under frozen-line dynamics with m-diagonal collisions, the initial conditions g(0) = f (e_0 + r e_1) and
f (e_0 - r e_1) (one real pattern f in both moments) give the same W(m, t) and the same field-aligned flux Pi_m(t),
for every Alfvenic realisation.

PL1 FR (static frozen field, no flow): the lemma holds exactly in the semi-discrete system, because the discrete
    grad_par(0) = P(d_z + {Psi0, .}) is a real antisymmetric operator; checked to round-off at every N.
PL2 NL (full ideal KRMHD, GANDALF RHS, RK4): W_+ - W_- and Pi_+ - Pi_- converge to zero with N at times where the
    frozen-line theorem is resolved (t = 0.062, 0.124), while the mirror-odd Gamma differs between +r and -r.
PL3 the k_z-odd correlation g_1 = +-r H f, H = i sgn(k_z) (cases sin+ and sin-), changes W(m) at a finite level.
PL4 perpendicular damping of g (-4e-6 k_perp^4 g) breaks the lemma at a finite level.
PL5 the lemma concerns quadratic statistics: the fourth moment <g_m^4> has a part linear in r, in FR and NL alike
    (finite, N-independent).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from dcore import (Box, Checks, SQRT5, W_m, flux_full, ladder, log_run, make_rhs, make_rk4, pweights,  # noqa: E402
                   slices, timer)

chk = Checks()
NS = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [16, 24, 32, 48]
M, Lam, beta, r, N0, dt = 8, SQRT5, 1.0, 0.8, 8, 0.002
TOUT = (0.062, 0.124)          # exact multiples of dt
A = ladder(M, Lam)
p = pweights(M, Lam)
u, V, WL = slices(M, Lam)
b0 = Box(N0)
rng8 = np.random.default_rng(21)
zp8 = b0.random_field(rng8, (1, 2.5), 2)
zm8 = b0.random_field(rng8, (1, 2.5), 2)
f8 = b0.random_field(rng8, (1, 2.5), 2)


def stats(bx, g, psi):
    a = g @ WL.T
    return dict(W=W_m(bx, g), Pi=flux_full(bx, g, psi, 1.0, A, p),
                G=float(np.sum(u * np.array([bx.norm2(a[..., j]) for j in range(M + 1)]))),
                q4=np.array([float(np.mean(bx.to_real(g[..., m], 2) ** 4)) for m in (0, 1, 2)]))


res = {}
for N in NS:
    bx = Box(N)
    zp_ = bx.embed(zp8, N0)
    zm_ = bx.embed(zm8, N0)
    zp_ *= 2.0 / bx.rms_grad(zp_)
    zm_ *= 0.5 / bx.rms_grad(zm_)
    psi0 = (zp_ - zm_) / 2
    f = bx.embed(f8, N0)
    f /= np.sqrt(bx.norm2(f))
    Hf = 1j * np.sign(bx.KZ) * f
    rhs_full, rhs_g = make_rhs(bx, M, Lam, beta)
    step_full, step_frozen = make_rk4(rhs_full, rhs_g, np.zeros(M + 1), dt)
    damp = jnp.asarray(np.exp(-4e-6 * bx.KP2 ** 2 * dt)[..., None])
    cases = [("cos+", r * f, "NL"), ("cos-", -r * f, "NL"), ("cos+", r * f, "FR"), ("cos-", -r * f, "FR")]
    if N <= 32:
        cases += [("sin+", r * Hf, "NL"), ("sin-", -r * Hf, "NL"), ("cos+damp", r * f, "NL"), ("cos-damp", -r * f, "NL")]
    out = {}
    for name, g1, kind in cases:
        g0 = np.zeros((N, N, N // 2 + 1, M + 1), complex)
        g0[..., 0] = f
        g0[..., 1] = g1
        zpj, zmj, gj = jnp.asarray(zp_), jnp.asarray(zm_), jnp.asarray(g0)
        zpF, zmF = jnp.asarray(psi0), jnp.asarray(-psi0)
        clock = timer()
        nmax = int(round(TOUT[-1] / dt))
        for n in range(1, nmax + 1):
            if kind == "NL":
                zpj, zmj, gj = step_full(zpj, zmj, gj)
                if "damp" in name:
                    gj = gj * damp
            else:
                gj = step_frozen(zpF, zmF, gj)
            for t in TOUT:
                if n == int(round(t / dt)):
                    psin = np.asarray(zpj - zmj) / 2 if kind == "NL" else psi0
                    out[(name, kind, t)] = stats(bx, np.asarray(gj), psin)
        log_run("d09b_parity_lemma_gandalf.py", f"parity lemma, ideal flow: {name} {kind}", N, M, nmax, clock(),
                dict(dt=dt, T=TOUT[-1]))

    def d(key, a_, b_, kind, t):
        x, y = out[(a_, kind, t)][key], out[(b_, kind, t)][key]
        return float(np.abs(np.asarray(x) - np.asarray(y)).sum() / np.abs(np.asarray(x) + np.asarray(y)).sum())

    for t in TOUT:
        row = dict(W_FR=d("W", "cos+", "cos-", "FR", t), Pi_FR=d("Pi", "cos+", "cos-", "FR", t),
                   W_NL=d("W", "cos+", "cos-", "NL", t), Pi_NL=d("Pi", "cos+", "cos-", "NL", t),
                   q4_FR=d("q4", "cos+", "cos-", "FR", t), q4_NL=d("q4", "cos+", "cos-", "NL", t),
                   G=(out[("cos+", "NL", t)]["G"], out[("cos-", "NL", t)]["G"]))
        if N <= 32:
            row.update(W_sin=d("W", "sin+", "sin-", "NL", t), W_damp=d("W", "cos+damp", "cos-damp", "NL", t))
        res[(N, t)] = row
        print(f"  N={N:2d} t={t:6.4f}: |X(+r)-X(-r)|/|X(+r)+X(-r)|  W: FR {row['W_FR']:.1e} NL {row['W_NL']:.1e};  "
              f"Pi: FR {row['Pi_FR']:.1e} NL {row['Pi_NL']:.1e};  <g^4>: FR {row['q4_FR']:.1e} NL {row['q4_NL']:.1e};  "
              f"Gamma(+r), Gamma(-r) = {row['G'][0]:+.4f}, {row['G'][1]:+.4f}"
              + (f";  W sin-type {row['W_sin']:.1e}, W with g damping {row['W_damp']:.1e}" if N <= 32 else ""), flush=True)

chk("PL1 frozen field: W(m) and Pi_m identical for +r and -r (exact lemma, every N)",
    max(max(res[k]["W_FR"], res[k]["Pi_FR"]) for k in res) < 1e-12, f"max {max(max(res[k]['W_FR'], res[k]['Pi_FR']) for k in res):.1e}")
for t in TOUT:
    eW = [res[(N, t)]["W_NL"] for N in NS]
    eP = [res[(N, t)]["Pi_NL"] for N in NS]
    # at t = 0.062 the run is well resolved (spectral convergence expected); at t = 0.124 the passive field is close
    # to the grid scale in this flow, so only a monotone decrease by > 5x is required there
    fac = 10 if t < 0.1 else 5
    mono = lambda e: all(e[i + 1] < e[i] for i in range(len(e) - 1))  # noqa: E731
    chk(f"PL2 t={t}: NL W_+ - W_- decreases monotonically and falls >{fac}x from N={NS[0]} to N={NS[-1]}",
        mono(eW) and eW[-1] < eW[0] / fac, " -> ".join(f"{x:.1e}" for x in eW))
    chk(f"PL2 t={t}: NL Pi_+ - Pi_- decreases monotonically and falls >{fac}x from N={NS[0]} to N={NS[-1]}",
        mono(eP) and eP[-1] < eP[0] / fac, " -> ".join(f"{x:.1e}" for x in eP))
    G = res[(NS[-1], t)]["G"]
    chk(f"PL2 t={t}: Gamma differs between +r and -r (mirror-odd)", abs(G[0] - G[1]) > 0.1 * (abs(G[0]) + abs(G[1])),
        f"{G[0]:+.4f} vs {G[1]:+.4f}")
small = [N for N in NS if N <= 32]
chk("PL3 sin (k_z-odd) correlation changes W(m) at a finite level", min(res[(N, 0.124)]["W_sin"] for N in small) > 1e-2,
    ", ".join(f"{res[(N, 0.124)]['W_sin']:.1e}" for N in small))
chk("PL4 perpendicular damping of g breaks the lemma at a finite level",
    res[(small[-1], 0.124)]["W_damp"] > 30 * res[(NS[-1], 0.124)]["W_NL"],
    ", ".join(f"{res[(N, 0.124)]['W_damp']:.1e}" for N in small))
q = [res[(N, 0.124)]["q4_FR"] for N in NS]
chk("PL5 <g_m^4> (fourth order, outside the lemma) has a part linear in r (finite, N-independent) in the frozen "
    "problem",
    min(q) > 1e-3 and max(q) / min(q) < 2, ", ".join(f"{x:.1e}" for x in q))
chk("PL5 ... and in the NL runs it exceeds the W_+ - W_- residual by > 30x at the largest N",
    res[(NS[-1], 0.124)]["q4_NL"] > 30 * res[(NS[-1], 0.124)]["W_NL"],
    f"{res[(NS[-1], 0.124)]['q4_NL']:.1e} vs {res[(NS[-1], 0.124)]['W_NL']:.1e}")
chk.finish()
