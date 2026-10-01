# Provenance: independent critic context given only the AS2018 (3.10) Hermite equation and
# the claim that it grows exponentially as a truncated-Hermite SDE. 2026-10-01. See claims C11.
# Run: uv run python studies/04-phase-space-helicity/derivations/critic_as2018.py (~50 s)

"""Critic check: is the AS2018 (JPP 84, eq. 3.10) Hermite hierarchy with a white-noise
phi(x,t) exponentially unstable in W = sum |g_{k,m}|^2 ?

1D periodic box, k = 2 pi n, Hermite m = 0..M, complex g_{k,m}.
  d_t g_{k,m} + i k [ s(m+1) g_{k,m+1} + s(m) g_{k,m-1} ]
             + s(m) sum_p i p phi_p g_{k-p,m-1} = 0        (nu = 0, f = 0 here)
with s(m) = sqrt(m/2), g_{M+1} = 0, phi real (phi_{-p} = phi_p^*), Gaussian white in time,
<phi_p phi_{p'}> = 2 kappa delta_{p,-p'} delta(t-t') for |n_p| in {1,2}.

Split step: exact linear propagation per k, then stochastic kick
   g += sum_p N_p g  dW_p ,   N_p g = -i p s(m) g_{k-p,m-1},   dW_p ~ sqrt(kappa dt) (a + i b).
Ito  : kick evaluated at pre-kick state.
Heun : Stratonovich predictor-corrector, dg = (1/2)[N(dW) g + N(dW)(g + N(dW) g)].
numpy only.
"""
import time
import numpy as np

rng = np.random.default_rng(1)
N = 32                       # Fourier modes, n = -N/2 .. N/2-1
PNS = [1, 2]                 # forced |n_p|
E_ENS = 6                    # realisations per (kappa, interpretation)
KAPPAS = [1e-4, 4e-4, 1.6e-3]
DTS = {1e-3: 3.0, 3e-4: 1.2}  # dt -> total integration time
nvec = np.arange(-N // 2, N // 2)
kvec = 2 * np.pi * nvec


def S_of(kappa: float) -> float:
    """S = sum_p p^2 kappa_p over all p with kappa_p != 0 (both signs of p)."""
    return kappa * sum((2 * np.pi * n) ** 2 for n in PNS) * 2


# ---------------------------------------------------------------------------
# 1. Analytic/algebraic check of antisymmetry of the nonlinear operator
# ---------------------------------------------------------------------------
def shift_k(g: np.ndarray, s: int) -> np.ndarray:
    """out[..., n, :] = g[..., n - s, :], zero outside the resolved k range (no aliasing)."""
    out = np.zeros_like(g)
    if s > 0:
        out[..., s:, :] = g[..., :-s, :]
    elif s < 0:
        out[..., :s, :] = g[..., -s:, :]
    else:
        out[...] = g
    return out


def op_raise(acc: np.ndarray, M: int) -> np.ndarray:
    """(D_raise acc)_m = s(m) acc_{m-1}  (the AS2018 form: g_m receives from g_{m-1} only)."""
    m = np.arange(M + 1)
    sm = np.sqrt(m / 2.0)
    out = np.zeros_like(acc)
    out[..., 1:] = sm[1:] * acc[..., :-1]
    return out


def op_antisym(acc: np.ndarray, M: int) -> np.ndarray:
    """(D_anti acc)_m = s(m) acc_{m-1} - s(m+1) acc_{m+1}  (matrix of d_v in orthonormal Hermite basis)."""
    m = np.arange(M + 1)
    sm = np.sqrt(m / 2.0)
    out = np.zeros_like(acc)
    out[..., 1:] = sm[1:] * acc[..., :-1]
    out[..., :-1] -= sm[1:] * acc[..., 1:]
    return out


def dense_matrix(op, Nk: int, M: int, s: int, p: float) -> np.ndarray:
    """Dense matrix of g -> -i p * op(shift_k(g, s)) on the (k, m) vector space (Nk*(M+1))."""
    dim = Nk * (M + 1)
    A = np.zeros((dim, dim), complex)
    for j in range(dim):
        e = np.zeros((Nk, M + 1), complex)
        e.flat[j] = 1.0
        A[:, j] = (-1j * p * op(shift_k(e, s), M)).ravel()
    return A


print("=== 1. Antisymmetry of the nonlinear operator N_p (dense matrix, N=8, M=6) ===")
Nk_small, M_small = 8, 6
for name, op in [("AS2018 raising-only", op_raise), ("antisymmetrised d_v", op_antisym)]:
    worst = 0.0
    nrm = 0.0
    for n_p in [1, 2]:
        p = 2 * np.pi * n_p
        Np = dense_matrix(op, Nk_small, M_small, n_p, p)
        Nmp = dense_matrix(op, Nk_small, M_small, -n_p, -p)
        # phi real => phi_{-p} = phi_p^*; realisation operator is sum_p phi_p N_p.
        # Norm preservation per realisation <=> N_p^dagger = -N_{-p} for every p.
        worst = max(worst, np.linalg.norm(Np.conj().T + Nmp))
        nrm = max(nrm, np.linalg.norm(Np))
    print(f"  {name:22s}: max_p ||N_p^dag + N_{{-p}}|| / ||N_p|| = {worst / nrm:.3e}")
print("  -> raising-only operator is NOT antisymmetric (it is nilpotent, strictly lower-triangular in m);")
print("     antisymmetrised operator IS antisymmetric to machine precision.\n")


# ---------------------------------------------------------------------------
# 2./3. Stochastic integration
# ---------------------------------------------------------------------------
def linear_propagator(M: int, dt: float) -> np.ndarray:
    m = np.arange(M + 1)
    H = np.zeros((M + 1, M + 1))
    off = np.sqrt((m[:-1] + 1) / 2.0)
    H[np.arange(M), np.arange(1, M + 1)] = off
    H[np.arange(1, M + 1), np.arange(M)] = off
    lam, V = np.linalg.eigh(H)
    # P_k = V exp(-i k lam dt) V^T, shape (N, M+1, M+1)
    ph = np.exp(-1j * kvec[:, None] * lam[None, :] * dt)
    return np.einsum('ij,kj,lj->kil', V, ph, V)


def kick(g: np.ndarray, dW: dict, op, M: int) -> np.ndarray:
    """sum_p (-i p dW_p) op( g_{k-p} ).  dW[n_p] has shape (B,) complex, dW[-n_p] = conj."""
    acc = np.zeros_like(g)
    for n_p, w in dW.items():
        p = 2 * np.pi * n_p
        acc += (-1j * p * w)[:, None, None] * shift_k(g, n_p)
    return op(acc, M)


def run(M: int, dt: float, T: float, op, kappas, ito_and_heun=True, seed=0):
    """Batch axis: [kappa][interp (Ito, Heun)][ensemble].  Returns t, lnW(B, nt)."""
    rng = np.random.default_rng(seed)
    nK = len(kappas)
    nI = 2
    B = nK * nI * E_ENS
    P = linear_propagator(M, dt)
    g = rng.standard_normal((B, N, M + 1)) + 1j * rng.standard_normal((B, N, M + 1))
    g /= np.sqrt(np.sum(np.abs(g) ** 2, axis=(1, 2)))[:, None, None]
    logW = np.zeros(B)
    nsteps = int(round(T / dt))
    rec_every = max(1, nsteps // 300)
    ts, lnWs = [], []
    sqk = np.repeat(np.sqrt(np.asarray(kappas) * dt), nI * E_ENS)  # per batch member
    is_heun = np.tile(np.repeat([False, True], E_ENS), nK)
    for it in range(nsteps):
        g = np.einsum('kij,bkj->bki', P, g)
        # same unit noise for Ito and Heun members of the same ensemble slot
        z = rng.standard_normal((2, len(PNS), E_ENS))
        dW = {}
        for ip, n_p in enumerate(PNS):
            zc = (z[0, ip] + 1j * z[1, ip])
            w = np.tile(zc, nK * nI) * sqk
            dW[n_p] = w
            dW[-n_p] = np.conj(w)
        k1 = kick(g, dW, op, M)
        if ito_and_heun:
            k2 = kick(g + k1, dW, op, M)
            dg = np.where(is_heun[:, None, None], 0.5 * (k1 + k2), k1)
        else:
            dg = k1
        g = g + dg
        W = np.sum(np.abs(g) ** 2, axis=(1, 2))
        logW += np.log(W)
        g /= np.sqrt(W)[:, None, None]
        if it % rec_every == 0:
            ts.append((it + 1) * dt)
            lnWs.append(logW.copy())
    return np.array(ts), np.array(lnWs).T, is_heun


def fit_rates(t, lnW):
    """Slope of lnW over the last half of the record, per batch member."""
    sel = t >= 0.5 * t[-1]
    A = np.vstack([t[sel], np.ones(sel.sum())]).T
    return np.linalg.lstsq(A, lnW[:, sel].T, rcond=None)[0][0]


def summarize(label, t, lnW, is_heun, kappas, M):
    rates = fit_rates(t, lnW)
    nI = 2
    rows = []
    for ik, kap in enumerate(kappas):
        S = S_of(kap)
        for ii, interp in enumerate(["Ito ", "Heun"]):
            sl = slice((ik * nI + ii) * E_ENS, (ik * nI + ii + 1) * E_ENS)
            r = rates[sl]
            # growth of ln <W>_ens (log-sum-exp) over the same window
            lse = np.log(np.mean(np.exp(lnW[sl] - lnW[sl].max(axis=0)), axis=0)) + lnW[sl].max(axis=0)
            r_mean = fit_rates(t, lse[None, :])[0]
            rows.append((kap, interp, r.mean(), r.std(), r_mean, S, r.mean() / (M * S)))
    return rows


t0 = time.time()
print("=== 2. AS2018 raising-only nonlinearity, nu = 0, no forcing, W(t) growth rates ===")
print("    gamma = d<lnW>/dt (ensemble mean +- std over %d realisations), gamma_m = d ln<W>/dt," % E_ENS)
print("    S = sum_p p^2 kappa_p (both signs of p) = 40 pi^2 kappa;  fit window = last half of run.")
hdr = f"{'M':>3} {'dt':>7} {'T':>5} {'kappa':>8} {'interp':>6} {'gamma':>9} {'+-':>7} {'gamma_m':>9} {'M*S':>8} {'gamma/(M S)':>12}"
print(hdr)
results = {}
for M in [16, 32]:
    for dt, T in DTS.items():
        t, lnW, is_heun = run(M, dt, T, op_raise, KAPPAS, seed=10 + M)
        for kap, interp, r, rs, rm, S, ratio in summarize("", t, lnW, is_heun, KAPPAS, M):
            results[(M, dt, kap, interp.strip())] = r
            print(f"{M:3d} {dt:7.0e} {T:5.1f} {kap:8.1e} {interp:>6} {r:9.3f} {rs:7.3f} {rm:9.3f} {M * S:8.3f} {ratio:12.3f}")
print(f"  [elapsed {time.time() - t0:.1f} s]\n")

print("=== 2b. Scaling summary (Ito, dt=1e-3): gamma ratios ===")
for M in [16, 32]:
    g1, g4, g16 = (results[(M, 1e-3, k, "Ito")] for k in KAPPAS)
    print(f"  M={M:2d}: gamma(4k)/gamma(k) = {g4 / g1:.2f}, gamma(16k)/gamma(4k) = {g16 / g4:.2f}   (claim: linear in kappa -> 4.00)")
for kap in KAPPAS:
    print(f"  kappa={kap:.1e}: gamma(M=32)/gamma(M=16) = {results[(32, 1e-3, kap, 'Ito')] / results[(16, 1e-3, kap, 'Ito')]:.2f}   (claim: linear in M -> 2.00)")
for M in [16, 32]:
    for kap in KAPPAS:
        a, b = results[(M, 1e-3, kap, "Ito")], results[(M, 3e-4, kap, "Ito")]
        c, d = results[(M, 1e-3, kap, "Heun")], results[(M, 3e-4, kap, "Heun")]
        print(f"  M={M:2d} kappa={kap:.1e}: Ito gamma(dt=1e-3)/gamma(dt=3e-4) = {a / b:.2f};  Heun = {c / d:.2f};  Heun/Ito (dt=1e-3) = {c / a:.2f}")
print()

t0 = time.time()
print("=== 3. Antisymmetrised d_v (s(m) g_{m-1} - s(m+1) g_{m+1}), same setup ===")
print(hdr)
for M in [16, 32]:
    for dt, T in DTS.items():
        t, lnW, is_heun = run(M, dt, T, op_antisym, KAPPAS, seed=20 + M)
        for kap, interp, r, rs, rm, S, ratio in summarize("", t, lnW, is_heun, KAPPAS, M):
            print(f"{M:3d} {dt:7.0e} {T:5.1f} {kap:8.1e} {interp:>6} {r:9.4f} {rs:7.4f} {rm:9.4f} {M * S:8.3f} {ratio:12.4f}")
print(f"  [elapsed {time.time() - t0:.1f} s]")
print("  Expectation: Heun (Stratonovich) conserves W up to O(dt) discretisation drift;")
print("  Ito drifts by 2 kappa sum_p |A_p g|^2 / W > 0 even for an antisymmetric operator.")
