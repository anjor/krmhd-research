#!/usr/bin/env python
"""Reference profiles for the neighbour correlator Gamma_m in the Hermite hierarchy.

Study 04, Phase 0, derivation D02.  Two stochastic reference systems share the
linear Hermite ladder of GANDALF (SPEC.md section 1, s = 1, Lambda = sqrt(5)):

    dg_0/dt = -ik ( g_1/sqrt2 )
    dg_1/dt = -ik ( g_2 + c g_0 ),                 c = (1 - 1/Lambda)/sqrt2
    dg_m/dt = -ik ( sqrt((m+1)/2) g_{m+1} + sqrt(m/2) g_{m-1} ) - nu_m g_m,  m >= 2
    g_{M+1} = 0,   nu_m = nu (m/M)^6 for m >= 2, else 0.

Per wavenumber the observables are (SPEC.md section 6)

    W_m     = |g_m|^2
    Pi_m    = -k sqrt(2(m+1)) Im[g_{m+1} g_m^*]       Hermite free-energy flux
    Gamma_m = c_m Re[g_{m+1} g_m^*],   c_0 = sqrt2 (1 - 1/Lambda), c_m = sqrt(2(m+1))

so that Gamma_m and -Pi_m/k are the real and imaginary parts of one complex
neighbour correlator.  Units: v_th = v_A = 1, k in units of 2 pi / L with L = 1,
time in L/v_th.

Part A -- linear phase mixing at a single k (reference profile a)
------------------------------------------------------------------
Write the ladder as dg/dt = L g with L = -ik A - N, A the real bidiagonal
ladder matrix (A couples m only to m +- 1) and N = diag(nu_m) real.  Expand the
propagator exp(L t) = sum_n (t^n/n!) L^n.  Every term of L^n is a product of n
factors, each either -ik A (purely imaginary, shifts m by one) or -N (real,
keeps m).  The (m, m') entry of L^n therefore contains exactly (m - m') mod 2
imaginary factors beyond an even number, i.e. it is real when m - m' is even
and purely imaginary when m - m' is odd.  Driving with g(0) = delta_{m0} gives

    g_m(t) = R_m(t)      (real)      for even m,
    g_m(t) = -i I_m(t)   (I_m real)  for odd m,

and so Re[g_{m+1} g_m^*] = Re[(-i I_{m+1}) R_m] = 0 for even m and
Re[R_{m+1} (+i I_m)] = 0 for odd m: Gamma_m(t) = 0 identically, for every m
including m = 0, at every t and every nu.  Because floating-point products of
exact reals and exact imaginaries keep the zero parts exactly zero, the
numerically computed Gamma_m is zero to the last bit, not merely to round-off.

The same argument applies to the forced steady state.  With white-noise forcing
on g_0 (Ito kick xi sqrt(dt) each step, xi complex standard normal) the state is
g(t) = sum_j xi_j sqrt(dt) h(t - t_j) with h the impulse response above.  The
steady-state covariance is C = dt sum_n P^n e_0 e_0^T P^{n dagger}, P = exp(L dt),
because cross terms between independent kicks average to zero.  Each term has
the parity structure (entries with m - m' odd purely imaginary), hence
<Gamma_m> = c_m Re C_{m+1,m} = 0 exactly.  The script computes C by doubling
(C_{2n} = C_n + P^n C_n P^{n dagger}) and verifies that the Monte-Carlo averages
agree with it and that Re C_{m+1,m} vanishes to the last bit.

Collisionless conservation (nu = 0): W = (1 - 1/Lambda)|g_0|^2 + sum_{m>=1}|g_m|^2
and Gamma = sum_m Gamma_m are both exact invariants of the ladder (SPEC.md
section 2); they are checked to round-off on a random complex initial condition.

Part B -- stochastic echo model after Adkins & Schekochihin (2018, JPP 84,
905840107), reference profile (b)
--------------------------------------------------------------------------------
Periodic 1D domain, k = 2 pi n, n = -N/2..N/2-1, the same Hermite ladder as
Part A per k, white-noise forcing on g_{k,0} at |n_k| in {1, 2}, and a
Kraichnan (white-in-time) potential phi_p(t) with

    <phi_p(t) phi_{p'}(t')> = 2 kappa_p delta_{p,-p'} delta(t - t'),  phi_{-p} = phi_p^*,

kappa_p = kappa for |n_p| in {1, 2}, zero otherwise (AS2018 eq. 4.1).  The
nonlinearity is the Hermite image of the advection -(1/2) d_x phi d_v g:

    d_t g_{k,m} = [ladder]_{k,m} - nu_m g_{k,m} - c sum_p i p phi_p (J g_{k-p})_m + f_k delta_{m0}.

Two forms of J are implemented, selected on the command line (--nl):

  as2018 (c = 1, J = A_low):  AS2018 eq. (3.10), the Vlasov term for a
    Maxwellian-weighted expansion delta f = F_M sum_m g_m H_m / sqrt(2^m m!),
    where d_v (H_m F_M) = -H_{m+1} F_M: row m is fed by m - 1 only,
    (J g)_m = sqrt(m/2) g_{m-1}.  This form does not conserve sum |g|^2
    (AS2018 eq. 3.17).  Taken literally as an SDE it is exponentially unstable:
    its Stratonovich drift sum_p kappa_p B_p B_{-p} = (S/2) sqrt(m(m-1)) g_{m-2},
    S = sum_p p^2 kappa_p, is quasilinear velocity diffusion, which in the
    free-energy norm int delta f^2 / F_M is a sink only for structures with
    v^2 < m (the WKB regime g_{m-2} ~ -g_m that AS2018 assume in their 3.19-3.21)
    and a source for v^2 > m.  Kicks pump energy into those tail modes at the
    rate m S; nothing damps them, and the Hermite truncation puts the fastest
    at m ~ M.  Measured growth rate of W: 9 per unit time at S = 0.158 and 1.7
    at S = 0.040 (about M S), independent of dt (9e-4 vs 3e-4), with Ito,
    explicit-drift Stratonovich and Heun stepping alike, with or without the
    k = 0 column.  Lenard-Bernstein collisions nu m beat it only for
    nu > S/2, which puts the collisional cutoff (1.5 k sqrt2 / nu)^{2/3} below
    the nonlinear crossover: no echo range survives.  This form is kept for
    reference and is not run by default.

  conservative (c = 1/2, J = A_low - A_up, the default):  the same advection
    acting on the orthonormal-Hermite field, d_v psi_m = sqrt(m/2) psi_{m-1}
    - sqrt((m+1)/2) psi_{m+1}, with the (1, 0) entry weighted by (1 - 1/Lambda)
    exactly as the ladder A is, so that P J is antisymmetric for the same
    P = diag(1 - 1/Lambda, 1, ..., 1) that defines W.  The kick operator
    K = sum_p eta_p K_p, K_p = -(ip/2) J S_p (S_p the shift k -> k - p), then
    satisfies (P K)^+ = -P K, the Stratonovich evolution is W-unitary, and
    the Ito drift D = sum_p kappa_p K_p K_{-p} = (S/4) J^2 obeys
    2 Re[g^+ P D g] + sum_p 2 kappa_p (K_p g)^+ P (K_p g) = 0 identically
    (checked to round-off below), i.e. W is conserved in the mean step by
    step.  On WKB structures (g_{m+-1} ~ -+ i g_m) the kick has the same
    strength as the AS2018 form, |K_p g|^2 ~ (m/2) p^2 |g|^2, so S plays the
    same role (turbulent collisionality (m/2) S); on the anti-WKB tail modes
    the conservative kick vanishes, which is what removes the instability.

Numerically: exact propagation per k with exp((L_k + D) dt), L_k the ladder
with collisions and D the drift (both deterministic and linear), then the
zero-mean Euler-Maruyama kick g += K g with eta_p = phi_p sqrt(dt),
<|eta_p|^2> = 2 kappa_p dt, eta_{-p} = eta_p^*, evaluated on the propagated
field (independent of this step's eta: Ito).  The k = 0 column is zeroed
(GANDALF's zero_k0_mode; the mean distribution is not part of g).  g(x, v)
is complex (no reality condition on g_{-k}) so that the k-odd part of Gamma
is a non-trivial observable.  Part B uses M = 64 and nu = 10 (hypercollisions
(m/M)^6 as in Part A) so that the truncation reflection, which with nu = 1
would return ~75% of the flux at M = 64, is ~6% and the anti-phase-mixing
branch measures the echo; the collisional loss below m = 40 is ~7%.  A
kappa = 0 control with the identical ladder, forcing and M is solved exactly
(Lyapunov doubling per forced k) for the comparison.

Reflection symmetry.  Under x -> -x, v -> -v one has H_m(-v) = (-1)^m H_m(v)
and k -> -k, so g_{k,m} -> g'_{k,m} = (-1)^m g_{-k,m} and phi_p -> phi'_p =
phi_{-p}.  In the ladder, (-1)^m g_{-k,m+-1} = -g'_{k,m+-1} and k -> -k, two
sign changes: -i(-k)(-g'_{k,m+-1}) = -ik g'_{k,m+-1}, the same row.  In the
nonlinear term J couples m to m +- 1, so (J g_{-k-p})_m picks up -(-1)^m,
i.e. -c sum_p ip phi_p (-1)^m (J g_{-k-p})_m = +c sum_p ip phi_p (J g'_{k+p})_m
= -c sum_p ip phi'_p (J g'_{k-p})_m after p -> -p, again the same row; the
collision term is invariant.  The forcing is statistically identical at k
and -k and kappa_p = kappa_{-p}, so the stationary ensemble is invariant.
The map sends Gamma_{k,m} = c_m Re[g_{k,m+1} g_{k,m}^*] ->
c_m Re[(-1)^{2m+1} g_{-k,m+1} g_{-k,m}^*] = -Gamma_{-k,m}, hence
<Gamma_{k,m}> = -<Gamma_{-k,m}>, <Gamma_{0,m}> = 0 and sum_k <Gamma_{k,m}> = 0
exactly.  The same map gives Pi_{k,m} -> +Pi_{-k,m}: the flux is k-even and
Gamma is k-odd.  The k-odd part Gamma^odd_m = sum_{k>0}[Gamma_{k,m} -
Gamma_{-k,m}] is not constrained by any symmetry of the model and is reported
with its noise level.  Both forms of J obey this symmetry.

Flux suppression.  In a statistically steady state the k-summed net flux
through any m is fixed by the energy budget, sum_k Pi_m = injection -
dissipation below m (+ nonlinear source below m for the as2018 form), so the
stochastic echo cannot reduce the net k-summed flux at fixed injection.  What
the echo does is populate the anti-phase-mixing branch: with
A_{k,m} = i^m (g_{k,m} + i g_{k,m+1})/2 and B_{k,m} = (-i)^m (g_{k,m} - i g_{k,m+1})/2
one has exactly Pi_{k,m} = k sqrt(2(m+1)) (|A|^2 - |B|^2), so Pi = Pi^+ - Pi^-
with Pi^+ = sum_k |k| sqrt(2(m+1)) |forward|^2 (A for k > 0, B for k < 0).
The suppression ratio reported is the net-to-forward ratio Pi/Pi^+ of the
echo run divided by the same ratio for the kappa = 0 control; the raw ratio
of net fluxes to Part A is printed as well.

Outputs: reference_profiles.npz next to this file and a printed report.  Exit
code 0 iff every check passes.  numpy only.  Usage:
    uv run python 02_reference_profiles.py [--nl conservative|as2018]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Parameters (single source of truth for the script; no physics in the loops)
# ----------------------------------------------------------------------------
LAMBDA: float = np.sqrt(5.0)
C_LAMBDA: float = (1.0 - 1.0 / LAMBDA) / np.sqrt(2.0)
W0_WEIGHT: float = 1.0 - 1.0 / LAMBDA          # weight of |g_0|^2 in W
HYPER_N: int = 6

# Part A
M_LIN: int = 128
K_LIN: float = 2.0 * np.pi
NU: float = 1.0
DT_LIN: float = 0.01
T_WARM_LIN: float = 100.0
T_AVG_LIN: float = 1000.0
T_BLOCK_LIN: float = 50.0
N_REAL_LIN: int = 8
IMPULSE_TIMES: tuple[float, ...] = (0.25, 1.0, 3.0, 10.0)

# Part B
M_ECHO: int = 64
NU_ECHO: float = 10.0
N_K: int = 64
FORCED_N: tuple[int, ...] = (1, 2)              # |n| of forced / advecting modes
KAPPA: float = 4.0e-4
DT_ECHO: float = 9.0e-4
T_WARM_ECHO: float = 15.0
T_AVG_ECHO: float = 50.0
T_BLOCK_ECHO: float = 5.0
N_REAL_ECHO: int = 4
NL_FORM_DEFAULT: str = "conservative"            # or "as2018" (unstable, see docstring)

SLOPE_RANGE: tuple[int, int] = (4, 40)
SEED: int = 20261001
TABLE_M: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32, 64)

HERE = Path(__file__).resolve().parent
OUT_NPZ = HERE / "reference_profiles.npz"


# ----------------------------------------------------------------------------
# Linear operator
# ----------------------------------------------------------------------------
def ladder_matrix(M: int, Lambda: float = LAMBDA) -> np.ndarray:
    """Real (M+1)x(M+1) ladder matrix A of SPEC.md section 1.

    dg/dt = -ik A g.  A_{01} = 1/sqrt2, A_{10} = c_Lambda,
    A_{m,m+1} = sqrt((m+1)/2) (m >= 1), A_{m,m-1} = sqrt(m/2) (m >= 2),
    zero truncation g_{M+1} = 0 (no row/column M+1).
    """
    A = np.zeros((M + 1, M + 1))
    A[0, 1] = 1.0 / np.sqrt(2.0)
    A[1, 0] = (1.0 - 1.0 / Lambda) / np.sqrt(2.0)
    for m in range(1, M):
        A[m, m + 1] = np.sqrt((m + 1) / 2.0)
    for m in range(2, M + 1):
        A[m, m - 1] = np.sqrt(m / 2.0)
    return A


def collision_rates(M: int, nu: float) -> np.ndarray:
    """Hypercollision rates nu_m = nu (m/M)^6 for m >= 2, zero for m = 0, 1."""
    m = np.arange(M + 1, dtype=float)
    rates = nu * (m / M) ** HYPER_N
    rates[:2] = 0.0
    return rates


def generator(k: float, M: int, nu: float) -> np.ndarray:
    """Complex generator L = -ik A - diag(nu_m) of dg/dt = L g at one k."""
    return -1j * k * ladder_matrix(M) - np.diag(collision_rates(M, nu))


def expm(L: np.ndarray) -> np.ndarray:
    """Matrix exponential by scaling and squaring with a Taylor series.

    Scales L by 2^-s so that ||L/2^s||_1 <= 1/4, sums the Taylor series to
    machine precision, squares s times.  Products of exactly-real and
    exactly-imaginary entries keep zero parts exactly zero, so the parity
    structure of the ladder propagator (docstring, Part A) survives bit-exactly.
    """
    n = L.shape[0]
    norm1 = np.linalg.norm(L, 1)
    s = 0 if norm1 == 0 else max(0, int(np.ceil(np.log2(norm1 / 0.25))))
    X = L / 2.0**s
    E = np.eye(n, dtype=complex) + X
    term = X.copy()
    for j in range(2, 60):
        term = term @ X / j
        E += term
        if np.linalg.norm(term, 1) < 1e-17 * np.linalg.norm(E, 1):
            break
    for _ in range(s):
        E = E @ E
    return E


# ----------------------------------------------------------------------------
# Observables
# ----------------------------------------------------------------------------
def coupling_coeffs(M: int) -> np.ndarray:
    """c_m, m = 0..M-1: c_0 = sqrt2 (1 - 1/Lambda), c_m = sqrt(2(m+1)) for m >= 1."""
    c = np.sqrt(2.0 * (np.arange(M) + 1.0))
    c[0] = np.sqrt(2.0) * (1.0 - 1.0 / LAMBDA)
    return c


def pair_correlator(g: np.ndarray) -> np.ndarray:
    """g_{m+1} g_m^* along the last axis, m = 0..M-1."""
    return g[..., 1:] * np.conj(g[..., :-1])


def flux_from_correlator(C: np.ndarray, k: np.ndarray | float) -> np.ndarray:
    """Pi_m = -k sqrt(2(m+1)) Im[C_m] with C_m = g_{m+1} g_m^*; k broadcasts on the left."""
    M = C.shape[-1]
    return -np.asarray(k)[..., None] * np.sqrt(2.0 * (np.arange(M) + 1.0)) * C.imag


def gamma_from_correlator(C: np.ndarray) -> np.ndarray:
    """Gamma_m = c_m Re[C_m]."""
    return coupling_coeffs(C.shape[-1]) * C.real


def free_energy(g: np.ndarray) -> np.ndarray:
    """W = (1 - 1/Lambda)|g_0|^2 + sum_{m>=1} |g_m|^2 along the last axis."""
    w = np.abs(g) ** 2
    return W0_WEIGHT * w[..., 0] + w[..., 1:].sum(axis=-1)


def forward_backward_flux(W: np.ndarray, C: np.ndarray, k: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Phase-mixing (+) and anti-phase-mixing (-) parts of Pi_m from <W_m> and <C_m>; Pi = Pi^+ - Pi^- exactly.

    A_m = i^m (g_m + i g_{m+1})/2 and B_m = (-i)^m (g_m - i g_{m+1})/2 split
    g_m = (-i)^m A_m + i^m B_m (AS2018 eq. 3.22, 3.28 generalised to complex g).
    |A|^2 = (W_m + W_{m+1})/4 - Im[C_m]/2, |B|^2 = (W_m + W_{m+1})/4 + Im[C_m]/2
    with C_m = g_{m+1} g_m^*, so |A|^2 - |B|^2 = -Im[C_m] and
    Pi = k sqrt(2(m+1)) (|A|^2 - |B|^2).  For k > 0 the A branch propagates to
    higher m (forward); for k < 0 the B branch.  Linear in W and C, so it can
    be evaluated on time averages.  Returns (Pi^+, Pi^-), shape of C.
    """
    M = C.shape[-1]
    half = (W[..., :-1] + W[..., 1:]) / 4.0
    a2 = half - C.imag / 2.0
    b2 = half + C.imag / 2.0
    kk = np.asarray(k, dtype=float)[..., None]
    pref = np.abs(kk) * np.sqrt(2.0 * (np.arange(M) + 1.0))
    fwd = np.where(kk > 0, a2, b2)
    bwd = np.where(kk > 0, b2, a2)
    return pref * fwd, pref * bwd


def local_slope(W: np.ndarray, m_lo: int, m_hi: int) -> float:
    """Least-squares slope of log W_m vs log m over m_lo <= m <= m_hi."""
    m = np.arange(m_lo, m_hi + 1)
    return float(np.polyfit(np.log(m), np.log(W[m_lo : m_hi + 1]), 1)[0])


# ----------------------------------------------------------------------------
# Checks bookkeeping
# ----------------------------------------------------------------------------
class Checks:
    """Collects named pass/fail results and prints them."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.n = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.n += 1
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
        if not ok:
            self.failures.append(f"{name} ({detail})" if detail else name)


# ----------------------------------------------------------------------------
# Part A
# ----------------------------------------------------------------------------
def part_a_impulse(chk: Checks) -> None:
    """A1: impulse response parity (Gamma_m == 0 bit-exactly) and nu = 0 invariants."""
    M, k = M_LIN, K_LIN
    L = generator(k, M, NU)
    e0 = np.zeros(M + 1, dtype=complex)
    e0[0] = 1.0
    worst_gamma = 0.0
    worst_parity = 0.0
    for t in IMPULSE_TIMES:
        g = expm(L * t) @ e0
        gam = gamma_from_correlator(pair_correlator(g))
        worst_gamma = max(worst_gamma, float(np.max(np.abs(gam))))
        even, odd = np.arange(M + 1) % 2 == 0, np.arange(M + 1) % 2 == 1
        worst_parity = max(worst_parity, float(np.max(np.abs(g[even].imag))),
                           float(np.max(np.abs(g[odd].real))))
    wmax = float(np.max(np.abs(g) ** 2))
    chk.check("A1 impulse response: Gamma_m = 0 for all m at t in %s" % (IMPULSE_TIMES,),
              worst_gamma <= 1e-12 * wmax, f"max|Gamma_m| = {worst_gamma:.1e}")
    chk.check("A1 impulse response: g_m real (even m) / imaginary (odd m)",
              worst_parity <= 1e-12, f"max violation = {worst_parity:.1e}")

    # nu = 0 invariants on a random complex initial condition
    rng = np.random.default_rng(SEED + 1)
    g0 = rng.standard_normal(M + 1) + 1j * rng.standard_normal(M + 1)
    L0 = generator(k, M, 0.0)
    W_init = float(free_energy(g0))
    gam_init = float(gamma_from_correlator(pair_correlator(g0)).sum())
    gam_scale = float(np.sum(coupling_coeffs(M) * np.abs(g0[1:]) * np.abs(g0[:-1])))
    errW = errG = 0.0
    for t in IMPULSE_TIMES:
        g = expm(L0 * t) @ g0
        errW = max(errW, abs(float(free_energy(g)) - W_init) / W_init)
        errG = max(errG, abs(float(gamma_from_correlator(pair_correlator(g)).sum()) - gam_init) / gam_scale)
    chk.check("A1 nu = 0, random IC: W conserved to round-off", errW < 1e-10, f"rel err {errW:.1e}")
    chk.check("A1 nu = 0, random IC: Gamma = sum_m Gamma_m conserved to round-off",
              errG < 1e-10, f"err/scale {errG:.1e}, Gamma(0)/scale = {gam_init / gam_scale:+.3f}")


def lyapunov_steady_state(P: np.ndarray, dt: float, n_doublings: int) -> np.ndarray:
    """Exact covariance C = dt sum_{n<2^J} P^n e_0 e_0^T P^{n+} of the kicked ladder by doubling."""
    n = P.shape[0]
    C = np.zeros((n, n), dtype=complex)
    C[0, 0] = dt
    Q = P.copy()
    for _ in range(n_doublings):
        C = C + Q @ C @ Q.conj().T
        Q = Q @ Q
    return C


def part_a_forced(chk: Checks) -> dict[str, np.ndarray | float]:
    """A2: forced steady state of linear phase mixing, Monte-Carlo plus exact Lyapunov solution."""
    M, k, dt = M_LIN, K_LIN, DT_LIN
    P = expm(generator(k, M, NU) * dt)
    R = N_REAL_LIN
    rng = np.random.default_rng(SEED + 2)
    n_warm = int(round(T_WARM_LIN / dt))
    n_block = int(round(T_BLOCK_LIN / dt))
    n_blocks = int(round(T_AVG_LIN / T_BLOCK_LIN))
    g = np.zeros((M + 1, R), dtype=complex)
    sqdt = np.sqrt(dt)

    def kick() -> np.ndarray:
        return (rng.standard_normal(R) + 1j * rng.standard_normal(R)) / np.sqrt(2.0) * sqdt

    for _ in range(n_warm):
        g = P @ g
        g[0] += kick()
    W_blk = np.zeros((n_blocks, R, M + 1))
    C_blk = np.zeros((n_blocks, R, M), dtype=complex)
    for b in range(n_blocks):
        Wacc = np.zeros((M + 1, R))
        Cacc = np.zeros((M, R), dtype=complex)
        for _ in range(n_block):
            g = P @ g
            g[0] += kick()
            Wacc += np.abs(g) ** 2
            Cacc += g[1:] * np.conj(g[:-1])
        W_blk[b] = Wacc.T / n_block
        C_blk[b] = Cacc.T / n_block
    nb = n_blocks * R
    W = W_blk.reshape(nb, M + 1)
    Pi = flux_from_correlator(C_blk, k).reshape(nb, M)
    Gam = gamma_from_correlator(C_blk).reshape(nb, M)
    W_mean, Pi_mean, Gam_mean = W.mean(0), Pi.mean(0), Gam.mean(0)
    Gam_sig = Gam.std(0, ddof=1) / np.sqrt(nb)
    Pi_sig = Pi.std(0, ddof=1) / np.sqrt(nb)
    Pip_mean, Pim_mean = forward_backward_flux(W_mean, C_blk.reshape(nb, M).mean(0), k)

    # exact steady state (same discrete-time model), 2^17 dt = 1310 time units
    C_ex = lyapunov_steady_state(P, dt, 17)
    W_ex = C_ex.diagonal().real
    Cpair_ex = np.array([C_ex[m + 1, m] for m in range(M)])
    Pi_ex = flux_from_correlator(Cpair_ex, k)
    Gam_ex = gamma_from_correlator(Cpair_ex)
    eps_W = W0_WEIGHT  # injection rate of W: (1 - 1/Lambda) <|xi|^2>

    lo, hi = SLOPE_RANGE
    sl = slice(lo, hi + 1)
    pi_plateau = Pi_mean[sl]
    ok_i = bool(np.all(pi_plateau > 0) and (pi_plateau.max() - pi_plateau.min()) <= 0.2 * pi_plateau.mean())
    chk.check(f"A2 <Pi_m> > 0 and constant within 20% over {lo} <= m <= {hi}", ok_i,
              f"mean {pi_plateau.mean():.4f}, spread {(pi_plateau.max() - pi_plateau.min()) / pi_plateau.mean():.3f}, "
              f"injection eps_W = {eps_W:.4f}")
    slope = local_slope(W_mean, lo, hi)
    chk.check(f"A2 local slope of <W_m> over {lo} <= m <= {hi} in [-0.7, -0.3]",
              -0.7 <= slope <= -0.3, f"slope {slope:+.3f}")
    ratio = np.abs(Gam_mean) / np.where(Gam_sig > 0, Gam_sig, np.inf)
    chk.check("A2 |<Gamma_m>| < 3 sigma for every m (Monte-Carlo)", bool(np.all(ratio < 3.0)),
              f"max |Gamma|/sigma = {ratio.max():.2f} at m = {int(ratio.argmax())}")
    chk.check("A2 exact steady state: <Gamma_m> = c_m Re C_{m+1,m} = 0 for every m",
              float(np.max(np.abs(Gam_ex))) <= 1e-12 * float(W_ex.max()),
              f"max |Gamma_exact| = {np.max(np.abs(Gam_ex)):.1e}")
    relW = np.max(np.abs(W_mean[: hi + 1] - W_ex[: hi + 1]) / W_ex[: hi + 1])
    relP = np.max(np.abs(Pi_mean[1 : hi + 1] - Pi_ex[1 : hi + 1]) / Pi_ex[1 : hi + 1])
    chk.check(f"A2 Monte-Carlo <W_m>, <Pi_m> match exact Lyapunov solution within 15% for m <= {hi}",
              relW < 0.15 and relP < 0.15, f"max rel dev W {relW:.3f}, Pi {relP:.3f}")
    chk.check("A2 energy balance: <Pi_m> over plateau equals injection within 10%",
              abs(pi_plateau.mean() - eps_W) < 0.1 * eps_W,
              f"<Pi>/eps_W = {pi_plateau.mean() / eps_W:.3f}")
    echo_frac = float((Pim_mean[sl] / Pip_mean[sl]).mean())
    print(f"  A2 info: Pi_0 = {Pi_mean[0]:.4f} (expected <|xi|^2> = 1 since (1-1/Lambda) Pi_0 = eps_W), "
          f"W_total = {free_energy(np.sqrt(W_mean)):.3f}, "
          f"backward/forward flux Pi^-/Pi^+ over plateau = {echo_frac:.3f} (truncation reflection)")
    return dict(W=W_mean, Pi=Pi_mean, Pi_sigma=Pi_sig, Gamma=Gam_mean, Gamma_sigma=Gam_sig,
                W_exact=W_ex, Pi_exact=Pi_ex, Pi_fwd=Pip_mean, Pi_bwd=Pim_mean,
                slope=slope, plateau=float(pi_plateau.mean()), echo_frac=echo_frac,
                net_over_fwd=float((Pi_mean[sl] / Pip_mean[sl]).mean()))


# ----------------------------------------------------------------------------
# Part B
# ----------------------------------------------------------------------------
def nonlinear_operator(M: int, form: str) -> tuple[np.ndarray, float]:
    """Matrix J and prefactor c of the kick K_p = -c i p J S_p (docstring, Part B).

    'as2018':       c = 1,   J = A_low (row m fed by m - 1 with sqrt(m/2)).
    'conservative': c = 1/2, J = A_low - A_up, P J antisymmetric with P = diag(1 - 1/Lambda, 1, ...).
    """
    A = ladder_matrix(M)
    A_low = np.tril(A)
    A_up = np.triu(A)
    if form == "as2018":
        return A_low, 1.0
    if form == "conservative":
        return A_low - A_up, 0.5
    raise ValueError(form)


def linear_control(M: int, nu: float, k_list: np.ndarray, inj_var: float, dt: float
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact kappa = 0 steady state of Part B's ladder: k-summed W_m, C_m = <g_{m+1} g_m^*>, Pi_m."""
    W = np.zeros(M + 1)
    C = np.zeros(M, dtype=complex)
    Pi = np.zeros(M)
    for k in k_list:
        P = expm(generator(k, M, nu) * dt)
        Cs = lyapunov_steady_state(P, dt * inj_var, 17)
        W += Cs.diagonal().real
        pair = np.array([Cs[m + 1, m] for m in range(M)])
        C += pair
        Pi += flux_from_correlator(pair, k)
    return W, C, Pi


def part_b_echo(chk: Checks, lin: dict, form: str) -> dict[str, np.ndarray | float]:
    """Stochastic echo model (docstring, Part B), k-resolved Gamma, flux branches, control."""
    M, N, dt, R, nu = M_ECHO, N_K, DT_ECHO, N_REAL_ECHO, NU_ECHO
    n_idx = np.fft.fftfreq(N, 1.0 / N).astype(int)        # 0, 1, ..., N/2-1, -N/2, ..., -1
    k = 2.0 * np.pi * n_idx
    kpos = k > 0
    J, c_nl = nonlinear_operator(M, form)
    p_list = [2.0 * np.pi * n for n in FORCED_N]            # p > 0; -p handled by conjugate
    S = KAPPA * sum(2.0 * p * p for p in p_list)             # sum over +-p of kappa_p p^2
    D = S * c_nl**2 * (J @ J)                                # Ito drift of the Stratonovich SDE
    forced_idx = np.array([j for j, n in enumerate(n_idx) if abs(n) in FORCED_N])
    inj_var = 1.0 / len(forced_idx)                          # total <|xi|^2> = 1 as in Part A
    k0_idx = int(np.flatnonzero(n_idx == 0)[0])
    Pw = np.ones(M + 1)
    Pw[0] = W0_WEIGHT

    # B0: mean-energy balance of kick + drift on a random field (per k, shifts drop out)
    rng0 = np.random.default_rng(SEED + 4)
    gt = rng0.standard_normal(M + 1) + 1j * rng0.standard_normal(M + 1)
    Jg = J @ gt
    # sum over +-p of <|eta_p|^2> |K_p g|_P^2 = 2 S c^2 |J g|_P^2 (shifts drop out of the k-sum)
    pump = 2.0 * S * c_nl**2 * float(np.sum(Pw * np.abs(Jg) ** 2))
    drift_term = 2.0 * float(np.real(np.vdot(gt, Pw * (D @ gt))))
    scale = float(np.sum(Pw * np.abs(gt) ** 2)) * S * M
    resid = abs(pump + drift_term) / scale
    if form == "conservative":
        chk.check("B0 nonlinear kick conserves W in the mean (Ito drift = -pumping) to round-off",
                  resid < 1e-12, f"residual/scale {resid:.1e}")
    else:
        print(f"  B0 info: as2018 form, mean W source per unit time / (S M W) = {(pump + drift_term) / scale:+.3e} (not conserved)")

    # propagators per k with the drift folded in; L(-k) = conj(L(k)) and D real => conj
    P = np.empty((N, M + 1, M + 1), dtype=complex)
    cache: dict[int, np.ndarray] = {}
    for j, n in enumerate(n_idx):
        if abs(n) not in cache:
            cache[abs(n)] = expm((generator(2.0 * np.pi * abs(n), M, nu) + D) * dt)
        P[j] = cache[abs(n)] if n >= 0 else np.conj(cache[abs(n)])
    kick_rel = np.sqrt(M * S * dt)                           # rms kick/field at m = M (WKB estimate)
    m_cross = (2.0 * 2.0 * np.pi / S) ** (2.0 / 3.0) * 2.0 ** (1.0 / 3.0)
    print(f"  B info: form = {form}, kappa = {KAPPA:.2e}, S = sum_p p^2 kappa_p = {S:.3f}, "
          f"nonlinear/phase-mixing crossover at k = 2pi near m ~ {m_cross:.0f}, "
          f"rms kick/field at m = M per step = {kick_rel:.3f}, nu = {nu}, nu_M dt = {nu * dt:.1e}, "
          f"N = {N}, M = {M}, dt = {dt}, {R} realisations")

    rng = np.random.default_rng(SEED + 3)
    g = np.zeros((N, M + 1, R), dtype=complex)                # layout (k, m, realisation)
    sqdt = np.sqrt(dt)
    eta_amp = np.sqrt(2.0 * KAPPA * dt)
    shift_idx = {n: (np.arange(N) - n) % N for n in (*FORCED_N, *(-n for n in FORCED_N))}  # g_{k-p}
    n_forced = forced_idx.size
    Jlow = np.diag(J, -1)[:, None]                            # (J g)_m gets Jlow[m-1] g_{m-1}, m >= 1
    Jup = np.diag(J, 1)[:, None]                              # (J g)_m gets Jup[m] g_{m+1}, m <= M-1
    has_up = bool(np.any(Jup != 0))

    def step(g: np.ndarray) -> np.ndarray:
        g = np.matmul(P, g)                                   # exact linear + drift propagation per k
        gnew = g.copy()
        for p, n in zip(p_list, FORCED_N):
            eta = eta_amp * (rng.standard_normal(R) + 1j * rng.standard_normal(R)) / np.sqrt(2.0)
            for coef, gs in (((-1j * c_nl * p) * eta, g[shift_idx[n]]),             # phi_p
                             ((1j * c_nl * p) * np.conj(eta), g[shift_idx[-n]])):   # phi_{-p} = phi_p^*
                cc = coef[None, None, :]
                gnew[:, 1:, :] += (cc * Jlow) * gs[:, :-1, :]
                if has_up:
                    gnew[:, :-1, :] += (cc * Jup) * gs[:, 1:, :]
        xi = (rng.standard_normal((n_forced, R)) + 1j * rng.standard_normal((n_forced, R))) * np.sqrt(inj_var / 2.0)
        gnew[forced_idx, 0, :] += xi * sqdt
        gnew[k0_idx, :, :] = 0.0
        return gnew

    n_warm = int(round(T_WARM_ECHO / dt))
    n_block = int(round(T_BLOCK_ECHO / dt))
    n_blocks = int(round(T_AVG_ECHO / T_BLOCK_ECHO))
    t0 = time.time()
    W_hist = []
    for i in range(n_warm):
        g = step(g)
        if i % n_block == 0:
            W_hist.append(float(free_energy(np.moveaxis(g, 2, 0)).sum(axis=1).mean()))
    W_blk = np.zeros((n_blocks, R, N, M + 1))
    C_blk = np.zeros((n_blocks, R, N, M), dtype=complex)
    for b in range(n_blocks):
        Wacc = np.zeros((N, M + 1, R))
        Cacc = np.zeros((N, M, R), dtype=complex)
        for _ in range(n_block):
            g = step(g)
            Wacc += np.abs(g) ** 2
            Cacc += g[:, 1:, :] * np.conj(g[:, :-1, :])
        W_blk[b] = np.moveaxis(Wacc, 2, 0) / n_block
        C_blk[b] = np.moveaxis(Cacc, 2, 0) / n_block
        W_hist.append(float((Pw * W_blk[b].sum(axis=1)).sum(axis=-1).mean()))
    print(f"  B info: {n_warm + n_blocks * n_block} steps in {time.time() - t0:.1f} s; "
          f"W_total(t) every {T_BLOCK_ECHO} time units: " + " ".join(f"{w:.3g}" for w in W_hist))

    nb = n_blocks * R
    Wk = W_blk.reshape(nb, N, M + 1)
    Pik = flux_from_correlator(C_blk, k).reshape(nb, N, M)
    Gamk = gamma_from_correlator(C_blk).reshape(nb, N, M)
    W_sum = Wk.sum(1)                                    # (nb, M+1)
    Pi_sum = Pik.sum(1)
    Gam_sum = Gamk.sum(1)
    Gam_odd = Gamk[:, kpos, :].sum(1) - Gamk[:, k < 0, :].sum(1)
    Gam_kpos = Gamk[:, kpos, :].sum(1)
    Pip_k, Pim_k = forward_backward_flux(Wk.mean(0), C_blk.reshape(nb, N, M).mean(0), k)

    def mean_se(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return x.mean(0), x.std(0, ddof=1) / np.sqrt(x.shape[0])

    W_m, W_s = mean_se(W_sum)
    Pi_m, Pi_s = mean_se(Pi_sum)
    G_m, G_s = mean_se(Gam_sum)
    Go_m, Go_s = mean_se(Gam_odd)
    Gp_m, Gp_s = mean_se(Gam_kpos)
    Pip_m, Pim_m = Pip_k.sum(0), Pim_k.sum(0)
    Gk_mean = Gamk.mean(0)                                 # (N, M) k-resolved

    # exact kappa = 0 control on the same ladder, forcing and M
    k_forced = k[forced_idx]
    W_c, C_c, Pi_c = linear_control(M, nu, k_forced, inj_var, dt)
    Pip_c, Pim_c = np.zeros(M), np.zeros(M)
    for kk in k_forced:
        Pc = expm(generator(kk, M, nu) * dt)
        Cs = lyapunov_steady_state(Pc, dt * inj_var, 17)
        fp, fm = forward_backward_flux(Cs.diagonal().real, np.array([Cs[m + 1, m] for m in range(M)]), kk)
        Pip_c += fp
        Pim_c += fm

    lo, hi = SLOPE_RANGE
    sl = slice(lo, hi + 1)
    eps_W = W0_WEIGHT
    stable = bool(np.isfinite(W_hist).all()) and W_hist[-1] < 1e3 * eps_W
    chk.check("B1 steady state reached: W_total finite, no growth over the averaging window",
              stable and abs(W_hist[-1] - W_hist[len(W_hist) // 2]) < 0.5 * W_hist[-1],
              f"W_total = {W_hist[-1]:.3g} (control {float(np.sum(Pw * W_c)):.3g})")
    net_ratio_A = float(Pi_m[sl].mean() / lin["plateau"])
    net_ratio_c = float(Pi_m[sl].mean() / Pi_c[sl].mean())
    nf_echo = float((Pi_m[sl] / Pip_m[sl]).mean())
    nf_ctrl = float((Pi_c[sl] / Pip_c[sl]).mean())
    suppression = nf_echo / nf_ctrl
    echo_frac = float((Pim_m[sl] / Pip_m[sl]).mean())
    ctrl_frac = float((Pim_c[sl] / Pip_c[sl]).mean())
    slope = local_slope(W_m, lo, hi)
    slope_c = local_slope(W_c, lo, hi)
    print(f"  B info: net k-summed flux over {lo}<=m<={hi}: {Pi_m[sl].mean():.4f} = {net_ratio_c:.3f} x control "
          f"= {net_ratio_A:.3f} x Part A (budget-constrained; injection {eps_W:.3f})")
    print(f"  B info: net/forward flux Pi/Pi^+ = {nf_echo:.3f} (control {nf_ctrl:.3f}, Part A {lin['net_over_fwd']:.3f}); "
          f"suppression ratio echo/control = {suppression:.3f}")
    print(f"  B info: backward/forward Pi^-/Pi^+ = {echo_frac:.3f} (control {ctrl_frac:.3f}, Part A {lin['echo_frac']:.3f})")
    print(f"  B info: local W slope over {lo}<=m<={hi}: echo {slope:+.3f}, control {slope_c:+.3f}; "
          f"W_m(echo)/W_m(control) at m = {lo}, {hi}: {W_m[lo] / W_c[lo]:.2f}, {W_m[hi] / W_c[hi]:.2f}")
    chk.check(f"B(i) echo suppression: net-to-forward flux Pi/Pi^+ over {lo}<=m<={hi} below the kappa = 0 control",
              suppression < 0.9, f"ratio {suppression:.3f}; raw net-flux ratios: {net_ratio_c:.3f} x control, "
              f"{net_ratio_A:.3f} x Part A")
    r_even = np.abs(G_m) / np.where(G_s > 0, G_s, np.inf)
    chk.check("B(ii) k-summed Gamma_m consistent with zero within 3 sigma for all m (reflection symmetry)",
              bool(np.all(r_even < 3.0)), f"max |Gamma|/sigma = {r_even.max():.2f} at m = {int(r_even.argmax())}")
    r_odd = np.abs(Go_m) / np.where(Go_s > 0, Go_s, np.inf)
    n_sig = int(np.sum(r_odd > 3.0))
    chi2 = float(np.sum(r_odd**2))
    distinguishable = chi2 > M + 5.0 * np.sqrt(2.0 * M)
    print(f"  B(iii) Gamma^odd_m: {n_sig} of {M} moments beyond 3 sigma, chi^2/M = {chi2 / M:.2f} "
          f"(1.00 +- {np.sqrt(2.0 / M):.2f} if zero); max |Gamma^odd|/sigma = {r_odd.max():.1f} at m = {int(r_odd.argmax())}; "
          f"Gamma^odd IS {'' if distinguishable else 'NOT '}distinguishable from zero")
    gsum = Gp_m[sl].sum()
    gsum_s = float(np.sqrt(np.sum(Gp_s[sl] ** 2)))
    print(f"  B(iii) sum over {lo}<=m<={hi} of the k>0 part: {gsum:+.3e} +- {gsum_s:.1e} (naive, blocks assumed independent in m); "
          f"relative to sum_m |Pi_m|/k_f: {gsum / (Pi_m[sl].sum() / (2 * np.pi)):+.3e}")
    return dict(W=W_m, W_sigma=W_s, Pi=Pi_m, Pi_sigma=Pi_s, Gamma=G_m, Gamma_sigma=G_s,
                Gamma_odd=Go_m, Gamma_odd_sigma=Go_s, Gamma_k=Gk_mean, k=k,
                Pi_fwd=Pip_m, Pi_bwd=Pim_m, W_ctrl=W_c, Pi_ctrl=Pi_c, Pi_ctrl_fwd=Pip_c, Pi_ctrl_bwd=Pim_c,
                slope=slope, slope_ctrl=slope_c, net_ratio=net_ratio_A, net_ratio_ctrl=net_ratio_c,
                suppression=suppression, odd_distinguishable=distinguishable, chi2_odd=chi2, form=form)


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def print_table(title: str, W: np.ndarray, Pi: np.ndarray, Gam: np.ndarray, Gs: np.ndarray,
                extra: tuple[str, np.ndarray, np.ndarray] | None = None) -> None:
    hdr = f"{'m':>4} {'<W_m>':>12} {'<Pi_m>':>12} {'<Gamma_m>':>12} {'sigma':>10}"
    if extra is not None:
        hdr += f" {extra[0]:>12} {'sigma':>10}"
    print(f"  {title}")
    print("  " + hdr)
    for m in TABLE_M:
        if m >= len(Gam):
            continue
        line = f"{m:>4} {W[m]:>12.4e} {Pi[m]:>12.4e} {Gam[m]:>+12.3e} {Gs[m]:>10.1e}"
        if extra is not None:
            line += f" {extra[1][m]:>+12.3e} {extra[2][m]:>10.1e}"
        print("  " + line)


def main() -> int:
    t_start = time.time()
    form = NL_FORM_DEFAULT
    if "--nl" in sys.argv:
        form = sys.argv[sys.argv.index("--nl") + 1]
    chk = Checks()
    print("Study 04 D02: reference profiles for Gamma_m")
    print(f"Lambda = {LAMBDA:.6f}, c_Lambda = {C_LAMBDA:.5f}, nu = {NU}, k_A = 2pi")
    print("Part A: linear phase mixing, M = %d" % M_LIN)
    part_a_impulse(chk)
    lin = part_a_forced(chk)
    print("Part B: stochastic echo model (AS2018-type), M = %d, N_k = %d, nu = %g, form = %s" % (M_ECHO, N_K, NU_ECHO, form))
    echo = part_b_echo(chk, lin, form)

    print()
    print_table("Reference (a): linear phase mixing, k = 2pi, white-noise forcing on g_0",
                lin["W"], lin["Pi"], lin["Gamma"], lin["Gamma_sigma"])
    print_table("Reference (b): echo model, k-summed; last columns: Gamma^odd_m = sum_{k>0}[Gamma_k - Gamma_-k]",
                echo["W"], echo["Pi"], echo["Gamma"], echo["Gamma_sigma"],
                ("Gamma^odd_m", echo["Gamma_odd"], echo["Gamma_odd_sigma"]))

    m_lin = np.arange(M_LIN + 1)
    M_min = min(M_LIN, M_ECHO)

    def pad(x: np.ndarray, n: int) -> np.ndarray:
        out = np.full(n, np.nan)
        out[: len(x)] = x[:n]
        return out

    np.savez(
        OUT_NPZ,
        m=m_lin,
        W_lin=lin["W"], Pi_lin=pad(lin["Pi"], M_LIN + 1), Gamma_lin=pad(lin["Gamma"], M_LIN + 1),
        Gamma_lin_sigma=pad(lin["Gamma_sigma"], M_LIN + 1),
        W_lin_exact=lin["W_exact"], Pi_lin_exact=pad(lin["Pi_exact"], M_LIN + 1),
        Pi_lin_fwd=pad(lin["Pi_fwd"], M_LIN + 1), Pi_lin_bwd=pad(lin["Pi_bwd"], M_LIN + 1),
        W_echo=pad(echo["W"], M_LIN + 1), Pi_echo=pad(echo["Pi"], M_LIN + 1),
        Gamma_echo=pad(echo["Gamma"], M_LIN + 1), Gamma_echo_sigma=pad(echo["Gamma_sigma"], M_LIN + 1),
        Gamma_echo_odd=pad(echo["Gamma_odd"], M_LIN + 1), Gamma_echo_odd_sigma=pad(echo["Gamma_odd_sigma"], M_LIN + 1),
        Pi_echo_fwd=pad(echo["Pi_fwd"], M_LIN + 1), Pi_echo_bwd=pad(echo["Pi_bwd"], M_LIN + 1),
        Gamma_echo_k=echo["Gamma_k"], k_echo=echo["k"],
        W_echo_ctrl=pad(echo["W_ctrl"], M_LIN + 1), Pi_echo_ctrl=pad(echo["Pi_ctrl"], M_LIN + 1),
        Pi_echo_ctrl_fwd=pad(echo["Pi_ctrl_fwd"], M_LIN + 1), Pi_echo_ctrl_bwd=pad(echo["Pi_ctrl_bwd"], M_LIN + 1),
        k=K_LIN, nu=NU, Lambda=LAMBDA, M=M_LIN, M_echo=M_ECHO, nu_echo=NU_ECHO, N_k=N_K, kappa=KAPPA,
        dt=DT_LIN, dt_echo=DT_ECHO, n_realisations=N_REAL_LIN, n_realisations_echo=N_REAL_ECHO,
        slope_lin=lin["slope"], slope_echo=echo["slope"], slope_echo_ctrl=echo["slope_ctrl"],
        flux_ratio_net=echo["net_ratio"], flux_ratio_net_ctrl=echo["net_ratio_ctrl"],
        flux_suppression=echo["suppression"], nl_form=echo["form"],
    )
    print()
    print(f"W slopes over {SLOPE_RANGE[0]}<=m<={SLOPE_RANGE[1]}: linear {lin['slope']:+.3f}, "
          f"echo {echo['slope']:+.3f} (kappa = 0 control {echo['slope_ctrl']:+.3f})")
    print(f"Flux: net-flux ratio echo/Part-A {echo['net_ratio']:.3f}, echo/control {echo['net_ratio_ctrl']:.3f}; "
          f"net/forward suppression ratio echo/control {echo['suppression']:.3f}")
    print(f"Gamma^odd (echo model) distinguishable from zero: {echo['odd_distinguishable']}")
    print(f"Saved {OUT_NPZ}")
    print(f"Runtime {time.time() - t_start:.1f} s, {chk.n} checks")
    if chk.failures:
        print("FAILED CHECKS:")
        for f in chk.failures:
            print("  - " + f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
