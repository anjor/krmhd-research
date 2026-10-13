"""Shared numerics for the continuum derivation scripts s02-s06 in this folder: the Maxwellian, the plasma dispersion
function and its real-axis values, the dielectric function eps of one branch and the spectral density
rho = F_M/|eps(u + i0)|^2, Gaussian test functions with exact principal-value integrals and Hermite coefficients, the
slice transform T[G], the Hermite ladder matrix A of the branch operator and its Gauss rule. It supports Section 3
and Appendix B of the paper. CMM below is Chandran, Mallet & Meyrand (2026, arXiv:2607.27981v1).

Conventions: u = v_par / v_th, F_M(u) = exp(-u^2)/sqrt(pi) (normalised in u), v_A = 1, s = v_th/v_A.
Branch operator   L G = u G - (u F_M / Lambda) M0[G],   M_k[G] = int u^k G du.
Free-energy form  [f, g] = int f g / F_M du - M0[f] M0[g] / Lambda.
Hilbert transform (Morrison's convention)  Hc[f](u) = (1/pi) P int f(v)/(v - u) dv.
Plasma dispersion function Z(zeta) = int F_M(x)/(x - zeta) dx (Im zeta > 0) = i sqrt(pi) w(zeta);
on the real axis Z(u + i0) = -2 D(u) + i sqrt(pi) e^{-u^2}, D = Dawson.
Dielectric function eps(zeta) = 1 - (1/Lambda) int u F_M/(u - zeta) du = 1 - (1 + zeta Z)/Lambda;
boundary value eps(u + i0) = eR(u) + i eI(u), eR = 1 - (1 + u ReZ)/Lambda, eI = -pi u F_M/Lambda.
Spectral density rho(u) = F_M / |eps(u + i0)|^2.
Hermite: g_m = int H_m(u) G du / sqrt(2^m m!)  (CMM B30), G = sum g_m H_m F_M / sqrt(2^m m!) (CMM B27).
"""
import numpy as np
from scipy.special import wofz, dawsn

SQPI = np.sqrt(np.pi)
SQRT5 = np.sqrt(5.0)


def FM(u):
    return np.exp(-np.asarray(u, dtype=float) ** 2) / SQPI


def ReZ(u):
    """Re Z(u + i0) = P int F_M(x)/(x - u) dx = -2 Dawson(u)."""
    return -2.0 * dawsn(u)


def Zc(zeta):
    """Z(zeta) = i sqrt(pi) w(zeta): the integral for Im zeta > 0 and its analytic continuation below."""
    return 1j * SQPI * wofz(zeta)


def eps_c(zeta, Lam):
    return 1.0 - (1.0 + zeta * Zc(zeta)) / Lam


def eR(u, Lam):
    u = np.asarray(u, dtype=float)
    return 1.0 - (1.0 + u * ReZ(u)) / Lam


def eI(u, Lam):
    u = np.asarray(u, dtype=float)
    return -np.pi * u * FM(u) / Lam


def rho(u, Lam):
    return FM(u) / (eR(u, Lam) ** 2 + eI(u, Lam) ** 2)


class Gauss:
    """Test function G(u) = sum_i c_i F_M((u - a_i)/s_i)/s_i  (s_i < sqrt2 keeps G^2/F_M integrable)."""

    def __init__(self, terms):
        self.terms = [tuple(map(float, t)) for t in terms]

    def __call__(self, u):
        u = np.asarray(u, dtype=float)
        return sum(c * FM((u - a) / s) / s for c, a, s in self.terms)

    def PV(self, u):
        """P int G(v)/(v - u) dv (exact): P int F_M((v-a)/s)/s/(v-u) dv = ReZ((u-a)/s)/s."""
        u = np.asarray(u, dtype=float)
        return sum(c * ReZ((u - a) / s) / s for c, a, s in self.terms)

    def Cz(self, z):
        """int G(v)/(v - z) dv for Im z > 0 (exact): sum c Z((z-a)/s)/s."""
        return sum(c * Zc((z - a) / s) / s for c, a, s in self.terms)

    def M0(self):
        return sum(c for c, a, s in self.terms)

    def M1(self):
        return sum(c * a for c, a, s in self.terms)

    def hermite(self, M):
        """Exact g_m, m = 0..M.  Generating function: sum_m (int H_m G du) t^m/m! = exp(2at + (s^2-1) t^2) for one
        term, so h_{m+1} = 2a h_m + 2(s^2-1) m h_{m-1} for h_m = int H_m G du, and with g_m = h_m/sqrt(2^m m!):
        g_0 = 1, g_1 = sqrt2 a, g_{m+1} = (sqrt2 a g_m + (s^2 - 1) sqrt(m) g_{m-1}) / sqrt(m+1)."""
        out = np.zeros(M + 1)
        for c, a, s in self.terms:
            k = s * s - 1.0
            g = np.zeros(M + 1)
            g[0] = 1.0
            if M >= 1:
                g[1] = np.sqrt(2.0) * a
            for n in range(1, M):
                g[n + 1] = (np.sqrt(2.0) * a * g[n] + k * np.sqrt(n) * g[n - 1]) / np.sqrt(n + 1.0)
            out += c * g
        return out


def T_of(G, u, Lam):
    """Slice transform T[G](u) = eR G/F_M + (u/Lambda) P int G/(v-u) dv (pointwise)."""
    u = np.asarray(u, dtype=float)
    return eR(u, Lam) * G(u) / FM(u) + (u / Lam) * G.PV(u)


def FMT_of(G, u, Lam):
    """F_M T[G] = eR G - eI Hc[G] (no division by F_M; safe at large |u|)."""
    u = np.asarray(u, dtype=float)
    return eR(u, Lam) * G(u) + (u / Lam) * FM(u) * G.PV(u)


def rhoT2(G, u, Lam):
    """rho T^2 = (F_M T)^2 / (F_M |eps|^2)."""
    ft = FMT_of(G, u, Lam)
    return ft ** 2 / (FM(u) * (eR(u, Lam) ** 2 + eI(u, Lam) ** 2))


def ladder(M, Lam, closure="zero"):
    """Matrix of L in the Hermite coefficients (zero closure g_{M+1} = 0): A[m,m+1] = A[m+1,m] = sqrt((m+1)/2),
    except A[1,0] = (1 - 1/Lambda)/sqrt2.  Lam = np.inf allowed."""
    A = np.zeros((M + 1, M + 1))
    for m in range(M):
        A[m, m + 1] = np.sqrt((m + 1) / 2.0)
        A[m + 1, m] = np.sqrt((m + 1) / 2.0)
    invL = 0.0 if np.isinf(Lam) else 1.0 / Lam
    A[1, 0] = (1.0 - invL) / np.sqrt(2.0)
    if closure == "symmetric":
        A[M, M - 1] += np.sqrt((M + 1) / 2.0)
    return A


def P0diag(M, Lam):
    invL = 0.0 if np.isinf(Lam) else 1.0 / Lam
    p = np.ones(M + 1)
    p[0] = 1.0 - invL
    return p


def jacobi(M, Lam):
    """Symmetrised ladder J = P0^{1/2} A P0^{-1/2} (zero closure): diagonal 0, off-diagonal
    b_0 = sqrt((1 - 1/Lambda)/2), b_m = sqrt((m+1)/2) for m >= 1.  Requires 1 - 1/Lambda > 0."""
    invL = 0.0 if np.isinf(Lam) else 1.0 / Lam
    b = np.sqrt((np.arange(M) + 1) / 2.0)
    b[0] = np.sqrt((1.0 - invL) / 2.0)
    return b


def gauss_rule(M, Lam):
    """Nodes u_j = eigenvalues of A (= of J) and Christoffel weights lam_j = mu0 v_{j,0}^2 (Golub-Welsch),
    mu0 = int rho du = Lambda/(Lambda - 1); returns (u, lam, V) with V the orthonormal eigenvectors of J."""
    from scipy.linalg import eigh_tridiagonal
    b = jacobi(M, Lam)
    u, V = eigh_tridiagonal(np.zeros(M + 1), b)
    mu0 = 1.0 / (1.0 - (0.0 if np.isinf(Lam) else 1.0 / Lam))
    return u, mu0 * V[0, :] ** 2, V
