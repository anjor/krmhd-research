# Provenance: written by an independent critic context that saw only the equations
# and the claimed coefficients (not the derivation, the repo, or the paper).
# Falsification test on random band-limited 3D fields, 2026-10-01. See docs/rediscovery.md.
# Run: uv run python studies/04-phase-space-helicity/derivations/critic_invariants.py

"""Independent numerical falsification test for two claimed quadratic invariants
of the Hermite-moment hierarchy (numpy only, no time stepping).

dQ/dt = sum_grid (dQ/dg_m) * RHS_m.  Fields are band-limited to |n| <= N/4 so
triple products (|n| <= 3N/4 < N) are integrated exactly on the N^3 grid.
"""
from __future__ import annotations

import numpy as np

N = 16
NMAX = N // 4  # band limit per direction


def kvec(n: int) -> np.ndarray:
    return 2 * np.pi * np.fft.fftfreq(n, d=1.0 / n)


KX = KY = KZ = kvec(N)
KXg, KYg, KZg = np.meshgrid(KX, KY, KZ[: N // 2 + 1], indexing="ij")
IKX, IKY, IKZ = 1j * KXg, 1j * KYg, 1j * KZg


def random_field(rng: np.random.Generator) -> np.ndarray:
    """Real, smooth, band-limited random field with modes |n| <= NMAX."""
    c = np.zeros((N, N, N), dtype=complex)
    n = np.fft.fftfreq(N, d=1.0 / N).astype(int)
    mask = (np.abs(n)[:, None, None] <= NMAX) & (np.abs(n)[None, :, None] <= NMAX) & (np.abs(n)[None, None, :] <= NMAX)
    c[mask] = rng.normal(size=mask.sum()) + 1j * rng.normal(size=mask.sum())
    f = np.real(np.fft.ifftn(c)) * N**3 / N**1.5
    return f


def dx(f: np.ndarray) -> np.ndarray:
    return np.fft.irfftn(IKX * np.fft.rfftn(f), s=f.shape, axes=(0, 1, 2))


def dy(f: np.ndarray) -> np.ndarray:
    return np.fft.irfftn(IKY * np.fft.rfftn(f), s=f.shape, axes=(0, 1, 2))


def dz(f: np.ndarray) -> np.ndarray:
    return np.fft.irfftn(IKZ * np.fft.rfftn(f), s=f.shape, axes=(0, 1, 2))


def bracket(f: np.ndarray, h: np.ndarray) -> np.ndarray:
    """{f,h} = df/dx dh/dy - df/dy dh/dx (real-space pointwise product)."""
    return dx(f) * dy(h) - dy(f) * dx(h)


def rhs(g: list[np.ndarray], phi: np.ndarray, psi: np.ndarray, lam: float,
        closure: str, s: float = 1.0) -> list[np.ndarray]:
    M = len(g) - 1
    c = (1.0 - 1.0 / lam) / np.sqrt(2.0)
    if closure == "Z":
        gM1 = np.zeros_like(g[M])
    elif closure == "C":
        gM1 = g[M - 1].copy()
    elif closure == "G":
        gM1 = 2.7 * g[M - 1] + 0.3 * g[M]
    else:
        raise ValueError(closure)
    gext = g + [gM1]

    def gpar(h: np.ndarray) -> np.ndarray:
        return dz(h) + bracket(psi, h)

    out = []
    for m in range(M + 1):
        if m == 0:
            h = gext[1] / np.sqrt(2.0)
        elif m == 1:
            h = gext[2] + c * gext[0]
        else:
            h = np.sqrt((m + 1) / 2.0) * gext[m + 1] + np.sqrt(m / 2.0) * gext[m - 1]
        out.append(-bracket(phi, g[m]) - s * gpar(h))
    return out


def dQdt(g: list[np.ndarray], dg: list[np.ndarray], a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Q = sum_m a_m g_{m+1} g_m + sum_m b_m g_m^2.  Returns (dQ/dt, scale)."""
    M = len(g) - 1
    val = 0.0
    scale = 0.0
    for m in range(M + 1):
        grad = 2.0 * b[m] * g[m]
        if m < M:
            grad = grad + a[m] * g[m + 1]
        if m > 0:
            grad = grad + a[m - 1] * g[m - 1]
        val += np.sum(grad * dg[m])
        scale += np.sum(np.abs(grad) * np.abs(dg[m]))
    return float(val), float(scale)


def forms(M: int, lam: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    z = np.zeros(M)
    # Claim 1: W with Lambda-weighted g_0
    bW = np.ones(M + 1)
    bW[0] = 1.0 - 1.0 / lam
    # Claim 1, closure C fix: modify m=M weight
    bWC = bW.copy()
    bWC[M] = np.sqrt(M) / (np.sqrt(M) + np.sqrt(M + 1))
    # Claim 2: Gamma
    aG = np.array([np.sqrt(2.0 * (m + 1)) for m in range(M)])
    aG[0] = np.sqrt(2.0) * (1.0 - 1.0 / lam)
    # Negative controls (Claim 3)
    bE = np.ones(M + 1)  # unweighted energy, weight 1 on g_0
    aGbad = aG.copy()
    aGbad[0] = np.sqrt(2.0)  # p_0 = sqrt(2) without Lambda factor
    return {
        "W": (z, bW),
        "W_Cfix": (z, bWC),
        "Gamma": (aG, np.zeros(M + 1)),
        "E_unweighted": (z, bE),
        "Gamma_p0=sqrt2": (aGbad, np.zeros(M + 1)),
    }


def main() -> None:
    rows = []
    hdr = f"{'M':>2} {'Lam':>7} {'seed':>4} {'Phi':>3} {'Psi':>3} {'cl':>2} | " + " ".join(f"{k:>15}" for k in forms(5, 3.0))
    print(hdr)
    print("-" * len(hdr))
    worst: dict[tuple[str, str], tuple[float, str]] = {}
    for M in (5, 9):
        for lam in (np.sqrt(5.0), -np.sqrt(5.0), 3.0):
            for seed in (0, 1, 2):
                rng = np.random.default_rng(seed + 100 * M)
                g = [random_field(rng) for _ in range(M + 1)]
                phi_f = random_field(rng)
                psi_f = random_field(rng)
                for use_phi, use_psi in ((1, 1), (0, 1), (1, 0)):
                    if seed > 0 and (use_phi, use_psi) != (1, 1):
                        continue  # only vary Phi/Psi off for seed 0 to keep table short
                    phi = phi_f if use_phi else np.zeros_like(phi_f)
                    psi = psi_f if use_psi else np.zeros_like(psi_f)
                    for cl in ("Z", "C", "G"):
                        dg = rhs(g, phi, psi, lam, cl)
                        res = {}
                        for name, (a, b) in forms(M, lam).items():
                            v, sc = dQdt(g, dg, a, b)
                            r = abs(v) / sc
                            res[name] = r
                            key = (name, cl)
                            tag = f"M={M} Lam={lam:+.3f} seed={seed} Phi={use_phi} Psi={use_psi}"
                            if key not in worst or r > worst[key][0]:
                                worst[key] = (r, tag)
                        line = f"{M:>2} {lam:>7.3f} {seed:>4} {use_phi:>3} {use_psi:>3} {cl:>2} | " + " ".join(f"{res[k]:>15.3e}" for k in res)
                        print(line)
    print()
    print("Worst-case relative residual per (form, closure):")
    for (name, cl), (r, tag) in sorted(worst.items()):
        print(f"  {name:>15} closure {cl}: {r:.3e}   [{tag}]")
    print()
    TOL = 1e-10

    def ok(name: str, cl: str) -> bool:
        return worst[(name, cl)][0] < TOL

    print("VERDICTS (tolerance %.0e):" % TOL)
    c1 = ok("W", "Z") and (not ok("W", "C")) and ok("W_Cfix", "C")
    print(f"  Claim 1 (W conserved under Z; not under C; fixed weight conserved under C): {'SUPPORTED' if c1 else 'REFUTED'}")
    c2 = ok("Gamma", "Z") and ok("Gamma", "C")
    print(f"  Claim 2 (Gamma conserved under Z and C): {'SUPPORTED' if c2 else 'REFUTED'}")
    c3 = (not ok("E_unweighted", "Z")) and (not ok("Gamma_p0=sqrt2", "Z"))
    print(f"  Claim 3 negative controls (E_unweighted, Gamma with p0=sqrt2 NOT conserved under Z): {'SUPPORTED' if c3 else 'REFUTED'}")
    print(f"  Exploratory: Gamma under generic closure G (2.7 g_(M-1) + 0.3 g_M): {'CONSERVED' if ok('Gamma', 'G') else 'NOT conserved'} (worst {worst[('Gamma','G')][0]:.3e})")
    print(f"  Exploratory: W under closure G: {'CONSERVED' if ok('W', 'G') else 'NOT conserved'} (worst {worst[('W','G')][0]:.3e})")


if __name__ == "__main__":
    main()
