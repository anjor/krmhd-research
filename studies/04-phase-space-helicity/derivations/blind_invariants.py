# Provenance: written by a fresh agent context that saw only the collisionless g_m
# hierarchy and the quadratic ansatz (no access to the repo, the paper, or the
# study documents). Blind rediscovery test, 2026-10-01. See docs/rediscovery.md.
# Run: uv run python studies/04-phase-space-helicity/derivations/blind_invariants.py

"""Self-checking derivation of all tridiagonal quadratic invariants of the
truncated Hermite streaming hierarchy

    d/dt g = -s D (A g),   D antisymmetric (ik_z at single k), A real tridiagonal:
        A[0,1] = 1/sqrt2,  A[1,0] = c,
        A[m,m+1] = sqrt((m+1)/2),  A[m+1,m] = sqrt((m+1)/2)  (m >= 1),
        A[M,M+1] absent (g_{M+1} = 0).

Quadratic form Q = g^dagger P g with P real symmetric tridiagonal,
    P[m,m] = b_m,   P[m,m+1] = P[m+1,m] = a_m / 2   (so a_m Re(g_{m+1} g_m^*)).

Conservation condition:  P A = (P A)^T, i.e. C := P A - A^T P = 0.

Claimed complete solution (M >= 2, any real c):
    Q1 (free energy):   b_0 = sqrt2*c, b_m = 1 (m>=1), a_m = 0
    Q2 (v_par-weighted): a_0 = 2c, a_m = sqrt(2(m+1)) (m>=1), b_m = 0
Exits non-zero if any check fails.
"""
from __future__ import annotations

import sys

import numpy as np
import sympy as sp

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {msg}")
    if not cond:
        FAILS.append(msg)


# ----------------------------------------------------------------------------
# Matrices
# ----------------------------------------------------------------------------
def A_matrix(M: int, c, closure: str = "zero", sym: bool = False):
    """Coupling matrix A of the hierarchy d/dt g = -s D (A g).

    closure='zero' : g_{M+1} = 0
    closure='copy' : g_{M+1} = g_{M-1}  (adds sqrt((M+1)/2) to A[M,M-1])
    """
    sqrt = sp.sqrt if sym else np.sqrt
    A = sp.zeros(M + 1, M + 1) if sym else np.zeros((M + 1, M + 1))
    A[0, 1] = 1 / sqrt(2) if sym else 1 / np.sqrt(2.0)
    A[1, 0] = c
    for m in range(1, M + 1):
        if m + 1 <= M:
            A[m, m + 1] = sqrt(sp.Rational(m + 1, 2)) if sym else np.sqrt((m + 1) / 2)
        A[m, m - 1] = (sqrt(sp.Rational(m, 2)) if sym else np.sqrt(m / 2)) if m >= 2 else A[m, m - 1]
    if closure == "copy":
        A[M, M - 1] = A[M, M - 1] + (sqrt(sp.Rational(M + 1, 2)) if sym else np.sqrt((M + 1) / 2))
    return A


def P_from_ab(a, b, sym: bool = False):
    M = len(b) - 1
    P = sp.zeros(M + 1, M + 1) if sym else np.zeros((M + 1, M + 1))
    for m in range(M + 1):
        P[m, m] = b[m]
    for m in range(M):
        P[m, m + 1] = a[m] / 2
        P[m + 1, m] = a[m] / 2
    return P


def invariant_Q1(M: int, c, sym: bool = False):
    """Free energy: b_0 = sqrt2 c, b_m = 1 (m>=1), a_m = 0."""
    b = [sp.sqrt(2) * c if sym else np.sqrt(2.0) * c] + [1] * M
    a = [0] * M
    return a, b


def invariant_Q2(M: int, c, sym: bool = False):
    """v_par-weighted energy: a_0 = 2c, a_m = sqrt(2(m+1)) (m>=1), b_m = 0."""
    a = [2 * c] + [sp.sqrt(2 * (m + 1)) if sym else np.sqrt(2.0 * (m + 1)) for m in range(1, M)]
    b = [0] * (M + 1)
    return a, b


# ----------------------------------------------------------------------------
# Step 3a: SymPy solve for concrete M = 6, symbolic c
# ----------------------------------------------------------------------------
print("=" * 72)
print("STEP 3a: SymPy solution of P A = (P A)^T for M = 6, symbolic c")
print("=" * 72)
c = sp.symbols("c", real=True)
M = 6
bs = sp.symbols(f"b0:{M+1}", real=True)
ps = sp.symbols(f"p0:{M}", real=True)  # p_m = a_m / 2
A6 = A_matrix(M, c, sym=True)
P6 = P_from_ab([2 * p for p in ps], list(bs), sym=True)
C6 = sp.simplify(P6 * A6 - A6.T * P6)
eqs = [sp.simplify(C6[i, j]) for i in range(M + 1) for j in range(i + 1, M + 1)]
eqs = [e for e in eqs if e != 0]
print("non-trivial constraint equations:")
for e in eqs:
    print("   ", e, "= 0")
unknowns = list(bs) + list(ps)
Mat, rhs = sp.linear_eq_to_matrix(eqs, unknowns)
null = Mat.nullspace()
print(f"rank of constraint matrix = {Mat.rank()},  #unknowns = {len(unknowns)},  "
      f"nullspace dim = {len(null)}")
check(len(null) == 2, "M=6 symbolic c: exactly 2 independent invariants")

# Express the general solution and compare with the closed form.
sol = sp.solve(eqs, unknowns, dict=True)
assert len(sol) == 1
sol = sol[0]
free = [u for u in unknowns if u not in sol]
print("free parameters:", free)
general = {u: sp.simplify(sol.get(u, u)) for u in unknowns}
for u in unknowns:
    print(f"   {u} = {general[u]}")

# Closed-form claim: b_m = B for m>=1, b_0 = sqrt2 c B ; p_m = K sqrt((m+1)/2) (m>=1), p_0 = cK
B, K = sp.symbols("B K", real=True)
claim = {bs[0]: sp.sqrt(2) * c * B}
claim.update({bs[m]: B for m in range(1, M + 1)})
claim[ps[0]] = c * K
claim.update({ps[m]: K * sp.sqrt(sp.Rational(m + 1, 2)) for m in range(1, M)})
Pclaim = P6.subs(claim)
Cclaim = sp.simplify(Pclaim * A6 - A6.T * Pclaim)
check(Cclaim == sp.zeros(M + 1, M + 1), "M=6: closed-form 2-parameter family satisfies PA = (PA)^T")
# and the general solution lies in the span of the claim (nullspace dim already 2 and claim is 2-dim):
v1 = sp.Matrix([claim[u].subs({B: 1, K: 0}) for u in unknowns])
v2 = sp.Matrix([claim[u].subs({B: 0, K: 1}) for u in unknowns])
check(sp.Matrix.hstack(v1, v2).rank() == 2, "M=6: closed-form vectors are independent (span the solution space)")

# Also check at the special values c = 0 (Lambda = 1) and c = 1/sqrt2 (Lambda -> inf) the dim is still 2.
for cval, label in [(0, "c=0 (Lambda=1)"), (1 / sp.sqrt(2), "c=1/sqrt2 (Lambda->inf)"), (sp.Rational(-3, 7), "c=-3/7")]:
    Mc, _ = sp.linear_eq_to_matrix([e.subs(c, cval) for e in eqs], unknowns)
    check(len(Mc.nullspace()) == 2, f"M=6, {label}: nullspace dim = 2")

# Nullspace dimension for a range of M (numeric c values), to show M-independence.
print()
print("nullspace dimension vs M (rational c = 3/5):")
for Mt in range(2, 13):
    bt = sp.symbols(f"b0:{Mt+1}", real=True)
    pt = sp.symbols(f"p0:{Mt}", real=True)
    At = A_matrix(Mt, sp.Rational(3, 5), sym=True)
    Pt = P_from_ab([2 * p for p in pt], list(bt), sym=True)
    Ct = Pt * At - At.T * Pt
    et = [sp.simplify(Ct[i, j]) for i in range(Mt + 1) for j in range(i + 1, Mt + 1)]
    et = [e for e in et if e != 0]
    Mt_mat, _ = sp.linear_eq_to_matrix(et, list(bt) + list(pt))
    nd = len(Mt_mat.nullspace())
    print(f"   M={Mt:2d}: dim = {nd}")
    check(nd == 2, f"M={Mt}: nullspace dim = 2")

# ----------------------------------------------------------------------------
# Step 3b: general m. The symmetry condition decouples into two recursions
#   (PA)_{m,m+1} = (PA)_{m+1,m}:  b_m alpha_m = b_{m+1} beta_m
#   (PA)_{m,m+2} = (PA)_{m+2,m}:  p_m alpha_{m+1} = p_{m+1} beta_m
# with alpha_m = sqrt((m+1)/2), beta_0 = c, beta_m = sqrt((m+1)/2) (m>=1).
# Verify the closed forms satisfy them for symbolic m >= 1.
# ----------------------------------------------------------------------------
print()
print("=" * 72)
print("STEP 3b: closed forms satisfy the recursions for symbolic m >= 1")
print("=" * 72)
m = sp.symbols("m", integer=True, positive=True)
alpha = lambda j: sp.sqrt((j + 1) / sp.Integer(2))
beta = lambda j: sp.sqrt((j + 1) / sp.Integer(2))  # j >= 1
b_m = lambda j: B
p_m = lambda j: K * sp.sqrt((j + 1) / sp.Integer(2))
r1 = sp.simplify(b_m(m) * alpha(m) - b_m(m + 1) * beta(m))
r2 = sp.simplify(p_m(m) * alpha(m + 1) - p_m(m + 1) * beta(m))
print("   b-recursion residual (m>=1):", r1)
print("   p-recursion residual (m>=1):", r2)
check(r1 == 0 and r2 == 0, "symbolic m: both recursions satisfied by closed forms")
# m = 0 boundary: b_0 alpha_0 = b_1 c  and  p_0 alpha_1 = p_1 c
r3 = sp.simplify(sp.sqrt(2) * c * B * alpha(0) - B * c)
r4 = sp.simplify(c * K * alpha(1) - K * c)
check(r3 == 0 and r4 == 0, "m=0 boundary conditions satisfied (b_0 = sqrt2 c B, p_0 = c K)")

# ----------------------------------------------------------------------------
# Step 5: numerical check, single k, D = i k_z
# ----------------------------------------------------------------------------
print()
print("=" * 72)
print("STEP 5: numerical dQ/dt at single k_z, random complex g")
print("=" * 72)
rng = np.random.default_rng(12345)


def dQdt(P: np.ndarray, A: np.ndarray, g: np.ndarray, kz: float, s: float) -> float:
    dg = -s * (1j * kz) * (A @ g)
    return float(2 * np.real(np.conj(g) @ P @ dg))


def scale(P: np.ndarray, A: np.ndarray, g: np.ndarray, kz: float, s: float) -> float:
    return abs(s * kz) * np.linalg.norm(P) * np.linalg.norm(A) * np.linalg.norm(g) ** 2


for Mn in (6, 20):
    for trial in range(3):
        cval = rng.normal()
        kz = rng.normal() * 3
        s = abs(rng.normal()) + 0.5
        g = rng.normal(size=Mn + 1) + 1j * rng.normal(size=Mn + 1)
        A = A_matrix(Mn, cval)
        sc = scale(np.eye(Mn + 1), A, g, kz, s)
        a1, b1 = invariant_Q1(Mn, cval)
        a2, b2 = invariant_Q2(Mn, cval)
        P1 = P_from_ab(a1, b1)
        P2 = P_from_ab(a2, b2)
        d1 = dQdt(P1, A, g, kz, s)
        d2 = dQdt(P2, A, g, kz, s)
        # generic tridiagonal quadratic form
        Pg = P_from_ab(rng.normal(size=Mn), rng.normal(size=Mn + 1))
        dg_ = dQdt(Pg, A, g, kz, s)
        # perturbed invariants (one coefficient off) must fail
        a1p, b1p = invariant_Q1(Mn, cval)
        b1p[0] = b1p[0] + 0.1
        d1p = dQdt(P_from_ab(a1p, b1p), A, g, kz, s)
        a2p, b2p = invariant_Q2(Mn, cval)
        a2p[0] = a2p[0] + 0.1
        d2p = dQdt(P_from_ab(a2p, b2p), A, g, kz, s)
        print(f"M={Mn:2d} c={cval:+.3f} kz={kz:+.3f}: |dQ1/dt|/scale={abs(d1)/sc:.2e}  "
              f"|dQ2/dt|/scale={abs(d2)/sc:.2e}  generic={abs(dg_)/sc:.2e}  "
              f"Q1(b0+0.1)={abs(d1p)/sc:.2e}  Q2(a0+0.1)={abs(d2p)/sc:.2e}")
        check(abs(d1) / sc < 1e-13, f"M={Mn} trial {trial}: Q1 conserved to round-off")
        check(abs(d2) / sc < 1e-13, f"M={Mn} trial {trial}: Q2 conserved to round-off")
        check(abs(dg_) / sc > 1e-4, f"M={Mn} trial {trial}: generic form NOT conserved")
        check(abs(d1p) / sc > 1e-6, f"M={Mn} trial {trial}: Q1 with wrong b_0 NOT conserved")
        check(abs(d2p) / sc > 1e-6, f"M={Mn} trial {trial}: Q2 with wrong a_0 NOT conserved")

# Also c = 0 (Lambda = 1): g_0 decouples as a passive scalar; Q1 has b_0 = 0, Q2 has a_0 = 0.
Mn = 6
A0 = A_matrix(Mn, 0.0)
g = rng.normal(size=Mn + 1) + 1j * rng.normal(size=Mn + 1)
sc = scale(np.eye(Mn + 1), A0, g, 1.3, 1.0)
for name, fn in (("Q1", invariant_Q1), ("Q2", invariant_Q2)):
    a_, b_ = fn(Mn, 0.0)
    d = dQdt(P_from_ab(a_, b_), A0, g, 1.3, 1.0)
    check(abs(d) / sc < 1e-13, f"c=0: {name} conserved")

# ----------------------------------------------------------------------------
# Step 4: truncation and alternative closure g_{M+1} = g_{M-1}
# ----------------------------------------------------------------------------
print()
print("=" * 72)
print("STEP 4: closure g_{M+1} = g_{M-1}")
print("=" * 72)
for Mn in (6, 20):
    cval = rng.normal()
    kz, s = 0.7, 1.1
    g = rng.normal(size=Mn + 1) + 1j * rng.normal(size=Mn + 1)
    Ac = A_matrix(Mn, cval, closure="copy")
    sc = scale(np.eye(Mn + 1), Ac, g, kz, s)
    a1, b1 = invariant_Q1(Mn, cval)
    a2, b2 = invariant_Q2(Mn, cval)
    d1 = dQdt(P_from_ab(a1, b1), Ac, g, kz, s)
    d2 = dQdt(P_from_ab(a2, b2), Ac, g, kz, s)
    # modified last coefficient: b_M = sqrt(M) / (sqrt(M) + sqrt(M+1))
    b1m = list(b1)
    b1m[Mn] = np.sqrt(Mn) / (np.sqrt(Mn) + np.sqrt(Mn + 1))
    d1m = dQdt(P_from_ab(a1, b1m), Ac, g, kz, s)
    print(f"M={Mn:2d} copy closure: |dQ1/dt|={abs(d1)/sc:.2e} (b_M=1, broken)  "
          f"|dQ1'/dt|={abs(d1m)/sc:.2e} (b_M=sqrtM/(sqrtM+sqrt(M+1)))  |dQ2/dt|={abs(d2)/sc:.2e}")
    check(abs(d1) / sc > 1e-6, f"M={Mn} copy closure: unmodified Q1 is NOT conserved")
    check(abs(d1m) / sc < 1e-13, f"M={Mn} copy closure: Q1 with modified b_M IS conserved")
    check(abs(d2) / sc < 1e-13, f"M={Mn} copy closure: Q2 unchanged and conserved")

# Symbolic confirmation of the copy-closure solution space for M = 6
Ac6 = A_matrix(6, c, closure="copy", sym=True)
Cc6 = sp.simplify(P6 * Ac6 - Ac6.T * P6)
eqc = [sp.simplify(Cc6[i, j]) for i in range(7) for j in range(i + 1, 7)]
eqc = [e for e in eqc if e != 0]
Mc6, _ = sp.linear_eq_to_matrix(eqc, unknowns)
check(len(Mc6.nullspace()) == 2, "M=6 copy closure, symbolic c: still exactly 2 invariants")
# pick the nullspace vector with p's = 0 (pure b-family) and read off b_6 / b_1
ib1, ib6 = unknowns.index(bs[1]), unknowns.index(bs[6])
ip = [unknowns.index(p) for p in ps]
bvecs = [v for v in Mc6.nullspace() if all(sp.simplify(v[i]) == 0 for i in ip)]
check(len(bvecs) == 1, "M=6 copy closure: exactly one pure-diagonal (b-family) invariant")
bM_ratio = sp.radsimp(sp.simplify(bvecs[0][ib6] / bvecs[0][ib1]))
target = sp.radsimp(sp.sqrt(6) / (sp.sqrt(6) + sp.sqrt(7)))
print("   copy closure: b_6 / b_1 =", bM_ratio, "   expected sqrt6/(sqrt6+sqrt7) =", target)
check(sp.simplify(bM_ratio - target) == 0,
      "M=6 copy closure: b_M/b_1 = sqrt(M)/(sqrt(M)+sqrt(M+1))")

# ----------------------------------------------------------------------------
# Step 1 (numerical support for the reduction): real-space model with
# {Phi,.} and nabla_par = d_z + {Psi,.} replaced by arbitrary real antisymmetric
# operators on N spatial points, acting identically on every moment.
# Q = sum_x g^T P g must be conserved for Q1, Q2 (and not for generic P).
# ----------------------------------------------------------------------------
print()
print("=" * 72)
print("STEP 1 check: full model with arbitrary antisymmetric {Phi,.} and nabla_par")
print("=" * 72)
Mn, N = 6, 9
cval = rng.normal()
s = 0.9
X = rng.normal(size=(N, N)); DPhi = X - X.T          # {Phi, .}
Y = rng.normal(size=(N, N)); Dpar = Y - Y.T          # d_z + {Psi, .}
A = A_matrix(Mn, cval)
G = rng.normal(size=(Mn + 1, N))                      # real fields g_m(x)
dG = -(G @ DPhi.T) - s * ((A @ G) @ Dpar.T)           # d/dt g_m(x)
def dQdt_full(P):
    return float(2 * np.sum(G * (P @ dG)))
sc = np.linalg.norm(A) * (np.linalg.norm(DPhi) + s * np.linalg.norm(Dpar)) * np.linalg.norm(G) ** 2
for name, fn in (("Q1", invariant_Q1), ("Q2", invariant_Q2)):
    a_, b_ = fn(Mn, cval)
    d = dQdt_full(P_from_ab(a_, b_))
    print(f"   {name}: |dQ/dt|/scale = {abs(d)/sc:.2e}")
    check(abs(d) / sc < 1e-13, f"full model: {name} conserved with Phi-advection and Psi-bent nabla_par")
dgen = dQdt_full(P_from_ab(rng.normal(size=Mn), rng.normal(size=Mn + 1)))
print(f"   generic: |dQ/dt|/scale = {abs(dgen)/sc:.2e}")
check(abs(dgen) / sc > 1e-4, "full model: generic form NOT conserved")
# Phi-term alone contributes nothing for ANY P:
dG_phi = -(G @ DPhi.T)
Pg = P_from_ab(rng.normal(size=Mn), rng.normal(size=Mn + 1))
dphi = float(2 * np.sum(G * (Pg @ dG_phi)))
check(abs(dphi) / sc < 1e-13, "full model: {Phi,.} term contributes zero for a generic P")

# ----------------------------------------------------------------------------
print()
if FAILS:
    print(f"{len(FAILS)} CHECK(S) FAILED:")
    for f in FAILS:
        print("   ", f)
    sys.exit(1)
print("ALL CHECKS PASSED")
