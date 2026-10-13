"""c07 (independent check): the fluid slices and the slices at s u = +-1. The speed c is the slow-mode phase speed
of ideal MHD at small k_par/k; the slow-wave equations of Schekochihin et al. (2009) diagonalise into Elsasser
fields that obey the slice equation at s u = -+c, and the entropy mode obeys it at u = 0; and in GANDALF's
right-hand side the slice with s u_j = +1 (resp. -1) does not depend on z- (resp. z+) while the other slices do. It
supports Section 7 of the paper.

 F1  c is the slow-mode phase speed: ideal-MHD dispersion relation w^4 - k^2 (cs^2 + vA^2) w^2 + k^2 kpar^2 cs^2 vA^2 = 0 gives
     w^2/kpar^2 -> cs^2 vA^2/(cs^2 + vA^2) as kpar/k -> 0, i.e. c = (1 + vA^2/cs^2)^{-1/2} (vA = 1).
 F2  Schekochihin et al. (2009) eqs (26)-(27) as printed: d_t X = M bgrad X, X = (dB/B0, u_par), M = [[0, k],[1, 0]],
     k = 1/(1 + vA^2/cs^2).  Left eigenvectors of M give z_par^+- = u_par +- sqrt(1 + vA^2/cs^2) dB/B0 (their eq. 29)
     with
     d z^+-/dt = +-c bhat.grad z^+-, which is the slice equation d_t a + {Phi + s u Psi, a} + s u d_z a = 0 at s u = -+c, and the
     advecting potential is [(1 -+ c) zeta+ + (1 +- c) zeta-]/2 (their eq. 30). Entropy mode (their eq. 23) = slice
     equation at u = 0.
 F3  the slices at s u = +-1 in GANDALF: choose beta so that a node u_j of the ladder matrix has s u_j = +1
     (resp. -1); then the j-th slice of
     krmhd_rhs's dg/dt does not depend on z- (resp. z+), while the other slices do.
Exit 1 on any failure.
"""
import sys
import numpy as np
import sympy as sp

fails = []


def check(name, ok, info=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)
    if not ok:
        fails.append(name)


print("F1  slow-mode phase speed")
w2, k, kp, cs = sp.symbols('w2 k kpar c_s', positive=True)
disp = w2**2 - k**2 * (cs**2 + 1) * w2 + k**2 * kp**2 * cs**2
eps = sp.Symbol('epsilon', positive=True)                  # kpar = eps k
A_ = cs**2 + 1
slow = (k**2 * A_ - sp.sqrt(k**4 * A_**2 - 4 * k**2 * (eps * k)**2 * cs**2)) / 2          # smaller root of disp in w2
check("the slow root solves the dispersion relation", sp.simplify(disp.subs({w2: slow, kp: eps * k})) == 0)
ser = sp.series(sp.simplify(slow / (eps * k)**2), eps, 0, 2).removeO()
c_expected = 1 / (1 + 1 / cs**2)
check("w^2/kpar^2 -> cs^2/(1 + cs^2) = c^2 with c = (1 + vA^2/cs^2)^{-1/2}", sp.simplify(ser.subs(eps, 0) - c_expected) == 0,
      f"leading term = {sp.simplify(ser.subs(eps, 0))}")
num = slow.subs({k: 1, eps: sp.Rational(1, 1000), cs: sp.Rational(7, 10)}) / sp.Rational(1, 1000)**2
check("numerical spot check at kpar/k = 1e-3, cs = 0.7", abs(float(num) - float(c_expected.subs(cs, sp.Rational(7, 10)))) < 1e-5,
      f"{float(num):.8f} vs {float(c_expected.subs(cs, sp.Rational(7, 10))):.8f}")

print("F2  Schekochihin et al. 2009 eqs (26)-(27) diagonalised")
x, y, z, t = sp.symbols('x y z t', real=True)
Phi = sp.Function('Phi')(x, y, z, t); Psi = sp.Function('Psi')(x, y, z, t)
bB = sp.Function('b')(x, y, z, t); up = sp.Function('u')(x, y, z, t)
pb = lambda f, g: sp.diff(f, x) * sp.diff(g, y) - sp.diff(f, y) * sp.diff(g, x)
ddt = lambda f: sp.diff(f, t) + pb(Phi, f)
bgrad = lambda f: sp.diff(f, z) + pb(Psi, f)
kap = 1 / (1 + 1 / cs**2)
c = sp.sqrt(kap)
Mmat = sp.Matrix([[0, kap], [1, 0]])
ev = Mmat.T.eigenvects()                                   # left eigenvectors of M
sol = {sp.diff(bB, t): kap * bgrad(up) - pb(Phi, bB), sp.diff(up, t): bgrad(bB) - pb(Phi, up)}   # Schekochihin et al. (2009) eqs (26), (27)
ok = True
found = []
for val, mult, vecs in ev:
    l = vecs[0] / vecs[0][1]                               # normalise the u_par coefficient to 1
    zf = l[0] * bB + l[1] * up
    # compare with Schekochihin et al. (2009) eq. (29): u_par +- sqrt(1 + vA^2/cs^2) dB/B0
    sgn = 1 if sp.simplify(val - c) == 0 else -1
    ok &= sp.simplify(l[0] - sgn * sp.sqrt(1 + 1 / cs**2)) == 0
    res = sp.simplify((ddt(zf) - val * bgrad(zf)).subs(sol))
    ok &= res == 0
    su = -val
    slice_eq = sp.diff(zf, t) + pb(Phi + su * Psi, zf) + su * sp.diff(zf, z)
    ok &= sp.simplify((slice_eq - (ddt(zf) - val * bgrad(zf)))) == 0
    zp_, zm_ = Phi + Psi, Phi - Psi
    adv30 = ((1 - sgn * c) * zp_ + (1 + sgn * c) * zm_) / 2
    ok &= sp.simplify(sp.expand(Phi + su * Psi - adv30)) == 0
    found.append((sgn, sp.simplify(val)))
check("left eigenvectors = Schekochihin et al. 2009 eq. (29) fields; d z+-/dt = +-c bhat.grad z+-; = slice eq. at s u = -+c; potential = their eq. (30)", ok,
      f"eigenvalues {found}")
ds = sp.Function('ds')(x, y, z, t)
check("entropy mode d ds/dt = 0 (Schekochihin et al. 2009 eq. 23) is the slice equation at u = 0", sp.simplify(ddt(ds) - (sp.diff(ds, t) + pb(Phi + 0 * Psi, ds))) == 0)

print("F3  slices at s u = +-1 with GANDALF: the slice with s u_j = +-1 ignores z-+")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from krmhd.physics import KRMHDState
from krmhd.spectral import SpectralGrid3D
from krmhd.timestepping import krmhd_rhs
from krmhd.hermite import compute_streaming_matrix
N = 16; M = 12
grid = SpectralGrid3D.create(Nx=N, Ny=N, Nz=N)
mask = np.asarray(grid.dealias_mask)
rng = np.random.default_rng(17)


def rf(amp):
    F = np.fft.rfftn(rng.standard_normal((N, N, N)), axes=(0, 1, 2)) * mask
    F[0, 0, 0] = 0
    return F / np.abs(np.fft.irfftn(F, s=(N, N, N), axes=(0, 1, 2))).max() * amp


for Lv in (np.sqrt(5.0), -np.sqrt(5.0)):
    T = np.asarray(compute_streaming_matrix(M, float(Lv), "zero"), dtype=float)
    lam_, W = np.linalg.eig(T.T); uj = lam_.real; W = W.real
    jpos = np.argmin(np.abs(uj - 1.3))                    # a positive node near 1.3
    beta = 1.0 / uj[jpos]**2                               # s u_j = 1
    gg = np.stack([rf(1.0) for _ in range(M + 1)], -1)
    zp, zm1, zm2 = rf(12.0), rf(9.0), rf(9.0)
    rhs = []
    for zm in (zm1, zm2):
        st = KRMHDState(z_plus=jnp.asarray(zp), z_minus=jnp.asarray(zm), g=jnp.asarray(gg), M=M, beta_i=float(beta), v_th=1.0,
                        nu=0.0, Lambda=float(Lv), time=0.0, grid=grid)
        rhs.append(np.asarray(krmhd_rhs(st, eta=0.0, v_A=1.0, closure="zero").g) @ W)
    diff = np.abs(rhs[0] - rhs[1]).max(axis=(0, 1, 2)) / np.abs(rhs[0]).max(axis=(0, 1, 2))
    others = np.delete(diff, jpos)
    check(f"Lambda={Lv:+.3f}: s u_j = +1 (u_j = {uj[jpos]:.4f}, beta = {beta:.4f}): slice j independent of z- ", diff[jpos] < 1e-12 and others.min() > 1e-3,
          f"rel change {diff[jpos]:.1e}; other slices change by >= {others.min():.2f}")
    jneg = np.argmin(np.abs(uj + uj[jpos]))                # the mirror node -u_j: s u = -1
    zm = rf(12.0); zp1, zp2 = rf(9.0), rf(9.0)
    rhs = []
    for zp in (zp1, zp2):
        st = KRMHDState(z_plus=jnp.asarray(zp), z_minus=jnp.asarray(zm), g=jnp.asarray(gg), M=M, beta_i=float(beta), v_th=1.0,
                        nu=0.0, Lambda=float(Lv), time=0.0, grid=grid)
        rhs.append(np.asarray(krmhd_rhs(st, eta=0.0, v_A=1.0, closure="zero").g) @ W)
    diff = np.abs(rhs[0] - rhs[1]).max(axis=(0, 1, 2)) / np.abs(rhs[0]).max(axis=(0, 1, 2))
    others = np.delete(diff, jneg)
    check(f"Lambda={Lv:+.3f}: s u_j = -1 (u_j = {uj[jneg]:.4f}): slice j independent of z+ ", diff[jneg] < 1e-12 and others.min() > 1e-3,
          f"rel change {diff[jneg]:.1e}; other slices change by >= {others.min():.2f}")

if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("ALL c07 CHECKS PASSED")
