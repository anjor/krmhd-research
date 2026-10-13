"""s08: the RMHD conventions taken from Chandran, Mallet & Meyrand (2026; CMM below) and GANDALF (z+ propagates
toward -z and z- toward +z, confirmed on GANDALF's right-hand side), the fluid limit (the slow-wave Elsasser fields
of Schekochihin et al. (2009) obey the slice equation at s u = -+c and the entropy mode obeys it at s u = 0), and the
slices at s u = +-1, which are advected by a single Elsasser field. It supports Section 2 (conventions) and Section 7
(fluid slices and the slices at s u = +-1) of the paper.

Checks:
 E1  CMM (3.23) gkeKRMHD and (3.24) vorticity with phi = 2 Phi/(rho_i v_th), A = -Psi/(rho_i v_A), v_th^2 = beta_i v_A^2 give
     d_t Psi = v_A grad_par Phi and d(lap Phi)/dt = v_A grad_par lap Psi.  Linearised, z+- = Phi +- Psi obey
     d_t z+- = +- v_A d_z z+-: z+ propagates toward -z, z- toward +z.  GANDALF agrees: krmhd_rhs on a pure z+ wave gives
     d_t z+ = +i k_z v_A z+.
 E2  Schekochihin et al. (2009) eqs (26)-(27) with z_par+- = u_par +- sqrt(1 + v_A^2/c_s^2) v_A dB_par/B0 (their
     eq. 29) give dz+-/dt -+ c bhat.grad z+- = 0, c = (1 + v_A^2/c_s^2)^{-1/2}; expanded this is their eq. (30), and it
     is the slice equation d_t a + {Phi + s u Psi, a} + s u d_z a = 0 with s u = -+ c.  The entropy mode (their eq. 23,
     d ds/dt = 0) is s u = 0.
 E3  Phi + s u Psi = [(1+su) z+ + (1-su) z-]/2: at s u = +1 the slice is advected by z+ only and streams toward +z, i.e. it
     co-moves with z- (which propagates toward +z); at s u = -1 it is advected by z- only and co-moves with z+.
Exit 1 on failure.
"""
import os
import sys
import numpy as np
import sympy as sp

fails = []


def check(name, cond, info=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info}", flush=True)
    if not cond:
        fails.append(name)


x, y, z, t = sp.symbols('x y z t', real=True)
rho_i, vth, vA, beta = sp.symbols('rho_i v_th v_A beta_i', positive=True)
pb = lambda f, g: sp.diff(f, x) * sp.diff(g, y) - sp.diff(f, y) * sp.diff(g, x)
lap = lambda f: sp.diff(f, x, 2) + sp.diff(f, y, 2)

print("E1  RMHD conventions (CMM) and direction of propagation")
Phi = sp.Function('Phi')(x, y, z, t)
Psi = sp.Function('Psi')(x, y, z, t)
rho_ = rho_i
phi = 2 * Phi / (rho_ * vth)
A = -Psi / (rho_ * vA)
# CMM (3.3)-(3.4): d/dt = d_t + (rho v_th/2){phi, .} ; grad_par = d_z - rho {A, .}
ddt = lambda f: sp.diff(f, t) + rho_ * vth / 2 * pb(phi, f)
gpar = lambda f: sp.diff(f, z) - rho_ * pb(A, f)
check("CMM d/dt = d_t + {Phi, .}", sp.simplify(ddt(Psi) - (sp.diff(Psi, t) + pb(Phi, Psi))) == 0)
check("CMM grad_par = d_z + (1/v_A){Psi, .}", sp.simplify(gpar(Phi) - (sp.diff(Phi, z) + pb(Psi, Phi) / vA)) == 0)
ind = sp.diff(A, t) + vth / 2 * gpar(phi)                                # (3.23) = 0
vort = sp.Rational(1, 2) * ddt(lap(phi)) + vth / beta * gpar(lap(A))     # (3.24) = 0
# solve (3.23) for d_t Psi and (3.24) for d_t lap Phi
dPsi = sp.solve(sp.Eq(ind, 0), sp.diff(Psi, t))[0]
check("(3.23) -> d_t Psi = v_A grad_par Phi", sp.simplify(dPsi - vA * (sp.diff(Phi, z) + pb(Psi, Phi) / vA)) == 0)
vort_sub = sp.simplify(vort.subs(beta, vth**2 / vA**2) * rho_ * vth)   # v_th^2 = beta v_A^2
target = sp.simplify(ddt(lap(Phi)) - vA * (sp.diff(lap(Psi), z) + pb(Psi, lap(Psi)) / vA))
check("(3.24) with v_th^2 = beta_i v_A^2 -> d(lap Phi)/dt = v_A grad_par lap Psi", sp.simplify(vort_sub - target) == 0)
f = sp.Function('f')
zp_wave = f(z + vA * t)
zm_wave = f(z - vA * t)
check("linear: d_t z+ = +v_A d_z z+ is solved by z+ = f(z + v_A t) (toward -z)", sp.simplify(sp.diff(zp_wave, t) - vA * sp.diff(zp_wave, z)) == 0)
check("linear: d_t z- = -v_A d_z z- is solved by z- = f(z - v_A t) (toward +z)", sp.simplify(sp.diff(zm_wave, t) + vA * sp.diff(zm_wave, z)) == 0)
# GANDALF: pure z+ wave
try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from krmhd.physics import KRMHDState
    from krmhd.spectral import SpectralGrid3D
    from krmhd.timestepping import krmhd_rhs
    Ng = 8
    grid = SpectralGrid3D.create(Nx=Ng, Ny=Ng, Nz=Ng)
    shp = (Ng, Ng, Ng // 2 + 1)
    zp = np.zeros(shp, complex); zm = np.zeros(shp, complex)
    iz, iy, ix = 1, 0, 1                                     # k_z = 1, k_y = 0, k_x = 1
    zp[iz, iy, ix] = 1.0 + 0.5j
    g = np.zeros(shp + (5,), complex)
    st = KRMHDState(z_plus=jnp.asarray(zp), z_minus=jnp.asarray(zm), g=jnp.asarray(g), M=4, beta_i=1.0, v_th=1.0,
                    nu=0.0, Lambda=float(np.sqrt(5.0)), time=0.0, grid=grid)
    d = krmhd_rhs(st, eta=0.0, v_A=1.0)
    kz = float(np.asarray(grid.kz)[iz])
    ratio = complex(np.asarray(d.z_plus)[iz, iy, ix] / zp[iz, iy, ix])
    check("GANDALF: d_t z+ = +i k_z v_A z+ for a pure z+ wave (z+ propagates toward -z)", abs(ratio - 1j * kz) < 1e-12, f"ratio = {ratio:.3f}, k_z = {kz}")
    zm2 = zp.copy(); zp2 = np.zeros(shp, complex)
    st2 = KRMHDState(z_plus=jnp.asarray(zp2), z_minus=jnp.asarray(zm2), g=jnp.asarray(g), M=4, beta_i=1.0, v_th=1.0,
                     nu=0.0, Lambda=float(np.sqrt(5.0)), time=0.0, grid=grid)
    d2 = krmhd_rhs(st2, eta=0.0, v_A=1.0)
    ratio2 = complex(np.asarray(d2.z_minus)[iz, iy, ix] / zm2[iz, iy, ix])
    check("GANDALF: d_t z- = -i k_z v_A z- (z- propagates toward +z)", abs(ratio2 + 1j * kz) < 1e-12, f"ratio = {ratio2:.3f}")
except Exception as e:
    import traceback
    traceback.print_exc()
    check(f"GANDALF direction check ran ({e})", False)

print("E2  slow-wave Elsasser fields and the entropy mode (Schekochihin et al. 2009, sec. 2.4-2.5)")
cs = sp.Symbol('c_s', positive=True)
upar = sp.Function('u')(x, y, z, t)
dB = sp.Function('b')(x, y, z, t)                   # dB_par/B0
vA1 = 1                                              # v_A = 1 units
ddt1 = lambda F: sp.diff(F, t) + pb(Phi, F)
bgrad = lambda F: sp.diff(F, z) + pb(Psi, F)        # bhat.grad with v_A = 1
eq26 = ddt1(dB) - bgrad(upar) / (1 + vA1**2 / cs**2)          # Schekochihin et al. (2009) eq. (26) = 0
eq27 = ddt1(upar) - vA1**2 * bgrad(dB)                         # Schekochihin et al. (2009) eq. (27) = 0
fac = sp.sqrt(1 + vA1**2 / cs**2)
c = 1 / fac
zpar_p = upar + fac * vA1 * dB
zpar_m = upar - fac * vA1 * dB
# use (26),(27) to eliminate time derivatives
sol = {sp.diff(dB, t): sp.solve(eq26, sp.diff(dB, t))[0], sp.diff(upar, t): sp.solve(eq27, sp.diff(upar, t))[0]}
for sgn, zf in ((+1, zpar_p), (-1, zpar_m)):
    res = sp.simplify((ddt1(zf) - sgn * c * bgrad(zf)).subs(sol))
    check(f"Schekochihin et al. 2009 eq. (29): d z{'+' if sgn > 0 else '-'}/dt {'-' if sgn > 0 else '+'} c bhat.grad z = 0", res == 0)
    # expanded form, Schekochihin et al. (2009) eq. (30): d_t z -+ c d_z z = -(1/2)(1 -+ c){zeta+, z} - (1/2)(1 +- c){zeta-, z}
    zeta_p = Phi + Psi; zeta_m = Phi - Psi
    form30 = sp.diff(zf, t) - sgn * c * sp.diff(zf, z) + sp.Rational(1, 2) * (1 - sgn * c) * pb(zeta_p, zf) + sp.Rational(1, 2) * (1 + sgn * c) * pb(zeta_m, zf)
    check(f"   ... equals Schekochihin et al. 2009 eq. (30) for z{'+' if sgn > 0 else '-'}", sp.simplify((form30 - (ddt1(zf) - sgn * c * bgrad(zf))).subs(sol)) == 0)
    # slice equation at s u = -sgn c
    su = -sgn * c
    slice_eq = sp.diff(zf, t) + pb(Phi + su * Psi, zf) + su * sp.diff(zf, z)
    check(f"   ... equals the slice equation with s u = {'-' if sgn > 0 else '+'}c", sp.simplify(slice_eq - (ddt1(zf) - sgn * c * bgrad(zf))) == 0)
check("entropy mode d ds/dt = 0 is the slice equation at s u = 0", True if sp.simplify(pb(Phi + 0 * Psi, upar) - pb(Phi, upar)) == 0 else False)
check("c = (1 + v_A^2/c_s^2)^{-1/2} -> 1 as c_s -> inf (high beta), -> c_s/v_A as c_s -> 0",
      sp.limit(c, cs, sp.oo) == 1 and sp.limit(c / cs, cs, 0) == 1)

print("E3  slices advected by one Elsasser field")
s, u = sp.symbols('s u', real=True)
zpS, zmS = sp.symbols('zp zm')
adv = lambda su: sp.expand((zpS + zmS) / 2 + su * (zpS - zmS) / 2)
check("s u = +1: Phi + s u Psi = z+", sp.simplify(adv(1) - zpS) == 0)
check("s u = -1: Phi + s u Psi = z-", sp.simplify(adv(-1) - zmS) == 0)
a = sp.Function('a')
slice_wave = a(z - s * u * t)
check("free streaming: a(z - s u t) solves d_t a + s u d_z a = 0 (s u = +1 moves toward +z, with z-)",
      sp.simplify(sp.diff(slice_wave, t) + s * u * sp.diff(slice_wave, z)) == 0)

if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("ALL s08 CHECKS PASSED")
