"""s06: the slice equation d_t a_u + {Phi + s u Psi, a_u} + s u d_z a_u = 0 in 3D. SymPy checks the identities
behind it (the advecting velocity U_u = u_E + s u bhat is divergence-free; the Elsasser form of Phi + s u Psi;
Casimir identities); an independent NumPy code checks that the slice projection of the Hermite right-hand side equals
the slice right-hand side, also with a velocity-independent hyperdiffusion, and that velocity-dependent collisions
couple slices; and the same projection is checked on GANDALF's own right-hand side krmhd_rhs. It supports Section 3
(the slice theorem; dissipation and collisions in slice variables) and Appendix B of the paper.

Branch equation (v_A = 1):  d_t G + {Phi, G} + s d_z(L G) + s {Psi, L G} = 0,  {f,g} = d_x f d_y g - d_y f d_x g.
The transform T acts on u only with real, (r,t)-independent kernel, so it commutes with {Phi,.}, {Psi,.}, d_z and with any
v-independent spatial operator; with T L = u T this gives for a_u = T[G](u):
   d_t a_u + {Phi + s u Psi, a_u} + s u d_z a_u = 0  <=>  d_t a_u + U_u . grad a_u = 0,
   U_u = zhat x grad_perp Phi + s u (zhat + zhat x grad_perp Psi) = u_E + s u bhat,   div U_u = 0.
Checks:
 SL1 SymPy: {Phi + s u Psi, a} + s u d_z a = U_u . grad a ; div U_u = 0 ; Phi + s u Psi = [(1+su) z+ + (1-su) z-]/2;
     Casimir identities C'(a){A,a} = {A,C(a)}, C'(a) d_z a = d_z C(a); box integrals of {A,B} and d_z C vanish (trig example).
 SL2 NumPy (own pseudo-spectral derivatives, N = 16, M = 24): for arbitrary random Phi, Psi, g, the slice projection of the
     Hermite right-hand side equals the slice right-hand side to round-off, also with a v-independent hyperdiffusion;
     an m-dependent collision operator couples slices (leakage reported).
 SL3 GANDALF: krmhd_rhs (float64) on random states with strong brackets, Lambda = +-sqrt5, beta_i = 1 and 0.5, both closures
     for the zero closure check; slice projection of dg/dt vs -{Phi + s u_j Psi, a_j} - s u_j d_z a_j computed with GANDALF's
     poisson_bracket_3d and derivative_z.
Exit 1 on failure.
"""
import os
import sys
import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cc_common import ladder, P0diag, gauss_rule, SQRT5

fails = []


def check(name, cond, info=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info}", flush=True)
    if not cond:
        fails.append(name)


# --------------------------------------------------------------------------------------------------------- SL1
print("SL1 identities (SymPy)")
x, y, z, s, u = sp.symbols('x y z s u', real=True)
Phi = sp.Function('Phi')(x, y, z)
Psi = sp.Function('Psi')(x, y, z)
a = sp.Function('a')(x, y, z)
pb = lambda f, g: sp.diff(f, x) * sp.diff(g, y) - sp.diff(f, y) * sp.diff(g, x)
grad = lambda f: sp.Matrix([sp.diff(f, x), sp.diff(f, y), sp.diff(f, z)])
zx = lambda f: sp.Matrix([-sp.diff(f, y), sp.diff(f, x), 0])          # zhat x grad_perp f
U = zx(Phi) + s * u * (sp.Matrix([0, 0, 1]) + zx(Psi))
lhs = pb(Phi + s * u * Psi, a) + s * u * sp.diff(a, z)
check("{Phi + s u Psi, a} + s u d_z a = U_u . grad a", sp.simplify(lhs - (U.T * grad(a))[0]) == 0)
check("div U_u = 0", sp.simplify(sp.diff(U[0], x) + sp.diff(U[1], y) + sp.diff(U[2], z)) == 0)
zp, zm = sp.symbols('z_p z_m')
check("Phi + s u Psi = [(1+su) z+ + (1-su) z-]/2 with Phi = (z+ + z-)/2, Psi = (z+ - z-)/2",
      sp.simplify((zp + zm) / 2 + s * u * (zp - zm) / 2 - ((1 + s * u) * zp + (1 - s * u) * zm) / 2) == 0)
C = sp.Function('C')
A_ = sp.Function('A')(x, y, z)
check("C'(a){A,a} = {A, C(a)}", sp.simplify(sp.diff(C(a), a) * pb(A_, a) - pb(A_, C(a))) == 0)
check("{A, a^2/2} = a {A, a} and {A,B} = div(B zhat x grad A)",
      sp.simplify(pb(A_, a**2 / 2) - a * pb(A_, a)) == 0 and
      sp.simplify(pb(A_, a) - (sp.diff(a * (-sp.diff(A_, y)), x) + sp.diff(a * sp.diff(A_, x), y))) == 0)
# periodic example: box integral of {A, C(a)} vanishes for trig polynomials
At = sp.sin(x) * sp.cos(2 * y) + sp.cos(x + z)
at = sp.cos(x) + sp.sin(y - z) / 3
expr = pb(At, at**4) + sp.diff(at**4, z)
val = sp.integrate(sp.integrate(sp.integrate(sp.expand(sp.expand_trig(expr)), (x, 0, 2 * sp.pi)), (y, 0, 2 * sp.pi)), (z, 0, 2 * sp.pi))
check("int_box [{A, a^4} + d_z a^4] = 0 (trig example)", sp.simplify(val) == 0)

# --------------------------------------------------------------------------------------------------------- SL2
print("SL2 slice decomposition of the Hermite right-hand side (own NumPy code)")
N, M = 16, 24
L = 2 * np.pi
k1 = np.fft.fftfreq(N, d=L / (2 * np.pi * N))
KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing='ij')
rng = np.random.default_rng(11)


def rand_field(amp=1.0, kmax=5):
    c = (rng.standard_normal((N, N, N)) + 1j * rng.standard_normal((N, N, N)))
    kk = np.sqrt(KX**2 + KY**2 + KZ**2)
    c *= (kk <= kmax) * (kk > 0) / np.maximum(kk, 1)**2
    return amp * np.real(np.fft.ifftn(c)) * N**3 / 50


d = lambda f, K: np.real(np.fft.ifftn(1j * K * np.fft.fftn(f)))
pbn = lambda f, g: d(f, KX) * d(g, KY) - d(f, KY) * d(g, KX)
lap4 = lambda f: np.real(np.fft.ifftn((KX**2 + KY**2)**2 * np.fft.fftn(f)))
for Lam in (SQRT5, -SQRT5):
    for sv in (1.0, np.sqrt(0.5)):
        A = ladder(M, Lam); p0 = P0diag(M, Lam)
        uj, lam, V = gauss_rule(M, Lam)
        Phi_n, Psi_n = rand_field(150.0), rand_field(150.0)
        g = np.stack([rand_field(1.0) for _ in range(M + 1)], axis=-1)       # g[..., m]
        Ag = g @ A.T
        eta = 0.013
        rhs = (-np.stack([pbn(Phi_n, g[..., m]) for m in range(M + 1)], -1)
               - sv * np.stack([d(Ag[..., m], KZ) for m in range(M + 1)], -1)
               - sv * np.stack([pbn(Psi_n, Ag[..., m]) for m in range(M + 1)], -1)
               - eta * np.stack([lap4(g[..., m]) for m in range(M + 1)], -1))
        Pm = V.T * np.sqrt(p0)[None, :]                     # slice map: a_j = V_j . (P0^{1/2} g)
        a_ = g @ Pm.T
        lhs = rhs @ Pm.T
        srhs = np.stack([-pbn(Phi_n + sv * uj[j] * Psi_n, a_[..., j]) - sv * uj[j] * d(a_[..., j], KZ) - eta * lap4(a_[..., j])
                         for j in range(M + 1)], -1)
        err = np.abs(lhs - srhs).max() / np.abs(srhs).max()
        nl = np.abs(np.stack([pbn(Phi_n, g[..., m]) for m in range(M + 1)], -1)).max()
        st = np.abs(sv * np.stack([d(Ag[..., m], KZ) for m in range(M + 1)], -1)).max()
        check(f"Lambda={Lam:+.3f}, s={sv:.3f}: slice projection of Hermite RHS = slice RHS (with hyperdiffusion)",
              err < 1e-12, f"rel err {err:.1e}, max|{{Phi,g}}|/max|s d_z A g| = {nl / st:.1f}")
        # m-dependent collisions couple slices
        Cm = np.diag(np.where(np.arange(M + 1) >= 2, (np.arange(M + 1) / M)**6, 0.0))
        Cs = Pm @ (Cm @ np.linalg.inv(Pm))                  # collision operator in slice coordinates
        off = np.abs(Cs - np.diag(np.diag(Cs))).max() / np.abs(np.diag(Cs)).max()
        print(f"    hypercollisions (m/M)^6 in slice coordinates: max off-diagonal/max diagonal = {off:.2f}")
check("m-dependent collision operator is not diagonal in slices", off > 0.05)

# --------------------------------------------------------------------------------------------------------- SL3
try:
    import krmhd
    print(f"SL3 GANDALF's right-hand side (krmhd {krmhd.__version__}, float64)")
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from krmhd.physics import KRMHDState, poisson_bracket_3d
    from krmhd.spectral import SpectralGrid3D, derivative_z
    from krmhd.timestepping import krmhd_rhs

    def hermitian_fix(f):
        """f(-k) = conj f(k) on the kx = 0 and Nyquist planes of an rfft array [Nz, Ny, Nkx, ...]."""
        f = f.copy()
        for plane in (0, f.shape[2] - 1):
            sub = f[:, :, plane, ...]
            flipped = np.conj(np.roll(np.roll(sub[::-1, ::-1, ...], 1, axis=0), 1, axis=1))
            f[:, :, plane, ...] = 0.5 * (sub + flipped)
        return f

    for Lam in (SQRT5, -SQRT5):
        for beta in (1.0, 0.5):
            Ng, Mg = 16, 16
            grid = SpectralGrid3D.create(Nx=Ng, Ny=Ng, Nz=Ng)        # 2 pi box: k_min = 1
            mask = np.asarray(grid.dealias_mask)
            kx = np.asarray(grid.kx)[None, None, :]; ky = np.asarray(grid.ky)[None, :, None]; kz = np.asarray(grid.kz)[:, None, None]
            kk = np.sqrt(kx**2 + ky**2 + kz**2); kk[0, 0, 0] = 1.0
            shp = (Ng, Ng, Ng // 2 + 1)
            r2 = np.random.default_rng(5)
            env = 1.0 * Ng**3 * kk**-2.0     # GANDALF stores unnormalised rfftn coefficients: physical amplitude = coef/N^3
            zp = hermitian_fix((r2.standard_normal(shp) + 1j * r2.standard_normal(shp)) * env * mask)
            zm = hermitian_fix((r2.standard_normal(shp) + 1j * r2.standard_normal(shp)) * env * mask)
            zp[0, 0, 0] = 0; zm[0, 0, 0] = 0
            gg = hermitian_fix((r2.standard_normal(shp + (Mg + 1,)) + 1j * r2.standard_normal(shp + (Mg + 1,))) * mask[..., None])
            gg[0, 0, 0, :] = 0
            st = KRMHDState(z_plus=jnp.asarray(zp), z_minus=jnp.asarray(zm), g=jnp.asarray(gg), M=Mg, beta_i=beta, v_th=1.0,
                            nu=0.0, Lambda=float(Lam), time=0.0, grid=grid)
            dg = np.asarray(krmhd_rhs(st, eta=0.0, v_A=1.0, closure="zero").g)
            sv = np.sqrt(beta)
            uj, lam, V = gauss_rule(Mg, Lam)
            Pm = V.T * np.sqrt(P0diag(Mg, Lam))[None, :]
            a_ = gg @ Pm.T
            lhs = dg @ Pm.T
            phi = (zp + zm) / 2; psi = (zp - zm) / 2
            args = (grid.kx, grid.ky, grid.Nz, grid.Ny, grid.Nx, grid.dealias_mask)
            srhs = np.zeros_like(lhs)
            for j in range(Mg + 1):
                aj = jnp.asarray(a_[..., j])
                br = poisson_bracket_3d(jnp.asarray(phi + sv * uj[j] * psi), aj, *args)
                dz = derivative_z(aj, grid.kz)
                rj = np.array((-br - sv * uj[j] * dz) * grid.dealias_mask)
                rj[0, 0, 0] = 0
                srhs[..., j] = rj
            err = np.abs(lhs - srhs).max() / np.abs(srhs).max()
            # strength of the brackets relative to streaming
            nlb = np.abs(np.asarray(poisson_bracket_3d(jnp.asarray(phi), jnp.asarray(gg[..., 3]), *args))).max()
            stm = np.abs(sv * np.asarray(derivative_z(jnp.asarray(gg[..., 3]), grid.kz))).max()
            check(f"GANDALF Lambda={Lam:+.3f}, beta_i={beta}: slices of dg/dt = slice RHS", err < 1e-12,
                  f"rel err {err:.1e}; max|{{Phi,g_3}}|/max|s d_z g_3| = {nlb / stm:.2f}; max|z+-| (physical) = {np.abs(np.fft.irfftn(zp, s=(Ng, Ng, Ng), axes=(0, 1, 2))).max():.2f}")
except Exception as e:
    import traceback
    traceback.print_exc()
    check(f"GANDALF check ran ({e})", False)

if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("ALL s06 CHECKS PASSED")
