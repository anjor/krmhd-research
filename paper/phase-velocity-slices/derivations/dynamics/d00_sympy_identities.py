"""d00: exact (SymPy) checks of the algebra behind the dynamical results of the paper.  The script verifies that the
RMHD flux equation is the ideal induction equation, the commutator [D_t, grad_par] = {R, .}, Cauchy's solution and
the intertwining grad_par(t) U(t) = U(t) grad_par(0) for an explicit twisting flow, the similarity lemma behind the
parity lemma, the pointwise two-slice identity, the Elsasser forms of the slice advection and the fourth-order
Gaussian algebra.  It supports Section 6 (frozen-line theorem); E4 and E7 also support Section 7 (parity lemma and
fourth-order statement), E5 Section 5 (two-slice law) and E6 Section 7 (slices advected by one Elsasser field).

E1  RMHD flux equation = ideal induction equation.  With B = zhat + zhat x grad Psi and u = zhat x grad Phi
    (B0 = v_A = 1), d_t B - curl(u x B) = zhat x grad R,  R = d_t Psi + {Phi, Psi} - d_z Phi,
    and its z-component vanishes identically; also div u = div B = 0.
E2  Jacobi: [D_t, grad_par] f = {R, f}  (D_t = d_t + {Phi,.}, grad_par = d_z + {Psi,.}).
E3  Cauchy's solution for an explicit 3D twisting flow Phi = Omega(z) r^2/2 (flow map X_perp = Rot(Omega(z) t) xi_perp):
    (a) B(X(xi,t),t) = (dX/dxi) B(xi,0) gives Psi(x,t) = Psi0(Rot(-Omega t) x_perp, z) + t Omega'(z) r^2/2;
    (b) this Psi solves the flux equation for arbitrary Psi0;
    (c) grad_par(t) [h o X_t^{-1}] = [grad_par(0) h] o X_t^{-1} for arbitrary h  (the key commutation);
    (d) a frozen-in shear flow u = S z xhat: k_par = k_z(t) + S t k_x is conserved while k_z(t) = k_z(0) - S t k_x.
E4  Similarity lemma: D = diag((-i)^m); D^{-1} (i A) D = B is real with S B S = -B (S = diag((-1)^m)) for any
    tridiagonal A; hence F(kappa) = exp[-t(i s kappa A + C)] = D R(kappa) D^{-1}, R real, R(-kappa) = S R(kappa) S
    for any real diagonal C.  Parity of the products used by the parity lemma.
E5  Two-slice law, pointwise form: the slice equations imply
    d_t(a_u a_u') = -{Phi, a_u a_u'} - s u' grad_par(a_u a_u') - s (u - u') a_u' grad_par a_u,
    and {Phi, F}, grad_par F are divergences (so they integrate to zero on the torus).
E6  Elsasser forms: Phi + s u Psi = [(1 + su) z+ + (1 - su) z-]/2; GANDALF's z+ nonlinearity
    -(1/2) inv_lap[{z+, lap z-} + {z-, lap z+} - lap {z+, z-}] satisfies
    lap(NL+) = -{z-, lap z+} + {d_x z+, d_x z-} + {d_y z+, d_y z-}, and the last two terms equal
    (Z-.grad) lap z+ - zhat.curl[(Z-.grad) Z+] (Z+- = zhat x grad z+-), the vector-advection correction.
E7  Gaussian moments for the fourth-order statement: for a mirror slice pair with injection amplitudes
    c_{+-u} = v (p +- delta), sum of fourth powers = 2 v^4 (p^4 + 6 p^2 delta^2 + delta^4) (even in delta), and
    E[X^4] = 3 sigma^4 for Gaussian X.
"""
import sys

import sympy as sp

ok_all = True


def check(name: str, expr) -> None:
    global ok_all
    e = sp.simplify(expr)
    ok = e == 0
    ok_all &= ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: residual = {e}")


x, y, z, t = sp.symbols("x y z t", real=True)
Phi = sp.Function("Phi")(x, y, z, t)
Psi = sp.Function("Psi")(x, y, z, t)
f = sp.Function("f")(x, y, z, t)


def br(a, b):
    return sp.diff(a, x) * sp.diff(b, y) - sp.diff(a, y) * sp.diff(b, x)


def zcross_grad(a):
    return sp.Matrix([-sp.diff(a, y), sp.diff(a, x), 0])


def curl(V):
    return sp.Matrix([sp.diff(V[2], y) - sp.diff(V[1], z),
                      sp.diff(V[0], z) - sp.diff(V[2], x),
                      sp.diff(V[1], x) - sp.diff(V[0], y)])


def div(V):
    return sp.diff(V[0], x) + sp.diff(V[1], y) + sp.diff(V[2], z)


print("E1  flux equation <=> ideal induction equation")
u = zcross_grad(Phi)
B = sp.Matrix([0, 0, 1]) + zcross_grad(Psi)
E = sp.diff(B, t) - curl(u.cross(B))
R = sp.diff(Psi, t) + br(Phi, Psi) - sp.diff(Phi, z)
target = zcross_grad(R)
for i, c in enumerate("xyz"):
    check(f"(d_t B - curl(u x B))_{c} - (zhat x grad R)_{c}", sp.expand(E[i] - target[i]))
check("div u", div(u))
check("div B", div(B))
check("u . grad f = {Phi, f}", u.dot(sp.Matrix([sp.diff(f, x), sp.diff(f, y), sp.diff(f, z)])) - br(Phi, f))
check("B . grad f = d_z f + {Psi, f}", B.dot(sp.Matrix([sp.diff(f, x), sp.diff(f, y), sp.diff(f, z)]))
      - (sp.diff(f, z) + br(Psi, f)))

print("E2  Jacobi commutator")
Dt = lambda a: sp.diff(a, t) + br(Phi, a)  # noqa: E731
Dp = lambda a: sp.diff(a, z) + br(Psi, a)  # noqa: E731
check("[D_t, grad_par] f - {R, f}", sp.expand(Dt(Dp(f)) - Dp(Dt(f)) - br(R, f)))

print("E3  Cauchy's solution for a twisting flow Phi = Omega(z) r^2 / 2")
Om = sp.Function("Omega")(z)
xi1, xi2, xi3 = sp.symbols("xi1 xi2 xi3", real=True)
Psi0 = sp.Function("Psi0")
th_xi = sp.Function("Omega")(xi3) * t
X = sp.Matrix([sp.cos(th_xi) * xi1 - sp.sin(th_xi) * xi2, sp.sin(th_xi) * xi1 + sp.cos(th_xi) * xi2, xi3])
PhiT = Om * (x ** 2 + y ** 2) / 2
uT = zcross_grad(PhiT)
# X solves dX/dt = u(X, t)
uX = uT.subs({x: X[0], y: X[1], z: X[2]}, simultaneous=True)
for i in range(3):
    check(f"flow map: dX_{i}/dt - u_{i}(X)", sp.diff(X[i], t) - uX[i])
J = X.jacobian([xi1, xi2, xi3])
B0xi = sp.Matrix([-sp.diff(Psi0(xi1, xi2, xi3), xi2), sp.diff(Psi0(xi1, xi2, xi3), xi1), 1])
B_cauchy = J * B0xi  # B(X(xi,t), t)
th = Om * t
xr = sp.cos(th) * x + sp.sin(th) * y        # Rot(-Omega t) x_perp
yr = -sp.sin(th) * x + sp.cos(th) * y
PsiT = Psi0(xr, yr, z) + t * sp.diff(Om, z) * (x ** 2 + y ** 2) / 2
B_euler = sp.Matrix([0, 0, 1]) + zcross_grad(PsiT)
B_euler_at_X = B_euler.subs({x: X[0], y: X[1], z: X[2]}, simultaneous=True)
for i, c in enumerate("xyz"):
    check(f"(a) Cauchy B(X,t) = J B(xi,0), component {c}", sp.simplify(sp.expand(B_cauchy[i] - B_euler_at_X[i])))
check("(b) Psi(t) solves d_t Psi + {Phi, Psi} - d_z Phi = 0",
      sp.simplify(sp.diff(PsiT, t) + br(PhiT, PsiT) - sp.diff(PhiT, z)))
# (c) with generic Omega(z) and parameterised concrete Psi0, h (generic-function composites defeat simplify)
c1, c2, c3, c4, d1, d2, d3 = sp.symbols("c1:5 d1:4", real=True)


def Psi0c(a, b, c):
    """Polynomial in (a, b) with z-dependent coefficients (keeps the trig only in the rotation)."""
    return c1 * a ** 2 * b + c2 * a * c ** 2 + c3 * b ** 3 * c + c4 * a * b * c


def hc(a, b, c):
    return d1 * a * b * c + d2 * a ** 3 + d3 * b ** 2 * c ** 2


xs, ys, zs = sp.symbols("xs ys zs", real=True)
co, si = sp.symbols("co si", real=True)       # cos(Omega t), sin(Omega t) with co^2 + si^2 = 1
# work with x_r = co x + si y, y_r = -si x + co y and d/dz of (co, si) = Omega' t (-si, co)
PsiTc = Psi0c(xr, yr, z) + t * sp.diff(Om, z) * (x ** 2 + y ** 2) / 2
hU = hc(xr, yr, z)                                                   # U(t) h = h o X_t^{-1}
lhs_c = sp.diff(hU, z) + br(PsiTc, hU)                               # grad_par(t) U(t) h
gp0_h = (sp.diff(hc(xs, ys, zs), zs) + sp.diff(Psi0c(xs, ys, zs), xs) * sp.diff(hc(xs, ys, zs), ys)
         - sp.diff(Psi0c(xs, ys, zs), ys) * sp.diff(hc(xs, ys, zs), xs))
rhs_c = gp0_h.subs({xs: xr, ys: yr, zs: z}, simultaneous=True)      # U(t) grad_par(0) h
res = sp.expand(lhs_c - rhs_c)
res = res.subs({sp.cos(Om * t): co, sp.sin(Om * t): si})
res = sp.expand(res)
res = sp.rem(sp.Poly(res, co, si).as_expr(), co ** 2 + si ** 2 - 1, co) if res != 0 else res
check("(c) grad_par(t) U(t) h - U(t) grad_par(0) h  (modulo cos^2 + sin^2 = 1)", sp.expand(res))
# (d) shear flow u = S z xhat, Phi = -S z y; frozen field b_x = S t; Fourier phase of an advected mode
S_, kx_, kz_ = sp.symbols("S k_x k_z", real=True)
PhiS = -S_ * z * y
PsiS = -S_ * t * y                            # b_perp = zhat x grad Psi = (S t, 0)
check("(d) shear: flux equation", sp.diff(PsiS, t) + br(PhiS, PsiS) - sp.diff(PhiS, z))
mode = sp.exp(sp.I * (kx_ * (x - S_ * z * t) + kz_ * z))   # e^{i k.x} carried by u = S z xhat
check("(d) shear: mode is advected (d_t + {Phi, .}) mode = 0", sp.simplify(sp.diff(mode, t) + br(PhiS, mode)))
kz_t = sp.simplify(sp.diff(sp.log(mode), z) / sp.I)
kpar_t = sp.simplify((sp.diff(mode, z) + br(PsiS, mode)) / (sp.I * mode))
check("(d) shear: Eulerian k_z(t) = k_z - S t k_x", sp.simplify(kz_t - (kz_ - S_ * t * kx_)))
check("(d) shear: k_par(t) = k_z(t) + S t k_x = k_z(0)", sp.simplify(kpar_t - kz_))

print("E4  similarity lemma (symbolic tridiagonal A, M = 5)")
M = 5
aup = sp.symbols(f"a0:{M}", real=True)
adn = sp.symbols(f"b0:{M}", real=True)
A = sp.zeros(M + 1, M + 1)
for m in range(M):
    A[m, m + 1] = aup[m]
    A[m + 1, m] = adn[m]
D = sp.diag(*[(-sp.I) ** m for m in range(M + 1)])
Bm = sp.simplify(D.inv() * sp.I * A * D)
check("D^{-1}(iA)D is real", sum(sp.Abs(sp.im(e)) for e in Bm))
S = sp.diag(*[(-1) ** m for m in range(M + 1)])
check("S B S + B = 0", sum(sp.Abs(e) for e in (S * Bm * S + Bm)))
check("S A S + A = 0", sum(sp.Abs(e) for e in (S * A * S + A)))
# F_{mn} = (-i)^{m-n} R_{mn} with R real: check the prefactor algebra used in the parity lemma for m, n <= 6
for m_ in range(7):
    for n_ in range(7):
        pref = sp.conjugate((-sp.I) ** m_) * (-sp.I) ** (n_ - 1)       # F_{m0}^* F_{n1} / (R_{m0} R_{n1})
        if sp.simplify(pref - sp.I ** (m_ - n_ + 1)) != 0:
            check(f"prefactor m={m_} n={n_}", pref - sp.I ** (m_ - n_ + 1))
check("F_{m0}^* F_{n1} = i^{m-n+1} R_{m0} R_{n1} for m, n <= 6", 0)

print("E5  two-slice law (pointwise identity)")
s, uu, up = sp.symbols("s u u'", real=True)
au = sp.Function("a_u")(x, y, z, t)
aup_ = sp.Function("a_up")(x, y, z, t)
rhs_u = -br(Phi + s * uu * Psi, au) - s * uu * sp.diff(au, z)
rhs_up = -br(Phi + s * up * Psi, aup_) - s * up * sp.diff(aup_, z)
lhs = aup_ * rhs_u + au * rhs_up
F = au * aup_
claim = -br(Phi, F) - s * up * Dp(F) - s * (uu - up) * aup_ * Dp(au)
check("d_t(a_u a_u') identity", sp.expand(lhs - claim))
Fg = sp.Function("F")(x, y, z)
Ph3 = sp.Function("Phi3")(x, y, z)
Ps3 = sp.Function("Psi3")(x, y, z)
check("{Phi, F} = d_y(Phi_x F) - d_x(Phi_y F)", sp.expand(br(Ph3, Fg) - (sp.diff(sp.diff(Ph3, x) * Fg, y)
                                                                         - sp.diff(sp.diff(Ph3, y) * Fg, x))))
check("grad_par F = d_z F + d_y(Psi_x F) - d_x(Psi_y F)", sp.expand(sp.diff(Fg, z) + br(Ps3, Fg)
                                                                     - (sp.diff(Fg, z) + sp.diff(sp.diff(Ps3, x) * Fg, y)
                                                                        - sp.diff(sp.diff(Ps3, y) * Fg, x))))

print("E6  Elsasser forms")
zp = sp.Function("zp")(x, y, z)
zm = sp.Function("zm")(x, y, z)
Phi_e = (zp + zm) / 2
Psi_e = (zp - zm) / 2
check("Phi + s u Psi = [(1+su) z+ + (1-su) z-]/2", sp.expand(Phi_e + s * uu * Psi_e - ((1 + s * uu) * zp + (1 - s * uu) * zm) / 2))
check("su = -1 slice advected by z- only", sp.expand((Phi_e + s * uu * Psi_e).subs(uu, -1 / s) - zm))
check("su = +1 slice advected by z+ only", sp.expand((Phi_e + s * uu * Psi_e).subs(uu, 1 / s) - zp))
lap = lambda a: sp.diff(a, x, 2) + sp.diff(a, y, 2)  # noqa: E731
inner_ = -sp.Rational(1, 2) * (br(zp, lap(zm)) + br(zm, lap(zp)) - lap(br(zp, zm)))
target6 = -br(zm, lap(zp)) + br(sp.diff(zp, x), sp.diff(zm, x)) + br(sp.diff(zp, y), sp.diff(zm, y))
check("lap(NL+) = -{z-, lap z+} + {z+_x, z-_x} + {z+_y, z-_y}", sp.expand(inner_ - target6))
# vector form: Z+- = zhat x grad z+-; the extra term equals (Z-.grad) lap z+ - zhat.curl[(Z-.grad) Z+]
Zp = zcross_grad(zp)
Zm = zcross_grad(zm)
adv = sp.Matrix([sum(Zm[k] * sp.diff(Zp[i], v) for k, v in enumerate((x, y, z))) for i in range(3)])  # (Z-.grad)Z+
curl_adv_z = sp.diff(adv[1], x) - sp.diff(adv[0], y)
adv_curl = sum(Zm[k] * sp.diff(lap(zp), v) for k, v in enumerate((x, y, z)))
extra = br(sp.diff(zp, x), sp.diff(zm, x)) + br(sp.diff(zp, y), sp.diff(zm, y))
check("{z+_x, z-_x} + {z+_y, z-_y} = (Z-.grad) lap z+ - zhat.curl[(Z-.grad) Z+]", sp.expand(extra - (adv_curl - curl_adv_z)))
check("(Z-.grad) f = {z-, f}", sp.expand(sum(Zm[k] * sp.diff(Fg, v) for k, v in enumerate((x, y, z))) - br(zm, Fg)))

print("E7  fourth-order algebra")
v, p, dl, sig = sp.symbols("v p delta sigma", positive=True)
check("c_u^4 + c_-u^4 = 2 v^4 (p^4 + 6 p^2 delta^2 + delta^4)", sp.expand((v * (p + dl)) ** 4 + (v * (p - dl)) ** 4
                                                                         - 2 * v ** 4 * (p ** 4 + 6 * p ** 2 * dl ** 2 + dl ** 4)))
xx = sp.symbols("xx", real=True)
gauss = sp.exp(-xx ** 2 / (2 * sig ** 2)) / sp.sqrt(2 * sp.pi * sig ** 2)
check("E[X^4] = 3 sigma^4", sp.integrate(xx ** 4 * gauss, (xx, -sp.oo, sp.oo)) - 3 * sig ** 4)
gm = sp.symbols("gamma", real=True)
check("(1+g)^2 + (1-g)^2 = 2(1 + g^2)", sp.expand((1 + gm) ** 2 + (1 - gm) ** 2 - 2 * (1 + gm ** 2)))

print("ALL SYMPY CHECKS PASS:", bool(ok_all))
sys.exit(0 if ok_all else 1)
