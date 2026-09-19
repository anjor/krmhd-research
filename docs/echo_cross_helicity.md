# Stochastic echo in imbalanced KRMHD: $m_{\rm cr}(\sigma_c,\beta)$

Working note. Sections marked **[NEW]** go beyond published work; **[LIT]** flags results
taken from the literature; **[GUESS]** flags where I am extrapolating without control.

---

## 0. Convention check — one correction to the brief

With $z^\pm = u_\perp \mp b_\perp$ and $\partial_t u_\perp = v_A\partial_z b_\perp$,
$\partial_t b_\perp = v_A \partial_z u_\perp$:

$$\partial_t z^\pm = \mp v_A \partial_z z^\pm \quad\Longrightarrow\quad
\partial_t z^\pm \pm v_A\partial_z z^\pm + z^\mp\!\cdot\!\nabla_\perp z^\pm = -\nabla_\perp p$$

so $z^+ = u-b$ propagates at $+v_A\hat z$ and is cascaded by $z^-$. Sanity check: an
Alfvén wave travelling along $+B_0$ has $\delta u_\perp = -\delta B_\perp/\sqrt{4\pi\rho}$,
i.e. $u=-b$, hence $z^-=0$. Correct.

Write $\xi \equiv v_\parallel/v_A$, $w_\pm(\xi) \equiv (1\mp\xi)/2$, so

$$U_\perp(\xi) = w_+ z^+ + w_- z^-, \qquad w_++w_-=1.$$

**Consequence, and it inverts the picture in the brief.** $w_+(\xi{=}1)=0$: the particle
riding at $v_\parallel = +v_A$ is advected *purely by $z^-$*, the counter-propagating
field — not by $z^+$. This is not a sign slip, it is the standard statement that a
particle streaming at $v_A$ along a pure $+z$-propagating Alfvén wave suffers no net
perpendicular displacement ($E\times B$ drift exactly cancels $v_\parallel \delta B_\perp/B_0$).
So the field a particle sees *frozen* is precisely the field that does *not* advect it.

Quantitatively, the slip rate of a particle past pattern $z^s$ (which translates rigidly
at $s v_A$) is $\omega^s_{\rm slip} = k_\parallel v_A|\xi - s|$, and since $|\xi-s| = 2|w_s|$,

$$\boxed{\ \omega^s_{\rm slip}(\xi) = 2k_\parallel v_A\,|w_s(\xi)|\ }\tag{0.1}$$

*exactly*. Advection weight and slip rate are the same function of $\xi$. Their ratio,
which is what the echo cares about, is **$\xi$-independent**:

$$\frac{k_\perp |w_s| z^s}{2k_\parallel v_A |w_s|} = \frac{k_\perp z^s}{2k_\parallel v_A}.\tag{0.2}$$

This is the first structural result: *at fixed $\sigma_c$, the coherence of the advecting
flow seen along a streaming characteristic does not depend on $v_\parallel$.* All the
$\beta$-dependence of the echo enters elsewhere (through $\omega_m \propto v_{\rm th}$,
and through the $\xi$-weighting of the *amplitude* $U(\xi)$).

Facts 1 and 3 in the brief survive intact. Fact 2 needs the pairing swapped; the
conclusion it was reaching for (imbalance $\to$ coherent advection) survives anyway, but
for a different reason — see §1.3.

---

## 1. (a) Lagrangian correlation time along the streaming characteristic

### 1.1 Decorrelation channels

Along $z = z_0 + v_\parallel t$ write $z^s(x_\perp,z,t) = Z^s(x_\perp, z - s v_A t; t)$,
where the explicit $t$ carries nonlinear distortion. The particle samples
$Z^s\!\left(x_\perp(t),\, z_0 + (v_\parallel - sv_A)t\right)$. Three channels:

| channel | rate | comment |
|---|---|---|
| parallel slip | $\omega^s_{\rm slip} = 2k_\parallel v_A|w_s|$ | eq. (0.1) |
| intrinsic (Elsasser) evolution | $\omega^s_{\rm nl} = k_\perp z^{-s}$ | $\to 0$ for $s=+$ as $\sigma_c\to1$ |
| perpendicular Lagrangian motion | $k_\perp U$ | **straining, not sweeping** |

The third row is the "sweeping vs straining" question. In the co-moving frame at
$\sigma_c=1$ the flow is a *steady* 2D incompressible field. Sweeping is absent by
construction (there is no mean perpendicular flow). What remains is that the particle
traverses the eddy: the Eulerian velocity along its own trajectory changes on the
turnover time $(k_\perp U)^{-1}$. But a steady 2D incompressible flow is a
one-degree-of-freedom Hamiltonian system: trajectories lie on closed streamlines, so the
Lagrangian velocity is *periodic*, not decorrelating. It contributes an $O(1)$ Lagrangian
Kubo number with an **oscillatory** correlation function, which is not the same thing as
a decaying one. I therefore separate:

- $\mathrm{Ku}_L \equiv k_\perp U\,\tau_L^{\rm osc} = O(1)$ always (trivial, uninformative);
- $\mathcal{C} \equiv$ number of turnover times over which the flow stays *coherent*
  (i.e. before the trajectory stops closing on itself). This is the parameter that
  controls the echo.

Only the first two rows destroy coherence. Hence

$$\tau_s^{-1} = k_\perp z^{-s} + 2k_\parallel v_A |w_s(\xi)|. \tag{1.1}$$

### 1.2 Correlation function

Assuming $\langle z^+ z^-\rangle = 0$ (zero residual energy — see §7),

$$C_L(\tau) = \sum_{s=\pm} w_s^2 E^s e^{-\tau/\tau_s},\qquad
U^2(\xi) = w_+^2E^+ + w_-^2E^-,\qquad
\tau_L = \frac{\sum_s w_s^2E^s\tau_s}{\sum_s w_s^2 E^s}.\tag{1.2}$$

with $E^\pm = (1\pm\sigma_c)E/2$, $z^\pm = \sqrt{E^\pm}$.

### 1.3 Coherence number **[NEW]**

Adopt imbalanced critical balance in the LGS07 form **[LIT: Lithwick, Goldreich &
Sridhar 2007]** — both Elsasser fields share a parallel scale set by the *slower* cascade
rate, that of the dominant field:

$$k_\parallel v_A = \chi\, \omega_{\rm nl}^+ = \chi k_\perp z^-,\qquad \chi = O(1).\tag{1.3}$$

Put $r \equiv z^-/z^+ = \sqrt{(1-\sigma_c)/(1+\sigma_c)}$ and normalise $z^+=1$:

$$\mathcal{C}(\xi,\sigma_c) = k_\perp U\tau_L
= \frac{1}{\sqrt{w_+^2+w_-^2r^2}}\left[\frac{w_+^2}{r\,(1+2\chi|w_+|)}
+ \frac{w_-^2 r^2}{1+2\chi|w_-|r}\right].\tag{1.4}$$

Limits ($\xi\to0$, $\chi=1$):

$$\mathcal{C} \simeq \tfrac14\,\frac{z^+}{z^-} = \tfrac14\sqrt{\frac{1+\sigma_c}{1-\sigma_c}}
\quad(\sigma_c\to1),\qquad \mathcal{C}\simeq 0.35\quad(\sigma_c=0).\tag{1.5}$$

**This is the central point of (a).** As $\sigma_c\to1$ the advection *amplitude* stays
$\sim z^+$ (fact 1 of the brief: no Elsasser suppression for a passive field) while
*every* decorrelation channel scales with $z^-$ — the intrinsic one because that is the
imbalanced cascade rate, and the slip one because critical balance ties $k_\parallel$ to
$z^-$ as well. Hence

$$\boxed{\ \mathcal{C} \propto z^+/z^- = \sqrt{(1+\sigma_c)/(1-\sigma_c)}\ }$$

Imbalance does not weaken the stirring; it makes it *coherent*. Whether that helps or
hurts the echo is question (d), answered in §5: it helps.

---

## 2. (b) Hermite-space critical balance

### 2.1 Two candidate rates — pick the right one

Free streaming $g\propto e^{-ik_\parallel v t}$ has Hermite content peaking at
$m(t) = (k_\parallel v_{\rm th}t)^2/2$, so

$$t(m) = \frac{\sqrt{2m}}{k_\parallel v_{\rm th}},\qquad
\frac{dm}{dt} = k_\parallel v_{\rm th}\sqrt{2m}.$$

Two rates can be built:

- **transfer rate** $\gamma_m = (dm/dt)/m = \sqrt{2}\,k_\parallel v_{\rm th}/\sqrt{m}$ — the
  analogue of a cascade rate;
- **ballistic frequency** $\omega_m = k_\parallel v_{\rm th}\sqrt{2m}$ — the rate at which
  the phase of $\tilde g_m$ winds.

Balancing $\gamma_m \sim \omega_{\rm nl}$ with Alfvénic CB gives $m_{\rm cr}\sim\beta$.
Balancing $\omega_m\sim\omega_{\rm nl}$ gives $m_{\rm cr}\sim\beta^{-1}$. The brief (and
**[LIT: Adkins & Schekochihin 2018]**) wants the latter, and it is the correct criterion
*for the echo* — the echo is a coupling of the $\pm$ Hermite branches, whose frequency
mismatch is $2\omega_m$, not a competition with the transfer rate. Nonlinearity can
reverse the Hermite flux only if it can bridge $2\omega_m$:

$$\boxed{\ \omega_{\rm nl} \sim \omega_{m_{\rm cr}} = k_\parallel v_{\rm th}\sqrt{2m_{\rm cr}}
\quad\Longrightarrow\quad m_{\rm cr} = \tfrac12\left(\frac{\omega_{\rm nl}}{k_\parallel v_{\rm th}}\right)^2\ }\tag{2.1}$$

Balanced check: $\omega_{\rm nl}=k_\perp z$, $k_\parallel v_A = k_\perp z$,
$v_{\rm th}=\sqrt\beta v_A$ $\Rightarrow m_{\rm cr} = 1/(2\beta)$. Recovers
$m_{\rm cr}\sim\beta^{-1}$ **[LIT]**. Good.

### 2.2 Naive imbalanced generalisation

Insert $\omega_{\rm nl} = k_\perp U(\xi)$ and (1.3):

$$m_{\rm cr}^{\rm naive} = \frac{1}{2\chi^2\beta}\,\frac{w_+^2E^+ + w_-^2E^-}{E^-}
\;\xrightarrow[\ \xi\to0\ ]{}\; \frac{1}{4\chi^2\beta\,(1-\sigma_c)}.\tag{2.2}$$

(Algebra: at $\xi=0$, $w_\pm=1/2$, so the ratio is $(1+r^2)/(4r^2)$ with
$r^2=(1-\sigma_c)/(1+\sigma_c)$; $(1+r^2)=2/(1+\sigma_c)$, $4r^2 = 4(1-\sigma_c)/(1+\sigma_c)$,
giving $1/[2(1-\sigma_c)]$, hence the boxed prefactor.)

So the naive answer is a **divergence**, $m_{\rm cr}\propto(1-\sigma_c)^{-1}$: perfect
fluidization at $\sigma_c\to1$. §3 shows this is wrong.

---

## 3. (e) The differential-advection channel is *not* a third channel

At $\beta\sim1$, $w_\pm$ vary by $O(1)$ across the distribution, so particles at different
$v_\parallel$ are advected by materially different flows. Project the offending piece onto
Hermite:

$$\frac{v_\parallel}{v_A}\,b_\perp\!\cdot\!\nabla_\perp g
\;\longrightarrow\;
\frac{v_{\rm th}}{v_A}\,(b_\perp\!\cdot\!\nabla_\perp)
\left[\sqrt{\tfrac{m+1}{2}}\,g_{m+1} + \sqrt{\tfrac{m}{2}}\,g_{m-1}\right],$$

because $v_\parallel H_m \leftrightarrow \sqrt{(m{+}1)/2}\,H_{m+1} + \sqrt{m/2}\,H_{m-1}$.
This has *exactly the same $\sqrt m$ ladder structure* as the streaming term
$v_\parallel\partial_z g \to i k_\parallel v_{\rm th}[\dots]$. Combining,

$$ik_\parallel v_{\rm th}\big[\cdot\big] + \frac{v_{\rm th}}{v_A}(b_\perp\!\cdot\!\nabla_\perp)\big[\cdot\big]
= i v_{\rm th}\,k_\parallel^{\rm eff}\big[\cdot\big],\qquad
k_\parallel^{\rm eff} \equiv k_\parallel - \frac{i}{v_A}\,b_\perp\!\cdot\!\nabla_\perp .\tag{3.1}$$

That is just $\nabla_\parallel = \partial_z + (\delta B_\perp/B_0)\!\cdot\!\nabla_\perp$.

> **Answer to (e): it is not a distinct channel and not a renormalisation of the
> *nonlinearity* either. It is the field-line-bending part of $v_\parallel\nabla_\parallel$
> — a renormalisation of the *first* channel (phase mixing), i.e. streaming along the
> perturbed field line.** Its rate relative to the linear streaming term is
> $$\frac{\omega_{\rm diff}}{\omega_{\rm stream}} = \frac{k_\perp b_\perp}{2k_\parallel v_A}
> \simeq \frac{1}{4\chi}\frac{z^+}{z^-} \sim \mathcal{C},$$
> both $\propto\sqrt m$, so no new $m$-scaling. **[NEW]**

The consequence is decisive: at $\sigma_c\to1$, $k_\parallel\to0$ but
$k_\parallel^{\rm eff}v_A \to k_\perp b_{\rm rms} \simeq k_\perp z^+/2$ stays finite.
**Phase mixing is not switched off by imbalance — it is taken over by field-line
wandering of the dominant field itself.**

With $b_\perp = (z^- - z^+)/2$ and $\langle z^+z^-\rangle=0$,
$b_{\rm rms}^2 = (E^++E^-)/4$, so

$$\big(k_\parallel^{\rm eff}v_A\big)^2 = k_\perp^2\!\left[\chi^2E^- + \tfrac14(E^++E^-)\right].\tag{3.2}$$

---

## 4. (b) $m_{\rm cr}(\sigma_c,\beta)$ **[NEW]**

Substituting (3.2) and $\omega_{\rm nl} = k_\perp U(\xi)$ into (2.1), with
$E^\pm=(1\pm\sigma_c)/2$, $E=1$:

$$m_{\rm cr}(\xi;\sigma_c,\beta)
= \frac{1}{2\beta}\,\frac{w_+^2E^++w_-^2E^-}{\chi^2E^-+\tfrac14(E^++E^-)}
= \frac{1+\xi^2-2\sigma_c\xi}{2\beta\left[1+2\chi^2(1-\sigma_c)\right]}.\tag{4.1}$$

(Numerator algebra: $(1-\xi)^2(1+\sigma_c)+(1+\xi)^2(1-\sigma_c) = 2(1+\xi^2)-4\sigma_c\xi$.)
Note $1+\xi^2-2\sigma_c\xi = (1-\sigma_c\xi)^2 + \xi^2(1-\sigma_c^2) > 0$.

Maxwellian average, $f(\xi)\propto e^{-\xi^2/\beta}$, $\langle\xi^2\rangle=\beta/2$:

$$\boxed{\ \langle m_{\rm cr}\rangle(\sigma_c,\beta)
= \frac{2+\beta}{4\beta\left[1+2\chi^2(1-\sigma_c)\right]}
\ \overset{\chi=1}{=}\ \frac{2+\beta}{4\beta\,(3-2\sigma_c)}\ }\tag{4.2}$$

Properties:

1. $\sigma_c=0$, $\beta\ll1$: $m_{\rm cr}\to1/(6\beta)$. Recovers $\beta^{-1}$ **[LIT]**.
2. $\beta\gg1$: $m_{\rm cr}\to 1/[4(3-2\sigma_c)] = O(1)$. No fluidized range at high
   $\beta$ — free energy phase-mixes straight to the collisional cutoff.
3. **$\sigma_c$-dependence is weak: a factor $3$ from $\sigma_c=0$ to $1$, not a
   divergence.** The naive $(1-\sigma_c)^{-1}$ of (2.2) is cancelled because
   $\omega_{\rm nl}$ *and* $k_\parallel^{\rm eff}v_{\rm th}$ are both set by $z^+$ once
   field-line wandering dominates.
4. **$v_\parallel$-resolved structure.** At $\sigma_c\to1$,
   $m_{\rm cr}(\xi)\to(1-\xi)^2/(2\beta)$ — it **vanishes at $\xi=+1$**. Particles riding
   the dominant wave feel no advection ($w_+=0$), so they have no echo and phase-mix
   freely. At $\beta\lesssim0.1$ these are $e^{-1/\beta}$-rare and irrelevant; at
   $\beta\gtrsim1$ they are a finite fraction of the distribution and dominate the
   residual flux. This is the physically interesting consequence of the
   differential-advection channel, and it is what makes the $(\sigma_c,\beta)$ scan
   non-trivial rather than a one-parameter family.

---

## 5. (c) Echo efficiency $\eta$ and the $W(m)$ slopes

### 5.1 $\eta$ is a reflection coefficient in Hermite space

In the Kanekar et al. variables $\tilde g_{m,k} = (i\,\mathrm{sgn}\,k_\parallel)^m g_{m,k}$
**[LIT: Kanekar, Schekochihin, Dorland & Loureiro 2015]** the WKB branches
$g_m^\pm$ propagate to large / small $m$, and

$$W(m) = |g_m^+|^2+|g_m^-|^2,\qquad
\Gamma_m = \omega_m\big(|g_m^+|^2 - |g_m^-|^2\big),\qquad
\eta(m) \equiv \frac{|g_m^-|^2}{|g_m^+|^2}.$$

So $\Gamma_m^{\rm net} = (1-\eta)\Gamma_m^{(+)}$ with $\eta$ *literally* the reflection
coefficient of the phase-mixing flux off the nonlinear coupling, as required by the brief.

**Steady-state constraint.** With no dissipation below the collisional cutoff,
$\Gamma^{\rm net}_m$ is $m$-independent. Since $\Gamma^{(+)}_m \sim \omega_m W(m)\propto\sqrt m\,W(m)$,

$$\boxed{\ W(m) \;\propto\; \frac{1}{\big[1-\eta(m)\big]\sqrt m}\ }\tag{5.1}$$

This single relation converts any $\eta$ into a spectrum. Check: $\eta=0 \Rightarrow
W\propto m^{-1/2}$, the linear phase-mixing spectrum **[LIT: Zocco & Schekochihin 2011;
Kanekar et al. 2015]**.

### 5.2 Stochastic branch ($\mathcal{C}\lesssim1$) **[LIT-adjacent]**

Born/random-phase reflection off a $\delta$-correlated coupling: transmission per
$m$-octave $\propto(\omega_m/\omega_{\rm nl})^2 = m/m_{\rm cr}$, so

$$1-\eta = \frac{m/m_{\rm cr}}{1+m/m_{\rm cr}}
\quad\Longrightarrow\quad
W\propto m^{-3/2}\ (m\ll m_{\rm cr}),\qquad W\propto m^{-1/2}\ (m\gg m_{\rm cr}).$$

The $m^{-3/2}$ below $m_{\rm cr}$ and $m^{-1/2}$ above is the known picture
**[LIT: Schekochihin et al. 2016; Adkins & Schekochihin 2018; Meyrand et al. 2019]**.
Note it drops out of (5.1) *automatically* once $1-\eta\propto m$ — I regard the
consistency as evidence the criterion (2.1) is the right one.

### 5.3 Coherent branch ($\mathcal{C}\gg1$) **[NEW, GUESS on the exponent]**

If the coupling is coherent over many turnovers, reflection below $m_{\rm cr}$ is
WKB-barrier-like, transmission exponentially small:

$$1-\eta \simeq \exp\!\big[-\mathcal{C}\,(1-m/m_{\rm cr})\big],\quad m<m_{\rm cr};
\qquad 1-\eta\to1,\quad m>m_{\rm cr}.$$

Then (5.1) gives $W(m)\propto m^{-1/2}e^{+\mathcal{C}(1-m/m_{\rm cr})}$: **an exponential
cutoff in Hermite space with $e$-folding width $\Delta m = m_{\rm cr}/\mathcal{C}$**, not a
power law. Since $m_{\rm cr}$ is $\sigma_c$-independent up to the factor $3$ while
$\mathcal{C}\propto[(1+\sigma_c)/(1-\sigma_c)]^{1/2}$, we predict
$\Delta m \propto \beta^{-1}\sqrt{1-\sigma_c}$: **the fluidized range narrows and steepens
as imbalance grows, even though $m_{\rm cr}$ barely moves.**

### 5.4 Interpolation and local slope **[ANSATZ]**

$$\boxed{\ 1-\eta(m;\sigma_c,\beta) = \frac{1}{1+m_{\rm cr}/m}\;
\exp\!\big[-\mathcal{C}\,(1-m/m_{\rm cr})_+\big]\ }\tag{5.2}$$

$$\frac{d\ln W}{d\ln m} = -\frac12 - \frac{1}{1+m/m_{\rm cr}}
- \mathcal{C}\,\frac{m}{m_{\rm cr}}\Theta(m_{\rm cr}-m).\tag{5.3}$$

Limits: $-3/2$ at $m\ll m_{\rm cr}$; $-(1+\mathcal{C})$ at $m\to m_{\rm cr}^-$; $-1/2$ for
$m>m_{\rm cr}$. The high-$m$ slope is $-1/2$ **for every $(\sigma_c,\beta)$** — that is a
rigid prediction, see §8.

---

## 6. (d) Does the echo survive *coherent* advection?

**Yes, and it is more efficient. Sign: positive.** Argument in four steps.

1. **The echo does not require randomness.** The Gould–O'Neil–Malmberg echo is
   deterministic: two impulses at $k_1$ and $k_2$ produce ballistic phase cancellation at
   $k_2-k_1$. In KRMHD the "impulses" are supplied by an advecting field with parallel
   wavenumber $q_\parallel\neq0$, which scatters $g$ from $k_\parallel$ to
   $k_\parallel - q_\parallel$; when the sign flips, phase mixing reverses. Nothing in this
   requires $\partial_t$ of the advecting field. Stochastic echo
   **[LIT: Schekochihin et al. 2016]** is the *statistical* limit of this, not the
   mechanism itself.

2. **A frozen, $v_\parallel$-independent flow gives $\eta=0$, not $\eta=1$** — worth stating
   because it sharpens what the mechanism actually needs. If
   $U_\perp = U_\perp(x_\perp)$ only, then $g(t) = g_0(X_\perp(-t), z-v_\parallel t, v_\parallel)$:
   the perpendicular advection factorises out and commutes with streaming, so it cannot
   touch the Hermite hierarchy at all. **The echo needs $\partial_z U_\perp \neq 0$** (so
   that streaming makes the Lagrangian map $v_\parallel$-dependent) — it does *not* need
   $\partial_t U_\perp \neq 0$. Critical balance guarantees $q_\parallel\sim k_\parallel$,
   so the requirement is met at any $\sigma_c$.

3. **Exact statement at $\sigma_c=1$.** With $z^-=0$, $z^+$ is an exact RMHD solution
   translating rigidly at $v_A$. In the co-moving frame,
   $$\partial_t g + w_+(v_\parallel)\,Z^+(x_\perp,z')\!\cdot\!\nabla_\perp g
   + (v_\parallel-v_A)\partial_{z'}g = 0$$
   is **linear, autonomous, and anti-Hermitian** in the free-energy norm. There is no
   intrinsic irreversibility: any transfer to high $m$ is a unitary rearrangement, and
   the net steady-state flux is fixed entirely by forcing/dissipation, not by the
   dynamics. Coherent advection therefore cannot *manufacture* forward Hermite flux.

4. **Why more efficient.** Stochastic cancellation is partial by construction (it is a
   leading-order cancellation with a random-phase residual, $1-\eta\propto m/m_{\rm cr}$).
   Coherent reflection off a static barrier is exponentially complete
   ($1-\eta\propto e^{-\mathcal{C}}$). Hence §5.3.

**The one thing I cannot settle analytically.** Which $k_\parallel$ enters the *slip* rate
(0.1): the lab-frame $k_\parallel$ of $z^+$, or the field-line-following
$k_\parallel^{\rm eff}$? The pattern $z^+$ translates rigidly in lab $z$, which argues for
lab $k_\parallel \sim \chi k_\perp z^-/v_A$ and hence $\mathcal{C}\to\infty$ (**Branch A**).
But the particle streams along the *perturbed* field line, whose wandering is generated by
$z^+$ itself; if that is what sets the rate at which it samples new $Z^+$, then
$\mathcal{C}\to k_\perp z^+/(2k_\perp b_{\rm rms}) = O(1)$ for all $\sigma_c$
(**Branch B**), and the echo never becomes coherent.

- Branch A: $W(m)$ below $m_{\rm cr}$ steepens without bound as $\sigma_c\to1$
  (exponential cutoff).
- Branch B: $W(m)\propto m^{-3/2}$ below $m_{\rm cr}$ at *all* $\sigma_c$; only $m_{\rm cr}$
  moves, by the factor $(3-2\sigma_c)^{-1}$.

**What would settle it:** directly measure, in the simulation, the correlation function of
$U_\perp$ sampled along streaming characteristics $z=z_0+v_\parallel t$ at several
$v_\parallel$, and extract $\tau_L$; then form $\mathcal{C}=k_\perp U\tau_L$ and check
whether it grows as $z^+/z^-$ or saturates at $O(1)$. This is a cheap diagnostic — it
needs only the Alfvénic fields, no kinetics. **I would run it first, before the full
$(\sigma_c,\beta)$ scan.**

---

## 7. Falsifiable predictions for a $(\sigma_c,\beta)$ scan

All from (4.2), (5.2), (5.3) with $\chi=1$. The **only** free parameter is one overall
constant in $m_{\rm cr}$, fixed here by declaring $m_{\rm cr}(\sigma_c{=}0,\beta{=}1)=5$;
every *ratio* in the table is parameter-free.

Diagnostics assumed measurable: $W(m)$ at fixed $k_\perp$ in the inertial range;
$\Gamma_m^{\rm net}$ from $\langle g_m g_{m\pm1}\rangle$; the split
$|g^\pm_m|^2$ via the Kanekar transform. Echo fraction quoted is
$\eta = 1-\Gamma^{\rm net}/\Gamma^{(+)}$ evaluated at the fixed moment
$m_{\rm ref}(\beta) = \tfrac12 m_{\rm cr}(0,\beta)$, i.e. **the same $m$ across a $\beta$
row**, so the $\sigma_c$-trend in each row is directly measurable without recalibration.

| $\beta$ | $\sigma_c$ | $m_{\rm cr}$ | $m_{\rm cr}/m_{\rm cr}(\sigma_c{=}0)$ | $\mathcal{C}$ | $\eta$ (A: coherent) | $\eta$ (B: stochastic) | slope at $m_{\rm ref}$ (A / B) | slope $m\!\gg\!m_{\rm cr}$ |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 0    | 35  | 1.00 | 1.0  | 0.80 | 0.80 | −1.67 / −1.67 | −1/2 |
| 0.1 | 0.3  | 44  | 1.25 | 1.4  | 0.87 | 0.84 | −1.76 / −1.61 | −1/2 |
| 0.1 | 0.6  | 58  | 1.67 | 2.0  | 0.94 | 0.88 | −1.87 / −1.57 | −1/2 |
| 0.1 | 0.9  | 88  | 2.50 | 4.4  | 0.99 | 0.91 | −2.21 / −1.53 | −1/2 |
| 0.1 | 0.99 | 103 | 2.94 | 14.1 | 0.997| 0.92 | −3.75 / −1.52 | −1/2 |
| 1   | 0    | 5.0 | 1.00 | 1.0  | 0.76 | 0.76 | −1.67 / −1.67 | −1/2 |
| 1   | 0.3  | 6.3 | 1.25 | 1.4  | 0.84 | 0.81 | −1.76 / −1.61 | −1/2 |
| 1   | 0.6  | 8.3 | 1.67 | 2.0  | **0.89** | **0.83** | −1.87 / −1.57 | −1/2 |
| 1   | 0.9  | 13  | 2.50 | 4.4  | 0.87 | 0.80 | −2.21 / −1.53 | −1/2 |
| 1   | 0.99 | 15  | 2.94 | 14.1 | 0.83 | 0.76 | −3.75 / −1.52 | −1/2 |
| 10  | 0    | 2.0 | 1.00 | 1.0  | 0.58 | 0.58 | −1.67 / −1.67 | −1/2 |
| 10  | 0.3  | 2.5 | 1.25 | 1.4  | 0.64 | 0.63 | −1.76 / −1.61 | −1/2 |
| 10  | 0.6  | 3.3 | 1.67 | 2.0  | 0.70 | 0.67 | −1.87 / −1.57 | −1/2 |
| 10  | 0.9  | 5.0 | 2.50 | 4.4  | 0.74 | 0.68 | −2.21 / −1.53 | −1/2 |
| 10  | 0.99 | 5.9 | 2.94 | 14.1 | 0.75 | 0.68 | −3.75 / −1.52 | −1/2 |

Bold marks the **non-monotonicity**: at $\beta=1$ the echo fraction peaks near
$\sigma_c\approx0.6$ and *falls* again at $\sigma_c\to1$. This is the
$\xi\to+1$ leak of §4.4 — particles with $v_\parallel\approx+v_A$ have
$m_{\rm cr}\propto(1-\xi)^2\to0$ and phase-mix unimpeded. At $\beta=0.1$ those particles
are $e^{-1/\beta}$-rare and the trend is monotonic; at $\beta=10$ the whole distribution is
far from the resonance in relative terms and the trend is again monotonic. **A peak in
$\eta(\sigma_c)$ at $\beta\sim1$ and nowhere else is the sharpest signature of the
theory.**

Additional predictions, in decreasing order of confidence:

- **P1.** $m_{\rm cr}\beta \to$ const as $\beta\to0$ at fixed $\sigma_c$; $m_{\rm cr}\to O(1)$
  at $\beta\gg1$. Combined: $m_{\rm cr}\propto(2+\beta)/\beta$.
- **P2.** $m_{\rm cr}(\sigma_c)/m_{\rm cr}(0) = 3/(3-2\sigma_c)$, **saturating at 3**, the
  same function of $\sigma_c$ at every $\beta$ (the $(\sigma_c,\beta)$ dependence
  factorises).
- **P3.** $W(m)\propto m^{-1/2}$ above $m_{\rm cr}$ at all $(\sigma_c,\beta)$.
- **P4.** $v_\parallel$-resolved residual free-energy flux at $\beta\gtrsim1$,
  $\sigma_c\gtrsim0.6$ is peaked at $v_\parallel\simeq+v_A$ (i.e. at the *co-propagating*
  resonance with the dominant Elsasser field), with width $\Delta v_\parallel/v_A\sim\sqrt{2\beta m}$.
- **P5.** Branch discriminator: below $m_{\rm cr}$, $W(m)$ either stays at $-3/2$ (B) or
  steepens towards $-(1+\mathcal{C})$ (A).

---

## 8. What kills the theory

In order of lethality:

1. **$m_{\rm cr}$ diverging as $(1-\sigma_c)^{-1}$.** Eq. (4.2) says the $\sigma_c$-enhancement
   *saturates at a factor 3* because $k_\parallel^{\rm eff}$ is taken over by field-line
   bending (§3). If a scan at $\beta=0.1$ finds $m_{\rm cr}$ growing by an order of
   magnitude between $\sigma_c=0$ and $0.9$ — i.e. tracking the naive (2.2) — then the
   identification $\omega_{\rm diff}=$ the $\nabla_\parallel$ nonlinearity is wrong, and with
   it the whole of §3–§4. **This is the single decisive measurement.**

2. **$m_{\rm cr}$ *decreasing* with $\sigma_c$ at $\beta=0.1$.** That would mean imbalance
   destroys the echo outright, falsifying §6 (the claim that coherence helps) at the root.
   Everything downstream fails.

3. **High-$m$ slope $\neq-1/2$, or $\sigma_c$-dependent.** (5.1) plus constant net flux
   forces $-1/2$ wherever $\eta\to0$. A $\sigma_c$-dependent high-$m$ slope would mean the
   net Hermite flux is not $m$-independent, i.e. there is hidden $m$-space dissipation or
   injection — the steady-state constraint underpinning all of §5 would be void.

4. **No peak in $\eta(\sigma_c)$ at $\beta\sim1$, and no $v_\parallel\simeq+v_A$
   localisation of the residual flux (P4).** Kills the differential-advection resonance,
   i.e. §4.4 and the interesting half of (e), while leaving §1–§3 standing.

Non-lethal: getting the *shape* of $\eta(m)$ wrong (5.2 is an ansatz), or Branch A vs B —
that is a discrimination, not a refutation.

---

## 9. Assumptions and weak points

**Assumptions used, in the order they bite:**

- **A1.** $\langle z^+z^-\rangle=0$ (zero residual energy). Used in (1.2) and in
  $b_{\rm rms}^2=(E^++E^-)/4$. Real imbalanced RMHD has finite residual energy, magnetically
  dominated; this would *increase* $b_{\rm rms}$ and hence $k_\parallel^{\rm eff}$, pushing
  $m_{\rm cr}$ *down*. Direction of error is known, magnitude is not.
- **A2.** LGS07 imbalanced critical balance, $k_\parallel v_A = \chi k_\perp z^-$, single
  parallel scale for both fields. The competing picture (Chandran 2008; Beresnyak &
  Lazarian 2008) gives different $k_\parallel^\pm$. This only affects the $\chi^2E^-$ term
  in (3.2), which is subdominant precisely where the interesting physics is
  ($\sigma_c\to1$), so the conclusions are more robust than the assumption.
- **A3.** Compressive $k_\parallel$ inherited from the Alfvénic field. Standard for a
  passive field **[LIT: Schekochihin et al. 2009]**, but $g$ also has $k_\parallel$ generated
  by its own forcing.
- **A4.** Criterion (2.1) — $\omega_{\rm nl}\sim\omega_m$ rather than
  $\omega_{\rm nl}\sim\gamma_m$. Justified by the branch-coupling argument and by the fact
  that it reproduces both $m_{\rm cr}\sim\beta^{-1}$ and the $m^{-3/2}$/$m^{-1/2}$ pair, but
  it is a scaling argument, not a calculation.
- **A5.** $m\gg1$ WKB throughout. At $\beta\gtrsim1$, (4.2) gives $m_{\rm cr}=O(1)$ and the
  whole framework is being used outside its domain. **Treat the $\beta=10$ row as
  qualitative only.**
- **A6.** Single $k_\perp$. Everything is evaluated at one perpendicular scale; the
  $(k_\perp,m)$ phase-space spectrum of Adkins & Schekochihin is collapsed to a
  one-dimensional problem. Cross-scale Hermite transfer is ignored.
- **A7.** Steady state, no collisions below $m_{\rm cr}$, no forcing in $m$.

**Weak points I am least happy with:**

- The functional form (5.2) is an ansatz stitching two limits. The exponents ($1$ in the
  Born term, $\mathcal{C}$ in the WKB term) are dimensional guesses. The *locations*
  ($m_{\rm cr}$) and the *limiting slopes* ($-3/2$, $-1/2$) are on firmer ground than the
  interpolation between them.
- Branch A vs B (§6) is unresolved and it is not cosmetic: it decides whether the
  sub-$m_{\rm cr}$ spectrum is a power law or an exponential. §6's diagnostic settles it
  cheaply.
- The claim in §1.1 that a steady 2D flow gives an *oscillatory* rather than decaying
  Lagrangian correlation is exact for strictly 2D steady flow, but $Z^+(x_\perp,z')$ is
  three-dimensional and the particle drifts in $z'$; I have modelled that drift as a
  simple exponential decorrelation at rate $k_\parallel v_A|1-\xi|$, which is the crudest
  step in §1.
- $\sigma_c$ is held fixed and scale-independent. In real imbalanced turbulence
  $\sigma_c(k_\perp)$ drifts through the inertial range, so $m_{\rm cr}$ and $\mathcal{C}$
  are functions of $k_\perp$ even at fixed global imbalance. In a forced simulation with
  imposed $\sigma_c$ this is controllable; in nature it is not.
- I have not checked whether the $\xi\to+1$ resonance (§4.4, P4) survives the Hermite
  representation. It is a statement about $v_\parallel$-space localisation, and Hermite
  moments are non-local in $v_\parallel$; the observable consequence may be weaker than the
  physical effect.

---

## References

- Adkins & Schekochihin 2018, *JPP* **84**, 905840107 — solvable Fourier–Hermite model.
- Schekochihin, Parker, Highcock, Dellar, Dorland & Hammett 2016, *JPP* **82**, 905820212 —
  stochastic echo, phase-mixing/anti-phase-mixing critical balance.
- Kanekar, Schekochihin, Dorland & Loureiro 2015, *JPP* **81**, 305810104 — Hermite
  Elsasser variables $\tilde g_m = (i\,\mathrm{sgn}k_\parallel)^m g_m$.
- Zocco & Schekochihin 2011, *PoP* **18**, 102309 — $W(m)\propto m^{-1/2}$.
- Meyrand, Kanekar, Dorland & Schekochihin 2019, *PNAS* **116**, 1185 — fluidization.
- Lithwick, Goldreich & Sridhar 2007, *ApJ* **655**, 269 — imbalanced critical balance.
- Schekochihin et al. 2009, *ApJS* **182**, 310 — KRMHD, passive compressive cascade.
- Gould, O'Neil & Malmberg 1967, *PRL* **19**, 219 — plasma echo.
