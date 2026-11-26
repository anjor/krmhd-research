# P1 argument

"Phase mixing against a passive-scalar cascade: the Hermite spectrum of kinetic reduced magnetohydrodynamic turbulence". This is a draft for Anjor's sign-off, 2026-11-26. It is written for the reframe he agreed that day. Numbers marked † are being rechecked and will be updated before the prose is.

## Claim

While the magnetic field lines stay frozen into the E×B flow, the KRMHD nonlinearity moves no free energy between Hermite moments or between field-aligned wavenumbers. In turbulence the Hermite spectrum is therefore linear phase mixing at the injected field-aligned wavenumber, cut off by a perpendicular cascade that removes free energy from every moment alike. In forced turbulence from 64³ to 256³ grid points, the Hermite flux is the phase-mixing flux of W(m) at the measured field-aligned wavenumber, with no return flux. Over the same range, the field-aligned spectrum of the free energy stays at the injected wavenumbers.

## The argument

The streaming speed is s = v_th/v_A = β_i^{1/2}. Free energy injected at field-aligned wavenumber κ phase mixes to moment m in a time t_m ~ √(2m)/(sκ). The perpendicular cascade removes it in a time T that the Alfvénic turbulence sets.

While the lines are frozen, the cascade only moves free energy across the field. It cannot change κ, so it cannot carry free energy back to low m. The flux in m is then Π(m) ≈ sκ√(2m) W(m), and the flux decreases with m as the sink removes free energy, dΠ/dm = −W/T. This gives

  W(m) ∝ m^{-1/2} exp[−√(2m)/(sκT)],  with a break where t_m ~ T, m_* = (sκT)²/2.

Write T = c_T τ_nl, with τ_nl ~ 1/(χ κ v_A) for κ of the order of the outer-scale k∥. Then m_* ≈ (c_T²/2) β_i/χ². The prefactor matters. In our runs c_T is about 11 to 16†, so at χ ≈ 1 the break m_* is 60 to 130† times β_i, not of order β_i.

## What the runs show

These hold at every resolution tested, 64³ to 256³:
1. The field-aligned Hermite flux equals the phase-mixing flux of W(m) at the measured field-aligned wavenumber. The comparison uses the exact linear coefficient. No deficit grows with m, so there is no return flux.
2. The field-aligned spectrum of the free energy stays at the injected wavenumbers, apart from a tail holding less than 0.5%† of it.
3. With the T measured at each resolution, the model describes W(m).

These are measured at 64³ only, and the paper says so:
4. D⊥(m)/W(m) is independent of m, to within 20% at β_i = 1. T is its median, so this checks the model's assumption. It is not an independent prediction.
5. The value of c_T, and with it m_*.
6. m_* ∝ β_i^{1.09} for 0.05 ≤ β_i ≤ 0.3, at χ ≈ 1.
7. The sink grows with the age of the free energy, as the same function of age at every β_i.
8. At β_i = 0.05, m_* grows faster than χ⁻² as χ falls.

One thing is open. T has not converged. It grows by about the same amount with each doubling of the resolution up to 256³, so whether the break becomes independent of the dissipation is not settled.

## Key figure

The key figure is fig:flux (h1_flux). It compares the field-aligned flux with the phase-mixing flux of W(m) at the measured κ, using the exact coefficient, and it shows result 1. fig:kpar (h1_kpar_spectra) shows result 2.

## What is new

- **S09 §6.3** compared Landau damping at a frozen parallel wavenumber with the perpendicular cascade. We make that comparison exact. With ideal induction the nonlinearity moves no free energy in m or in field-aligned wavenumber, for every volume-integrated single-point statistic.
- **Zocco & Schekochihin (2011), Schekochihin et al. (2016) and Kawazura et al. (2019)** have the functional form and the break scale m ~ β_i. We test the form in turbulence, with diagnostics computed along the field, and we measure the prefactor. At 64³ it is of order 100, not 1.
- **Meyrand et al. (2019)** attributed a steep spectrum and a small net flux to stochastic echoes. They argued that the compressive fluctuations inherit the parallel cascade. Along the field lines we find no return flux and no parallel cascade of the free energy at 64³ to 256³.
- **Chandran, Mallet & Meyrand (2026)** give the form of KRMHD that we use. Their subject is the gyrokinetic helicity and its conversion near the ion gyroscale, and they do not treat the Hermite spectrum. The argument here does not use their invariants. The companion paper discusses them.

## Not in this paper

- The slice decomposition, which is in the companion paper.
- Any statement that T or m_* is independent of the dissipation.
- A χ⁻⁴ law. It rests on two points at one resolution, with η changed between the runs.
- The fall of the collisional share with M as evidence about a physical collisional cutoff. M = 32 and 64 is a property of the runs.
