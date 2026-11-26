# P1 review rules

Each rule comes from one of Anjor's review comments and applies to the whole paper. Before a draft goes back to Anjor, check every open rule everywhere in the paper. Close a rule only when no instance is left. Comments from 2026-10-14 to 2026-11-16 are taken from the commit messages and the paper's checkpoint, which record them.

## Rules

**R1 (2026-10-14). One message.**
Anjor: "The paper length is too long and I agree is an issue. I am not quite sure what we are talking about."
Every section serves the claim in ARGUMENT.md. Move anything else to the companion paper or to `sections/parked/`. Status (2026-11-26, second referee round): open on length only. The draft is 32 pages, down from 37. The cuts are the referee's group 4: the errata footnotes, appendix A.2–A.3, the copy closure, tab:gates and fig:overview, the GANDALF column of the notation table, the Gaussian-tail asymptotics and the collisional share against M; the η rule moved to appendix B and the band-resolved paragraph and frozen copy were shortened. The budget closure is stated once (section 2.5), T against E_A/ε_A once in the body (section 4.2) and in the abstract, and the 64³-only results are listed once in the body (introduction), in the scorecard table and in the abstract. Every section still serves ARGUMENT.md. The figure items left for integration (fig:flux for every run, a flux-weighted field-aligned spectrum) will add space.

**R2 (2026-10-15). Say what is new, and let a reader follow each step.**
Anjor: "I am really struggling with section 3. I dont quite follow the argument ... what is the result of this paper that is not already covered in Chandran et al / Schekochihin et al?"
Say plainly what is new against Chandran, Mallet & Meyrand (2026) and Schekochihin et al. Each step of an argument must follow from the one before for a reader who was not in the room. Status: closed for this revision, pending integration. The introduction says what S09 lacks and what is new against the other works. Of the referee-2 "can't follow" items, 1.1–1.4, 1.7, 1.8, 1.11–1.13 and 1.15 are resolved in the prose; 1.5, 1.6, 1.9, 1.10 and 1.14 need numbers or figures from the parallel agent and are marked % INTEGRATE in the source.

**R3 (2026-11-13). Results are organised by claim, not reported as facts.**
Anjor, on section 7: "quite dense, a lot of just reporting of facts".
Every number in the results serves a claim that the section states first. Status: closed. Each subsection of §5 opens with its claim. The paragraph of §5.2 that stated its claim last is split and now opens with it, and the collisional-share numbers that served no claim are gone.

**R4 (2026-11-14). A statement about resolution is not a result.**
Anjor: 'beta=1 is not resolved at M=32'. He read it as a run that was too small, not as a physics result.
State resolution checks once, plainly, where they are used. Status: closed. "The break is resolved only at low β_i", "moves the break into the resolved range" and "extending the measurement to β_i = 1 is ongoing work" are gone; §5.4 says that measuring the break needs M large enough that m_* lies below M/2. The collisional share against M and N is gone. The resolution dependence of T is stated in §5.6 and summarised in the introduction and the discussion.

**R5 (2026-11-14). Maths in the style of S09 and Kanekar et al. (2015).**
Anjor: 'The mathy sections are still too mathy'.
Derive in running prose. Use no theorem, lemma or proof environments, and none of the terms that `paper/check_style.py` flags. Status: closed. check_style flags nothing in the files main.tex inputs after the second-round revision, and there are no theorem environments.

**R6 (2026-11-16). Define each term where it is first used, and do not overstate convergence.**
Anjor's comments of that day were on three points, and their wording was not recorded:
- the resolution wording;
- what "the budget closes" means;
- the decay length ℓ_*.

Commit 9732be9 resolved them. Say that the runs are not converged in the perpendicular resolution. Say that whether T becomes independent of the dissipation needs higher resolution. Define each term, for example the budget closing, ℓ_*, κ₁ and m_c, where it first appears. Status: closed except one item left for integration. χ, k⊥,out, k_z,out, u_out, τ_nl, T and c_T are defined in the introduction; the injected pattern in the introduction; c_m in §2.5; κ₁, κ₁(m) and the perpendicular sink in §4.1; ν_m, the four blocks and the block range in §5.1. κ is the field-aligned wavenumber, k_z the Eulerian one and k_z,out the outer-scale wavenumber in χ, and k∥ appears only as S09's k∥0. Open: the "collisional range m > M/2" against hypercollisions that reach 1/T near m = 14 at M = 32 (referee 2 item 2.5, % INTEGRATE). Every statement about T and the dissipation says that it is open.

**R7 (2026-11-26). Separate what holds at every resolution from what holds at 64³.**
Anjor: "Reframe sounds good. Let's get this paper done."
This adopts the three changes of the referee report of 2026-11-26 (`.loop/results/referee-20261126.md`):
- Lead with the results that hold from 64³ to 256³: the frozen-line result, no return flux, and the frozen field-aligned spectrum. Label T, m_*, the β_i and χ trends and the age-dependent sink as 64³ measurements, in the abstract, the introduction, section 5 and the discussion.
- Write the prefactor into m_*: m_* ≈ (c_T²/2) β_i/χ², with the measured c_T. Drop or qualify the equation with Kawazura's m ∼ β_i.
- Say what the test tests. T comes from D⊥/W, and prediction (i) is the model's assumption. The independent checks are the relation between the flux and the spectrum, and the frozen field-aligned spectrum. Use the exact linear flux coefficient.

Status: closed. The abstract, the introduction, the scorecard of §5 and the discussion lead with the 64³–256³ results and label c_T, m_*, the β_i and χ trends and the age sink as 64³. Following the amended ARGUMENT.md, T is given in units of both E_A/ε_A and 1/(k⊥u⊥) as a 64³ value (§4.2), "order 100 rather than 1" is gone, the age sink is described as a property of the 64³ dissipation range, the χ trend carries its caveat and is out of the abstract, and the growth of D⊥/W with the resolution (max/min 1.55 at 64³ to 1.91 at 256³) is a caveat to result 3 and scorecard row (i).

## Referee items for this round

The second referee report (`.loop/results/referee2-20261126.md`) has 15 "can't follow" items, 12 "not supported" items, 5 "not physics" items, 10 cuts and the repeats under R1. The prose revision resolves each one or marks it `% INTEGRATE: <item>` in the source where it needs a number or figure from the parallel figures-and-numbers agent: 1.5, 1.6, 1.9, 1.10, 1.14, 2.2, 2.3, 2.4, 2.5 and the fig:wm(b) point of 2.12. The statements that the s = 0 run was not made also carry a marker, because commit 099b6f0 added its configuration.
