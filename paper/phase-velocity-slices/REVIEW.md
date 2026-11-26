# P1 review rules

Each rule comes from one of Anjor's review comments and applies to the whole paper. Before a draft goes back to Anjor, check every open rule everywhere in the paper. Close a rule only when no instance is left. Comments from 2026-10-14 to 2026-11-16 are taken from the commit messages and the paper's checkpoint, which record them.

## Rules

**R1 (2026-10-14). One message.**
Anjor: "The paper length is too long and I agree is an issue. I am not quite sure what we are talking about."
Every section serves the claim in ARGUMENT.md. Move anything else to the companion paper or to `sections/parked/`. Status (2026-11-26 revision): open on length only. Every section now serves ARGUMENT.md, and the χ⁻⁴ law, the collisional-cutoff argument, the unsupported checks of §3.6, the long χ paragraph of §5.5 and a duplicate k_z-split passage are gone. The text is 16.9k words against 17.0k before the round and still 37 pages, because the definitions and block ranges the referee asked for used the space. Candidates for a further cut: tab:gates (it repeats a row of tab:checks), fig:overview, the band-resolved paragraph of §5.3 and the Gaussian-tail asymptotics in §4.1.

**R2 (2026-10-15). Say what is new, and let a reader follow each step.**
Anjor: "I am really struggling with section 3. I dont quite follow the argument ... what is the result of this paper that is not already covered in Chandran et al / Schekochihin et al?"
Say plainly what is new against Chandran, Mallet & Meyrand (2026) and Schekochihin et al. Each step of an argument must follow from the one before for a reader who was not in the room. Status: closed for this revision. The introduction says what is new against S09, Zocco & Schekochihin, Schekochihin et al. (2016), Kawazura et al., Meyrand et al. and Chandran, Mallet & Meyrand, and the referee's 19 "can't follow" items are resolved except the χ⁻⁴ guide in fig:mstar(b), which needs a figure change. Recheck with the next cold read.

**R3 (2026-11-13). Results are organised by claim, not reported as facts.**
Anjor, on section 7: "quite dense, a lot of just reporting of facts".
Every number in the results serves a claim that the section states first. Status: closed. Each subsection of §5 opens with its claim, and the scorecard says at which grid each result holds.

**R4 (2026-11-14). A statement about resolution is not a result.**
Anjor: 'beta=1 is not resolved at M=32'. He read it as a run that was too small, not as a physics result.
State resolution checks once, plainly, where they are used. Status: closed. "Marginally resolved" is gone, the M = 32 against 64 comparison is stated as a property of the truncation, and the resolution dependence of T is stated in §5.6 and summarised once in the introduction and the discussion.

**R5 (2026-11-14). Maths in the style of S09 and Kanekar et al. (2015).**
Anjor: 'The mathy sections are still too mathy'.
Derive in running prose. Use no theorem, lemma or proof environments, and none of the terms that `paper/check_style.py` flags. Status: closed. check_style flags nothing in the files main.tex inputs after the 2026-11-26 revision, and there are no theorem environments.

**R6 (2026-11-16). Define each term where it is first used, and do not overstate convergence.**
Anjor's comments of that day were on three points, and their wording was not recorded:
- the resolution wording;
- what "the budget closes" means;
- the decay length ℓ_*.

Commit 9732be9 resolved them. Say that the runs are not converged in the perpendicular resolution. Say that whether T becomes independent of the dissipation needs higher resolution. Define each term, for example the budget closing, ℓ_*, κ₁ and m_c, where it first appears. Status: closed. κ, T, τ_nl and c_T are defined in the introduction before the model equation; c_m in §2.5; κ₁, κ₁(m) and the perpendicular sink in §4.1; ν_m and the four blocks in §5.1. m_c is no longer used. Every statement about T and the dissipation says that it is open.

**R7 (2026-11-26). Separate what holds at every resolution from what holds at 64³.**
Anjor: "Reframe sounds good. Let's get this paper done."
This adopts the three changes of the referee report of 2026-11-26 (`.loop/results/referee-20261126.md`):
- Lead with the results that hold from 64³ to 256³: the frozen-line result, no return flux, and the frozen field-aligned spectrum. Label T, m_*, the β_i and χ trends and the age-dependent sink as 64³ measurements, in the abstract, the introduction, section 5 and the discussion.
- Write the prefactor into m_*: m_* ≈ (c_T²/2) β_i/χ², with the measured c_T. Drop or qualify the equation with Kawazura's m ∼ β_i.
- Say what the test tests. T comes from D⊥/W, and prediction (i) is the model's assumption. The independent checks are the relation between the flux and the spectrum, and the frozen field-aligned spectrum. Use the exact linear flux coefficient.

Status: closed. The abstract, the introduction, the scorecard of §5 and the discussion lead with the 64³–256³ results and label T, c_T, m_*, the β_i and χ trends and the age sink as 64³. Equation (34) is m_* ≈ (c_T²/2) β_i/χ² with c_T = 11–16, and Kawazura's m ∼ β_i is the c_T ∼ 1 case. §4.3 "What the simulations test" says that (i) checks the assumption and that (ii) and (iv) are the independent tests. The flux comparison uses s c_m κ₁(m) W(m).

## Referee items for this round

The referee report lists 19 "can't follow" items, 17 "not supported" items and 6 "not physics" items. The revision resolves each one or states in the checkpoint why it was left. The items under R7 come first.
