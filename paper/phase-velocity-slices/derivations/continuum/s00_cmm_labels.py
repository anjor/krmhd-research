"""s00: equation numbers of Chandran, Mallet & Meyrand (2026, arXiv:2607.27981v1; CMM below) read from their LaTeX
source. The script recomputes the number of every CMM equation label cited in the paper and in these derivations from
the numbering rules of the journal class file shipped with the source, checks it against the number quoted in the
paper (a label fails if its number differs), and checks a few printed formulas (Lambda+-, sigma, G+-, kappa as
printed in (B16), tau, F_M, v_th) by string matching. It supports the CMM equation numbers quoted in Sections 2, 3, 4
and 7 and Appendix A of the paper.

The source is downloaded data and is read as text only; run the script in isolated mode, from the repository root:

    uv run python -I paper/phase-velocity-slices/derivations/continuum/s00_cmm_labels.py [SRC_DIR]

SRC_DIR defaults to paper/phase-velocity-slices/data/cmm (git-ignored). If it does not hold the source, the script runs
../fetch_cmm.py as a separate process to download the arXiv e-print into it; if the source still cannot be obtained, it
prints SKIPPED and exits 0. Self-contained: no local imports.

Numbering rule taken from the paper's own class file jpp.cls (read as text):
  \\section resets the equation counter; main text numbers are (section.n); after \\appendix the section
  counter is lettered and equations are (B n).  Numbered environments: equation (one number unless
  \\nonumber/\\notag) and align (one number per line separated by \\\\, minus lines with \\nonumber).
  \\[ ... \\] and starred environments are unnumbered.
Prints the number of every label with PASS/FAIL and checks a few formulas by string matching.
Exit code 1 if a label is missing or a formula string is not found.
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SRC = HERE.parents[1] / "data" / "cmm"          # paper/phase-velocity-slices/data/cmm
FETCH = HERE.parent / "fetch_cmm.py"
TEX_NAME = "Gyrokinetic_helicity.tex"                   # top-level file of the e-print
CLS_NAME = "jpp.cls"


def locate(folder: Path) -> tuple[Path, Path] | None:
    """Return (main .tex, jpp.cls) in `folder`, or None if either is missing."""
    if not folder.is_dir():
        return None
    tex = sorted(folder.rglob(TEX_NAME))
    if not tex:
        tex = sorted(p for p in folder.rglob("*.tex")
                     if r"\begin{document}" in p.read_text(encoding="utf-8", errors="replace"))
    cls = sorted(folder.rglob(CLS_NAME))
    return (tex[0], cls[0]) if len(tex) == 1 and cls else None


def obtain(folder: Path) -> tuple[Path, Path] | None:
    """Find the source in `folder`; if absent, run the fetch helper (as a separate process, isolated mode) once."""
    found = locate(folder)
    if found is None and FETCH.is_file():
        print(f"CMM source not found; running {FETCH.name}", flush=True)
        try:
            res = subprocess.run([sys.executable, "-I", str(FETCH), str(folder)], capture_output=True, text=True,
                                 timeout=600)
            for line in (res.stdout + res.stderr).splitlines():
                print("  " + line)
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"  fetch failed: {exc!r}")
        found = locate(folder)
    return found


SRC_DIR = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_SRC
found = obtain(SRC_DIR)
if found is None:
    print("SKIPPED: the LaTeX source of arXiv:2607.27981v1 could not be obtained "
          "(run ../fetch_cmm.py with network access, or pass the folder that holds it)")
    sys.exit(0)
TEX, CLS = found

fails = []


def strip_comments(line: str) -> str:
    """Remove a LaTeX % comment (not an escaped \\%)."""
    out = []
    i = 0
    while i < len(line):
        if line[i] == '%' and (i == 0 or line[i - 1] != '\\'):
            break
        out.append(line[i])
        i += 1
    return ''.join(out)


src = open(TEX, encoding='utf-8', errors='replace').read()
cls = open(CLS, encoding='utf-8', errors='replace').read()
# confirm the numbering rule in jpp.cls
rule_main = r'\renewcommand\theequation{\thesection.\@arabic\c@equation}' in cls
rule_app = r'\renewcommand\theequation{\thesection\,\@arabic\c@equation}' in cls
rule_reset = r'\setcounter{equation}\z@' in cls
nores = r'\nosectioneqnreset' in src
ok_rule = (rule_main and rule_app and rule_reset) and not nores
print(f"  [{'PASS' if ok_rule else 'FAIL'}] jpp.cls: main (section.n) rule {rule_main}, appendix (B n) rule {rule_app},"
      f" reset at \\section {rule_reset}; paper calls \\nosectioneqnreset: {nores}")
if not ok_rule:
    fails.append("numbering rule")

text = '\n'.join(strip_comments(l) for l in src.split('\n'))
body = text[text.index(r'\begin{document}'):]

# tokenise: sections, appendix, numbered environments
tok = re.compile(r'\\section\*?\{|\\appendix\b|\\begin\{(equation|align|eqnarray)(\*?)\}|\\\[|\\end\{document\}')
pos = 0
sec = 0
appendix = False
eqn = 0
labels = {}
order = []
while True:
    m = tok.search(body, pos)
    if not m:
        break
    t = m.group(0)
    if t.startswith(r'\section'):
        starred = t.startswith(r'\section*')
        if not starred:
            sec += 1
            eqn = 0
        pos = m.end()
        continue
    if t.startswith(r'\appendix'):
        appendix = True
        sec = 0
        eqn = 0
        pos = m.end()
        continue
    if t == r'\[':
        end = body.index(r'\]', m.end())
        pos = end + 2
        continue
    if t.startswith(r'\end{document}'):
        break
    env, star = m.group(1), m.group(2)
    endtag = r'\end{' + env + star + '}'
    end = body.index(endtag, m.end())
    content = body[m.end():end]
    pos = end + len(endtag)
    if star:
        continue
    if env == 'equation':
        lines = [content]
    else:
        lines = re.split(r'\\\\', content)
        lines = [l for l in lines if l.strip()]
    for l in lines:
        numbered = not (r'\nonumber' in l or r'\notag' in l)
        if numbered:
            eqn += 1
        secname = chr(ord('A') + sec - 1) if appendix else str(sec)
        num = f"{secname}{eqn}" if appendix else f"{secname}.{eqn}"
        for lab in re.findall(r'\\label\{([^}]*)\}', l):
            labels[lab] = num if numbered else None
            order.append(lab)

# CMM labels cited in the paper and the derivations: (label, number quoted in the paper or the derivations, role)
USED = [
    ('eq:poissonbracket', '2.9', 'Poisson bracket {f,g} = z.(grad f x grad g)'),
    ('eq:WGK', '2.13', 'gyrokinetic free energy W'),
    ('eq:defHGK', '2.15', 'gyrokinetic helicity H'),
    ('eq:realimag', '2.17', 'Plemelj split of H into P-integral and i pi (psi^2/F)_{v=0}'),
    ('eq:defpsi', '2.18', 'psi_s = h_s - (q_s/T_0s) <chi>_R F_0s (Section 4: psi_s at v_par = 0)'),
    ('eq:GKvpar0', '2.22', 'v_par = 0 equation'),
    ('eq:ImHcons', '2.23', 'conservation of Im H'),
    ('eq:kperprhoe', '3.1', 'contains tau = T_0i/T_0e'),
    ('eq:defphiA', '3.2', 'normalised potentials varphi, A'),
    ('eq:ddt', '3.3', 'd/dt = d_t + u_perp.grad'),
    ('eq:gradpar', '3.4', 'grad_par = d_z + b_perp.grad/v_A'),
    ('eq:dAdt', '3.6', 'ITEF induction equation, with the grad_par(dn/n) term (Section 4, finite k_perp rho_i)'),
    ('eq:econt', '3.7', 'ITEF electron continuity, with the rho_i {dn/n, dB/B} term (Section 4)'),
    ('eq:dn', '3.10', 'ITEF quasineutrality with Gamma_0 and J_0 (Section 4)'),
    ('eq:J0J1', '3.11', 'J0 and 2J1/a expansions'),
    ('eq:Gamma0', '3.13', 'Gamma0, Gamma1 expansions'),
    ('eq:dB', '3.14', 'ITEF perpendicular pressure balance with Gamma_1 and J_1 (Section 4)'),
    ('eq:dgdt', '3.16', 'ITEF ion kinetic equation'),
    ('eq:gkeKRMHD', '3.23', 'KRMHD induction'),
    ('eq:vorticity', '3.24', 'KRMHD vorticity'),
    ('eq:KRMHDdgdt', '3.25', 'KRMHD ion kinetic equation for g_09'),
    ('eq:defg09', '3.26', 'g_09 = g_i - vperp^2 dB/B F0'),
    ('eq:dnKRMHD', '3.27', 'overline g_09 = dn/n - dB/B'),
    ('eq:perpAmpereKRMHD', '3.28', 'perpendicular pressure balance'),
    ('eq:KRMHDWcompr', '3.31', 'compressive free energy'),
    ('eq:HnAKRMHD', '3.33', 'H_ph-sp in KRMHD'),
    ('eq:KRMHDdndB', 'B1', 'dn/n = int G_n, dB/B = int G_B'),
    ('eq:Gn', 'B2', 'G_n'),
    ('eq:GB', 'B3', 'G_B'),
    ('eq:dGpmdt', 'B4', 'branch equation'),
    ('eq:GpGm', 'B5', 'G+ and G-'),
    ('eq:g09sigma', 'B6', 'sigma'),
    ('eq:Lambdapm', 'B7', 'Lambda+-'),
    ('eq:Gl', 'B8', 'Laguerre expansion'),
    ('eq:dGldt', 'B10', 'l >= 2 free streaming'),
    ('eq:Wpmcons', 'B12', 'dW+-/dt = 0 and dWtilde/dt = 0 (Appendix A)'),
    ('eq:Wpmcompr', 'B13', 'W+-'),
    ('eq:defWtilde', 'B14', 'W tilde'),
    ('eq:Wcomprtotal', 'B15', 'W_compr decomposition'),
    ('eq:defkappa', 'B16', 'kappa'),
    ('eq:Hentrtotal', 'B17', 'H decomposition'),
    ('eq:Hpm', 'B18', 'H+-'),
    ('eq:Hentrtilde', 'B19', 'H tilde'),
    ('eq:Khel0', 'B20', 'd/dt int G^2 v/(2F)'),
    ('eq:Khel1', 'B21', 'same, integrated by parts'),
    ('eq:Mn', 'B22', 'moments M_n'),
    ('eq:dM0dt', 'B23', 'dM0/dt'),
    ('eq:dM1dt', 'B24', 'dM1/dt'),
    ('eq:Gammapm', 'B25', 'Gamma+-'),
    ('eq:Gammacons', 'B26', 'dGamma/dt = 0'),
    ('eq:hermiteexp', 'B27', 'Hermite expansion'),
    ('eq:orthogonality', 'B29', 'Hermite orthogonality'),
    ('eq:valGm', 'B30', 'Hermite coefficients'),
    ('eq:WcomprHerm', 'B31', 'W+- in Hermite form'),
    ('eq:GammapmHerm', 'B32', 'Gamma+- in Hermite form'),
]
print(f"         {'label':22s} {'number':>7s} {'cited':>6s}  role")
for lab, cited, role in USED:
    num = labels.get(lab, 'MISSING')
    ok = num == cited
    print(f"  [{'PASS' if ok else 'FAIL'}] {lab:22s} {str(num):>7s} {cited:>6s}  {role}")
    if not ok:
        fails.append(lab)


# formula strings (whitespace-insensitive) that the derivations rely on
def squash(s: str) -> str:
    """Remove all whitespace."""
    return re.sub(r'\s+', '', s)


S = squash(text)
CHECKS = [
    ('Lambda+- (B7)', r'\Lambda^\pm=-\frac{\tau}{Z}+\beta_{i}^{-1}\pm\sqrt{\left(1+\frac{\tau}{Z}\right)^2+\beta_{i}^{-2}}'),
    ('sigma (B6)', r'\sigma=1+\frac{\tau}{Z}+\beta_{i}^{-1}+\sqrt{\left(1+\frac{\tau}{Z}\right)^2+\beta_{i}^{-2}}'),
    ('G+ (B5)', r'G^+=G_B+\sigma^{-1}\left(1+\frac{Z}{\tau}\right)G_n'),
    ('G- (B5)', r'G^-=G_n+\frac{2\tau}{\sigmaZ\beta_{i}}G_B'),
    ('kappa (B16) as printed', r'\kappa=\left[\left(1+\frac{\tau}{Z}\right)+\frac{1}{\beta_i^2}\right]^{1/2}'),
    ('tau definition', r'\tau\equivT_{0i}/T_{0e}'),
    ('F_M definition', r'F_M(v_\parallel)\equiv\pi^{-1/2}v_{\rm th\iti}^{-1}e^{-\hat{v}_\parallel^2}'),
    ('v_th definition', r'v_{\rm th\its}=\sqrt{2T_{0s}/m_s}'),
]
for name, pat in CHECKS:
    ok = squash(pat) in S
    print(f"  [{'PASS' if ok else 'FAIL'}] string check {name:28s}: {'found' if ok else 'NOT FOUND'}")
    if not ok:
        fails.append(name)

print(f"total labels parsed: {len(labels)}; appendix B labels: "
      + ", ".join(f"{l}={labels[l]}" for l in order if labels.get(l) and labels[l].startswith('B')))
if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("ALL s00 CHECKS PASSED")
