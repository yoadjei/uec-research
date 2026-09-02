"""Check the submission against the ICLR 2027 author guidelines.

Everything here is mechanically checkable from the sources. Anything that needs a real LaTeX build
(the exact page count) or a human (author quotas, reciprocal-reviewer eligibility, OpenReview
profiles) is reported as such instead of being silently passed.

    python paper/iclr2027/check_submission.py
"""
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
tex = (HERE / "main.tex").read_text(encoding="utf-8")
bib = (HERE / "references.bib").read_text(encoding="utf-8")

fails, warns = [], []


def check(name, ok, detail="", warn_only=False):
    mark = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    print(f"  [{mark}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        (warns if warn_only else fails).append(name)


print("=" * 86)
print("ANONYMITY  (breach = desk rejection)")
print("=" * 86)
body = tex.split(r"\begin{document}")[1]
check("no \\iclrfinalcopy active",
      not re.search(r"^[^%\n]*\\iclrfinalcopy", tex, re.M),
      "style file prints the anonymous block while it stays commented")
urls = re.findall(r"https?://[^\s},]+", body)
check("no URLs in the body", not urls, f"found {urls}" if urls else "")
selfref = [p for p in ["our previous work", "our earlier work", "our prior work",
                       "in our previous", "as we showed in", "github.com"]
           if p in body.lower()]
check("no self-identifying phrasing", not selfref, f"found {selfref}" if selfref else "")
check("no acknowledgements section", "acknowledg" not in body.lower())
check("no funding statement in body",
      not re.search(r"\b(funded by|grant no|this work was supported)\b", body, re.I))

print()
print("=" * 86)
print("REQUIRED AND RECOMMENDED SECTIONS")
print("=" * 86)
subs = re.findall(r"\\subsubsection\*\{([^}]+)\}", tex)
check("AI use statement present (required)",
      any("ai" in x.lower() for x in subs), f"sections: {subs}")
check("reproducibility statement present (recommended)",
      any("reproducib" in x.lower() for x in subs))
check("ethics statement present (recommended)",
      any("ethic" in x.lower() for x in subs))

order = [m.group(0) for m in re.finditer(
    r"\\subsubsection\*\{[^}]+\}|\\bibliography\{[^}]+\}|\\appendix", tex)]
bib_i = next(i for i, o in enumerate(order) if o.startswith(r"\bibliography"))
app_i = next((i for i, o in enumerate(order) if o == r"\appendix"), None)
stmt_i = [i for i, o in enumerate(order) if o.startswith(r"\subsubsection")]
check("statements come before the references", all(i < bib_i for i in stmt_i))
check("appendices come after the bibliography",
      app_i is not None and app_i > bib_i,
      "guidelines: appendices go after the bibliography")

print()
print("=" * 86)
print("FORMATTING")
print("=" * 86)
check("uses the provided style file", r"\usepackage{iclr2027_conference,times}" in tex)
for f in ["iclr2027_conference.sty", "iclr2027_conference.bst", "natbib.sty", "fancyhdr.sty"]:
    check(f"style file present: {f}", (HERE / f).exists())

figs = re.findall(r"includegraphics\[[^\]]*\]\{([^}]+)\}", tex)
missing = [f for f in figs if not (ROOT / "figures" / f).exists()]
check("all figures resolve", not missing, f"missing {missing}" if missing else f"{len(figs)} figures")

keys = set()
for m in re.findall(r"\\cite[tp]?\{([^}]+)\}", tex):
    keys |= {k.strip() for k in m.split(",")}
defined = set(re.findall(r"@\w+\{([^,]+),", bib))
check("every citation resolves", not (keys - defined), f"undefined: {sorted(keys - defined)}")
check("no uncited bib entries", not (defined - keys), f"uncited: {sorted(defined - keys)}")

for env in ("document", "abstract", "figure", "table", "enumerate", "tabular"):
    b = len(re.findall(rf"\\begin\{{{env}\}}", tex))
    e = len(re.findall(rf"\\end\{{{env}\}}", tex))
    check(f"{env} balanced", b == e, f"{b} begin / {e} end")

labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
check("no dangling \\ref", not (refs - labels), f"dangling: {sorted(refs - labels)}")

placeholders = re.findall(r"(TODO|FIXME|XXX|\bTBD\b|to be verified)", tex)
check("no placeholder text in the paper", not placeholders,
      f"found {set(placeholders)}" if placeholders else "")

# Placeholders in the .bib render into the reference list, so they are a submission blocker too.
bib_ph = re.findall(r"^\s*author\s*=\s*\{\{([^}]*(?:verif|TODO|TBD)[^}]*)\}\}", bib, re.M | re.I)
check("no placeholder authors in the bibliography", not bib_ph,
      f"would print in the reference list: {bib_ph}" if bib_ph else "")

print()
print("=" * 86)
print("PAGE BUDGET  (main text limit 9; references and appendices excluded)")
print("=" * 86)
main = body.split(r"\subsubsection*{Ethics statement}")[0]


def prose_words(t):
    t = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", " ", t, flags=re.S)
    t = re.sub(r"\\begin\{table\}.*?\\end\{table\}", " ", t, flags=re.S)
    t = re.sub(r"\$[^$]*\$", " x ", t)
    t = re.sub(r"\\cite[a-z]*\{[^}]*\}", " (ref) ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", t)
    return len(re.sub(r"[{}\\&~^_]", " ", t).split())


w = prose_words(main)
nfig = len(re.findall(r"\\begin\{figure\}", main))
ntab = len(re.findall(r"\\begin\{table\}", main))
print(f"  main-text prose words : {w}")
print(f"  floats                : {nfig} figures, {ntab} tables")

# Prefer the compiled document. The main text is everything before the ethics statement, since
# the statements, the references and the appendices are all excluded from the limit.
pages_txt = HERE / "out.txt"
measured = None
if (HERE / "main.pdf").exists() and pages_txt.exists():
    pages = pages_txt.read_text(encoding="utf-8", errors="replace").split("\f")
    for i, pg in enumerate(pages, 1):
        if "ETHICSSTATEMENT" in re.sub(r"\s+", "", pg.upper()):
            measured = i - 1
            break

if measured is not None:
    print(f"  MEASURED from main.pdf: main text = pages 1-{measured} ({measured} of 9)")
    check("main text within the page limit", measured <= 9,
          f"{measured} pages, measured on the compiled PDF")
else:
    est = w / 620 + nfig * 0.42 + ntab * 0.18 + 0.55
    print(f"  ESTIMATED PAGES       : ~{est:.1f} of 9  (no compiled PDF found)")
    check("estimated within the page limit", est <= 9.0,
          f"~{est:.1f} pages; ESTIMATE ONLY, run `make` to measure", warn_only=True)

print()
print("=" * 86)
print("NOT CHECKABLE HERE")
print("=" * 86)
for item in [
    "author quotas and reciprocal-reviewer eligibility: depends on OpenReview profiles",
    "dual-submission status: author knowledge",
    "reference accuracy for 2025-2026 entries: see docs/reference_audit.md",
    "supplementary code must be anonymised before upload",
]:
    print(f"  [MANUAL] {item}")

print()
print("=" * 86)
print(f"{len(fails)} failure(s), {len(warns)} warning(s)")
print("=" * 86)
sys.exit(1 if fails else 0)
