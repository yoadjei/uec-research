# Reference audit: what needs human verification before submission

The bibliography is `paper/iclr2027/references.bib` (43 entries, all cited, no dangling keys).

The assistant that drafted this paper has a knowledge cutoff of **May 2026**. Every entry dated
2025 or 2026 sits at or beyond that boundary. Those entries were compiled during the literature
pass from `docs/lit_matrix.csv` (titles, venues, arXiv identifiers) and `paper/novelty_delta.md`
(author lists), both built earlier in the project, but they cannot be re-confirmed from memory now.
They are listed below in priority order.

Older, widely cited work is listed separately at the end; those entries are standard and low-risk,
though page and volume numbers still merit a glance.

---

## Priority 1: removed from the submission, restore once verified

Two papers are catalogued in `lit_matrix.csv` but are **not cited in the current draft**, because
their author lists could not be supplied and `natbib` builds a citation from the author field: an
entry without one renders a broken citation and prints the placeholder into the reference list.

| key | arXiv | what to supply | where it belonged |
|---|---|---|---|
| `attribimposs2026` | 2605.21492 | full author list, title, month | theory corollary, on collinearity |
| `mcal2026` | 2603.04831 | full author list, title, month | protocol, with `rethinkrobust2025` |

Supply the author lists and both can be restored; the sentences currently make their points without
the citation, so nothing is misattributed in the meantime. `paper/iclr2027/check_submission.py`
fails if a placeholder author ever reaches the bibliography.

## Priority 2: author list taken from our own earlier notes, never re-checked

Confirm the **author list, exact title, year and venue** against the arXiv abstract page. Several
of these were recorded from a single pass and one is known to have been wrong once already: the
audit that seeded this project misattributed the ROS author list to the OpenXAI author set, and
listed Delta-Audit as single-author when it has two.

| key | arXiv | author list as recorded | also confirm |
|---|---|---|---|
| `fass2026` | 2604.02532 | Subramaniakuppusamy & Gajjar | that it is the CVPR 2026 XAI4CV workshop |
| `rsp2026` | 2601.11625 | Dhayalkar | title wording; venue status (ACL ARR) |
| `xray2026` | 2604.08513 | Elangovan et al. | the full list; `et al.` is a placeholder |
| `evoxplain2025` | 2512.22240 | Bensmail | whether there are co-authors |
| `hypclass2026` | 2603.15821 | Thackshanaramana, B. | name form and initials |
| `rethinkrobust2025` | 2512.06665 | Kiourti, Singh, Duraipandian, Zhou & Li | order and spelling |
| `deltaaudit2025` | 2508.19589 | Hemmat & Fatemi | first names; this one was corrected once |
| `mougan2025` | 2303.08081 | Mougan, Broelemann, Kasneci, Tiropanis & Staab | that the TMLR year is 2025, not 2023 |
| `agarwal2022ros` | 2203.06877 | Agarwal, Johnson, Pawelczyk, Krishna, Saxena, Zitnik & Lakkaraju | this list was wrong once; check carefully |

## Priority 3: cited in the bib but not in the paper's argument

`hinder2022` was first entered under a guessed title and has been corrected to match
`lit_matrix.csv` ("Model-Based Explanations of Concept Drift", arXiv 2303.09331). The corrected
title is still from our own notes and not re-confirmed; check it, since the Hinder/Vaquet/Hammer
group has several closely related papers and the exact one matters for the Section 2 sentence.

## Priority 4: standard works, low risk, worth a glance at volume and pages

`sundararajan2017`, `lundberg2017`, `lundberg2020tree`, `ribeiro2016`, `smilkov2017`,
`simonyan2013`, `erion2021`, `alvarez2018`, `yeh2019`, `bhatt2020`, `openxai2022`, `quantus2023`,
`krishna2022`, `ghorbani2019`, `dombrowski2019`, `slack2020`, `laberge2023`, `kulinski2023`,
`jain2019`, `ding2021folktables`, `chen2016xgboost`, `sanh2019distilbert`, `maas2011imdb`,
`kokhlikyan2020captum`, `krizhevsky2009cifar`, `hendrycks2019benchmarking`, `cliff1993`,
`holm1979`, `schuirmann1987`, `okabe2008`.

Two specific items in this group to check rather than assume:

- **`damour2020`** is entered with journal year 2022 (JMLR 23(226)) while the key says 2020, which
  is the arXiv year. Both are defensible; make the key and the year agree, or leave the key and
  keep the JMLR year.
- **`laberge2023`** has a JMLR volume but no page or article number.

---

## Other pre-submission items that are not references

1. **No LaTeX toolchain was available**, so `main.tex` has never been compiled. Page count is
   estimated at **~8.4 of 9 pages** by word and float counting, which is close enough to the limit
   that it must be checked on a real build before submission.
2. **Anonymity.** The paper contains no author names and no repository URL. The public repository
   for this project would de-anonymise it, so the reproducibility statement points to anonymised
   supplementary material instead. Do not paste the repository link into the submission.
3. **Appendix A is referenced from the introduction** and now exists; the cross-reference resolves.
4. The **AI use statement** is included and, per the ICLR guidelines, does not count toward the
   page limit.
