# Reference audit

**Status: closed.** Every entry in `paper/iclr2027/references.bib` (43 entries, all cited, no
dangling keys) has been checked against its source by the authors. The document compiles with
BibTeX reporting no errors and LaTeX reporting no undefined citations or references.

This file records what was corrected, so the corrections are not silently lost if the bibliography
is ever regenerated.

---

## Corrections applied after verification

| key | what was wrong | corrected to |
|---|---|---|
| `mougan2025` | year given as 2025; David Masip missing from the author list | TMLR **2023**; Mougan, Broelemann, Masip, Kasneci, Thiropanis, Staab |
| `fass2026` | first names wrong on both authors | Kamalasankari Subramaniakuppusamy, Jugal Gajjar; accepted at XAI4CV, CVPR 2026 |
| `rsp2026` | title truncated | "…A Training-Time Signal for **Stable Evidence and Shortcut Reliance**"; submitted to ACL Rolling Review |
| `xray2026` | author list was `Elangovan, Aparna and others` | Kabilan Elangovan and Daniel Ting |
| `evoxplain2025` | first name wrong | Chama Bensmail (sole author) |
| `hypclass2026` | name split into surname and initial | `{{Thackshanaramana B.}}` as a single unit |
| `rethinkrobust2025` | three first names and one surname wrong | Panagiota Kiourti, Anu Singh, Preeti Duraipandian, Weichao Zhou, Wenchao Li |
| `deltaaudit2025` | both first names wrong | Arshia Hemmat, Afsaneh Fatemi |
| `hinder2022` | hyphen in the title | "Model based Explanations of Concept Drift" |
| `laberge2023` | no article or page numbers | JMLR 24, article 364, pages 1–50 |
| `attribimposs2026` | removed for want of an author list | Drake Caraker, Bryan Arnold, David Rhoads; **restored and cited** |
| `mcal2026` | removed for want of an author list | Shailesh Sridhar, Anton Xue, Eric Wong; **restored and cited** |

`agarwal2022ros` was checked and the recorded seven-author list is correct. `damour2020` keeps its
key while carrying the official JMLR 2022 metadata, which is standard practice.

### Knock-on edit

Moving `mougan2025` to 2023 made the phrase "the 2025--2026 literature" wrong, since the three
senses of "explanation drift" now span 2023 to 2026. Both occurrences were reworded.

### A bug the compile caught

Every 2025--2026 entry carried a trailing `% VERIFY` marker on its `@article{key,` line. **`%` is
not a comment character inside a BibTeX entry**, so the markers were parsed as field data and
BibTeX failed with 8 errors, silently emptying the author, title, journal and year of the affected
entries. All eight were removed. This is the reason a bibliography has to be built and not merely
written.

---

## Verified by compilation

Built with MiKTeX 25.12 (`make` in `paper/iclr2027/`):

- **Main text: 8 of the 9 pages allowed**, measured on the compiled PDF as everything preceding the
  ethics statement. References (pages 9–11) and appendices (page 12) are excluded from the limit
  under the ICLR rules. Twelve pages in total.
- BibTeX: no errors. LaTeX: no undefined citations, no undefined references.
- Overfull boxes: **0**. The related-work table originally ran 97pt past the text block and into the
  margin; it now uses wrapped fixed-width columns.
- No style file, margin or spacing was altered to gain space. The page count comes from the content.

`paper/iclr2027/check_submission.py` re-runs the full guideline check and takes the page count from
the compiled PDF whenever one is present.

---

## Remaining items, none of them references

1. **Author quotas and reciprocal-reviewer eligibility** depend on OpenReview profiles and cannot be
   checked from the repository. At least one author must be registered to review three papers.
2. **Supplementary code must be anonymised before upload.** The repository as it stands identifies
   the authors. The paper itself carries no repository URL, by design.
3. **Dual-submission status** is author knowledge.
4. If any 2025–2026 preprint is published between submission and camera ready, refresh its venue.
