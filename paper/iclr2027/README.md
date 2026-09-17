# Manuscript withheld during review

`main.tex` and `main.pdf` are deliberately not tracked here. This repository is public and every
commit carries the authors' names, so keeping the manuscript in it would let a reviewer who searches
a distinctive phrase from the paper arrive at the authors' identity. They are ignored in
`.gitignore` and will be restored once decisions are out.

Everything needed to rebuild them is still here: the ICLR style files, the bibliography, the
`Makefile` and `check_submission.py`. A fresh clone therefore has the build but not its input, and
`make` will fail until `main.tex` is put back.

The compiled PDF remains reachable in this repository's history, from the commit that added it. That
is a known limitation of removing it from the tip rather than rewriting history.

## Checks

    python check_submission.py           # ICLR guideline check; takes the page count from the PDF
    python ../../experiments/verify_paper.py   # re-derives 81 headline numbers from the artefacts
