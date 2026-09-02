# Build the submission and report the main-text page count.
#
# MiKTeX here is a per-user install, so it is not on PATH by default.
MK ?= $(HOME)/AppData/Local/Programs/MiKTeX/miktex/bin/x64
PDFLATEX ?= $(MK)/pdflatex.exe
BIBTEX ?= $(MK)/bibtex.exe
PDFTOTEXT ?= $(MK)/miktex-pdftotext.exe

main.pdf: main.tex references.bib
	"$(PDFLATEX)" -interaction=nonstopmode main.tex
	"$(BIBTEX)" main
	"$(PDFLATEX)" -interaction=nonstopmode main.tex
	"$(PDFLATEX)" -interaction=nonstopmode main.tex
	@"$(PDFTOTEXT)" main.pdf out.txt
	@echo "--- overfull boxes (should be 0) ---"
	@grep -c Overfull main.log || true
	@python check_submission.py

check: main.pdf
	@python check_submission.py

clean:
	rm -f *.aux *.bbl *.blg *.log *.out out.txt *.png

.PHONY: check clean
