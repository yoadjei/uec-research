# Build the submission. Requires a TeX distribution; none was present when this was written.
main.pdf: main.tex references.bib
	pdflatex -interaction=nonstopmode main
	bibtex main
	pdflatex -interaction=nonstopmode main
	pdflatex -interaction=nonstopmode main
	@echo "--- page count ---"
	@pdfinfo main.pdf | grep Pages || true

clean:
	rm -f *.aux *.bbl *.blg *.log *.out main.pdf

.PHONY: clean
