MAIN = main

.PHONY: all clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex refs.bib figures/*.pdf
	latexmk -pdf -interaction=nonstopmode $(MAIN).tex

clean:
	latexmk -C
	rm -f $(MAIN).bbl $(MAIN).run.xml
