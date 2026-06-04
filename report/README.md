# HTGNN Forex Typst Report

This folder contains a Typst draft for the Geometric Deep Learning final project
report. It follows the file structure of the provided Typst example while using
content adapted to the uploaded `htgnn-forex-trading` repository.

## Structure

```text
main.typ
references.bib
template.typ
content/
  00-index.typ
  01-introduction.typ
  02-data-structuring.typ
  03-signal-structuring.typ
  04-allocation-strategies.typ
  05-baselines.typ
  06-htgnn-model.typ
  07-results.typ
  08-conclusions.typ
  09-future-work.typ
  10-ai-disclosure.typ
figures/graphics/
  htgnn_architecture.svg
```

## Compile

From this directory:

```bash
typst compile main.typ main.pdf
```

The report imports the Typst package `@preview/bloated-neurips:0.7.0`, matching
the style of the provided example template. If your Typst setup cannot download
preview packages automatically, install/cache that package first or replace the
NeurIPS wrapper in `main.typ` with your local template.

## Sections intentionally left blank

`content/07-results.typ` and `content/08-conclusions.typ` contain only section
headings, as requested. Fill them after the final experimental runs.
