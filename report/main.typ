// ============================================================
//  main.typ - Entry point
//  Geometric Deep Learning Final Project - Typst draft
// ============================================================

#import "@preview/bloated-neurips:0.7.0": neurips2023
#import "template.typ": compact-table, note

#set page(margin: (x: 1.35cm, y: 1.6cm))
#set text(font: "TeX Gyre Termes", size: 9pt)

#let authors = (
  (
    name: "Miguel Montes Lorenzo",
    affl: ("imat", "comillas"),
    email: "202105503@alu.comillas.edu",
  ),
)


#let affls = (
  "imat": (
    department: "Mathematical Engineering and Artificial Intelligence",
    institution: "",
    location: "",
  ),
  "comillas": (
    department: "",
    institution: "Universidad Pontificia Comillas",
    location: "Madrid, Spain",
  ),
)

#show: neurips2023.with(
  title: [Heterogeneous Temporal Graph Neural Networks for Forex Portfolio Allocation],
  authors: (authors, affls),
  keywords: (
    "geometric deep learning",
    "heterogeneous temporal graphs",
    "foreign exchange",
    "portfolio allocation",
    "mean-variance optimisation",
  ),
  abstract: [
    #set text(size: 9pt)
    Foreign exchange portfolio allocation is a suitable setting for
    heterogeneous temporal graph learning because currencies are relative assets
    linked to rates, commodities, equity risk, and macro-financial regimes. This
    project studies whether a Heterogeneous Temporal Graph Neural Network
    (HTGNN) can use those relations more effectively than pointwise neural
    baselines when allocating across USD, EUR, JPY, GBP, CNY, CAD, AUD, and CHF.
    The pipeline downloads daily Yahoo Finance series, groups instruments into
    market blocks, converts FX quotes into USD-denominated currency returns, and
    trains models to predict long-only allocation vectors rather than raw
    returns. The main modelling choice is to replace direct return prediction
    with supervised allocation imitation: each target is an ex-post
    mean-variance teacher portfolio built from the next realised cross-section
    and recent covariance matrix. The HTGNN extends the signal-only setup by
    encoding each market block as a temporal node with a GRU and applying
    relation-typed message passing centred on the portfolio signal node. The
    reported backtests compare the graph model with a pointwise NN and simple
    currency benchmarks.
  ],
  bibliography: bibliography("references.bib"),
  accepted: none,
  aux: (
    get-notice: accepted => [],
  ),
)

#set page(
  margin: (left: 2.2cm, right: 2.2cm, top: 1.7cm, bottom: 1.7cm),
  footer: context {
    let page-number = counter(page).at(here()).first()
    if page-number == 1 {
      []
    } else {
      align(center, text(size: 10pt, [#page-number]))
    }
  },
)
#show figure: set block(spacing: 1.6em)
#let small-caption(width: 100%, it) = {
  set align(center)
  block(width: width, context {
    set align(left)
    set text(size: 8pt)
    it.supplement
    if it.numbering != none {
      [ ]
      it.counter.display(it.numbering)
    }
    it.separator
    [ ]
    it.body
  })
}
#show figure.caption.where(kind: image): small-caption.with(width: 60%)
#show figure.caption.where(kind: table): small-caption
#let bottom-figure(it) = {
  let body = block(width: 100%, spacing: 1.6em, {
    set align(center)
    it.body
    v(8pt, weak: true)
    it.caption
  })

  if it.placement == none {
    body
  } else {
    place(it.placement, body, float: true, clearance: 2.3em)
  }
}
#show figure.where(kind: image): bottom-figure
#show figure.where(kind: table): bottom-figure

#include "content/00-index.typ"
