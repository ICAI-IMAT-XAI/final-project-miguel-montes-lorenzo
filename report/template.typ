// ============================================================
//  template.typ - Small reusable helpers
// ============================================================

#let accent = rgb("#17324d")
#let rule-col = rgb("#d9dee7")

#let note(title: "Note", body) = block(
  width: 100%,
  inset: (x: 10pt, y: 8pt),
  radius: 4pt,
  stroke: (left: 3pt + accent, rest: 0.5pt + rule-col),
  fill: rgb("#f6f8fb"),
)[
  #text(weight: "bold")[#title.] #body
]

#let compact-table(columns, body) = table(
  columns: columns,
  inset: (x: 4pt, y: 4pt),
  stroke: 0.35pt + rule-col,
  align: horizon,
  body,
)
