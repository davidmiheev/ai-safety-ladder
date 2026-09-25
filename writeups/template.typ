// Shared style for the ladder write-ups. Each section file does:
//   #import "template.typ": *
//   #show: writeup.with(title: ..., section: ..., summary: [...])

#let accent = rgb("#3d5a4a")
#let stamp = rgb("#8a4a2f")
#let soft = rgb("#5c5548")
#let rule = rgb("#c9c0a8")

// Fixed-precision number formatting (Typst's calc.round drops trailing zeros).
#let fx(x, d: 2) = {
  let s = str(calc.round(float(x), digits: d))
  if d == 0 { return s }
  let parts = s.split(".")
  let frac = if parts.len() > 1 { parts.at(1) } else { "" }
  parts.at(0) + "." + frac + "0" * (d - frac.len())
}
// Signed, e.g. +0.81 / −0.40
#let sfx(x, d: 2) = if float(x) >= 0 { "+" + fx(x, d: d) } else { "−" + fx(-float(x), d: d) }
#let pct(x, d: 0) = fx(float(x) * 100, d: d) + "%"

#let callout(title: none, body) = block(
  width: 100%, inset: (x: 12pt, y: 10pt), radius: 3pt,
  fill: accent.lighten(92%), stroke: (left: 2.5pt + accent),
)[
  #if title != none [#text(weight: "bold", fill: accent)[#title] \ ]
  #body
]

#let finding(body) = callout(title: "Finding", body)
#let caveat(body) = block(
  width: 100%, inset: (x: 12pt, y: 10pt), radius: 3pt,
  fill: stamp.lighten(92%), stroke: (left: 2.5pt + stamp),
)[#text(weight: "bold", fill: stamp)[Caveat] \ #body]

#let status(done, body) = {
  let (label, color) = if done == true { ("done", accent) } else if done == "partial" { ("in progress", stamp) } else { ("to do", soft) }
  box(inset: (x: 5pt, y: 2pt), radius: 2pt, stroke: 0.6pt + color, text(size: 8pt, fill: color, weight: "bold", upper(label)))
  h(6pt)
  body
}

#let status-table(..rows) = {
  set par(justify: false)
  grid(
    columns: (auto, 1fr), column-gutter: 8pt, row-gutter: 7pt, align: (right, left),
    ..rows.pos().map(((done, body)) => {
      let (label, color) = if done == true { ("done", accent) } else if done == "partial" { ("in progress", stamp) } else { ("to do", soft) }
      (box(inset: (x: 5pt, y: 2pt), radius: 2pt, stroke: 0.6pt + color,
        text(size: 8pt, fill: color, weight: "bold", upper(label))), body)
    }).flatten(),
  )
}

#let cmd(s) = block(
  width: 100%, inset: 8pt, radius: 2pt, fill: luma(245),
  text(font: "DejaVu Sans Mono", size: 8.5pt, s),
)

#let writeup(title: "", section: "", summary: none, date: none, body) = {
  set document(title: section + " — " + title, author: "AI Safety Experiments Ladder")
  set page(
    paper: "a4", margin: (x: 2.2cm, top: 2.2cm, bottom: 2.2cm), numbering: "1",
    header: context {
      if counter(page).get().first() > 1 [
        #set text(size: 8pt, fill: soft)
        #section — #title #h(1fr) AI Safety & Interpretability Ladder
        #v(-6pt) #line(length: 100%, stroke: 0.4pt + rule)
      ]
    },
  )
  set text(font: "New Computer Modern", size: 10.5pt, lang: "en")
  set par(justify: true, leading: 0.62em)
  set heading(numbering: "1.1")
  show heading.where(level: 1): it => {
    v(1.1em)
    text(size: 14pt, fill: accent, weight: "bold", it)
    v(0.4em)
  }
  show heading.where(level: 2): it => { v(0.7em); text(size: 11.5pt, weight: "bold", it); v(0.2em) }
  set math.equation(numbering: "(1)")
  show figure.caption: set text(size: 9pt, fill: soft)
  set figure(gap: 8pt)
  show table: set text(size: 9pt)
  set table(stroke: (x, y) => if y == 0 { (bottom: 0.7pt + black) } else { none }, inset: (x: 6pt, y: 4pt))
  show raw: set text(font: "DejaVu Sans Mono", size: 8.5pt)
  show link: it => text(fill: accent, it)

  // Title block
  text(size: 9pt, fill: stamp, tracking: 0.12em, upper("AI Safety & Interpretability · Experiments Ladder"))
  v(2pt)
  text(size: 22pt, weight: "bold")[#section #h(6pt) #title]
  v(-4pt)
  line(length: 100%, stroke: 1.2pt)
  if date != none { text(size: 9pt, fill: soft, date) }
  if summary != none {
    v(4pt)
    callout(title: "Summary", summary)
  }
  v(6pt)
  body
}
