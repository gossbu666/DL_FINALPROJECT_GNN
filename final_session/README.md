# Final submission (due Tue 7 Jul 2026)

Everything for the **final** deliverable: **presentation (slides PDF) + final report/paper**.
The final paper is a superset of the midway report (`../midway_report/`), which stays frozen as the
submitted midway snapshot.

```
final_session/
├── report/
│   ├── main.tex        # final paper (NeurIPS-like). [TODO] markers = fill with real numbers
│   ├── references.bib
│   └── figures/        # paper figures (add/remove diagram, Jacobian plot, ...)
└── slides/
    ├── slides.tex      # Beamer deck (Madrid). [TODO] markers = fill with results
    └── figures/
```

## Build

```bash
cd report  && latexmk -pdf main.tex     # -> report/main.pdf
cd ../slides && latexmk -pdf slides.tex  # -> slides/slides.pdf
```

(Run from `final_session/`. TeX Live already installed.)

## What's left to fill (the `[TODO]` markers)

The structure and prose are done; the gaps are exactly the things that need **real numbers**:

| Section | Fills from | Task |
|---|---|---|
| Results §6 headline table | `results/grid_cora.json` | 2 (GCN/DropEdge/Jaccard), 3 (FoSR), 5 (ProxyDelete) |
| Over-squashing probe | Jacobian probe output | 4 |
| Citeseer row | grid on Citeseer | 5 |
| Discussion / Conclusion / Abstract finding | the above results | after 2–5 |
| Figures | matplotlib / TikZ | as results land |

## Submission checklist (7 Jul)

- [ ] All `[TODO]` markers resolved (grep `\todo` in `report/main.tex` and `slides/slides.tex`)
- [ ] `report/main.pdf` builds clean, numbers match `results/*.json`
- [ ] `slides/slides.pdf` builds clean
- [ ] Author names/IDs correct; abstract states the actual finding
- [ ] Numbers reproducible from `scripts/` (integrity)
