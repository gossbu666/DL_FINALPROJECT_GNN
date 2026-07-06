# Over-Squashing vs. Adversarial Robustness in GNNs
### An "Add vs. Remove Edge" Investigation

Course project (Deep Learning, AIT). We ask a single question: **do methods that fix GNN
*over-squashing* also improve *adversarial robustness*?** — organized by the **"add vs. remove edge"**
axis (over-squashing rewiring adds/removes edges; attacks add edges; defenses remove them).

**Headline finding (honest negative result).** Over-squashing mitigations do **not** transfer to
robustness. Add-based rewiring (FoSR) and spectral remove-based pruning (ProxyDelete) behave like an
undefended GCN; only *attack-aware* defenses — GCN-Jaccard (similarity) and GCN-SVD (low-rank) — improve
robustness. The discriminating factor is the **editing criterion** (which edges you touch), **not** the
add/remove direction. Mechanism: Jaccard deletes ~50% of the attacker's inserted edges, while the
spectral/random methods are at chance. The result replicates on Citeseer, survives a budget-matched
control, and holds against a defense-aware attacker.

All numbers are real and reproducible from fixed-seed scripts.

## Headline results — Cora (test acc %, `nettack` split, Metattack, 3 seeds)

| Method | Clean | Robust@0.05 | Robust@0.10 | Gap@0.05 | Gap@0.10 |
|---|---|---|---|---|---|
| GCN (vanilla)        | 83.50 | 77.05 | 71.85 | 6.46 | 11.65 |
| FoSR (add)           | 82.53 | 76.78 | 70.96 | 5.75 | 11.57 |
| ProxyDelete (spectral rm.) | 82.60 | 76.71 | 71.68 | 5.89 | 10.92 |
| DropEdge (random rm.) | 83.17 | 76.98 | 71.40 | 6.19 | 11.77 |
| **GCN-Jaccard** (similarity rm.) | 82.16 | **78.84** | 74.45 | 3.32 | 7.71 |
| **GCN-SVD** (low-rank) | 77.58 | 77.20 | **75.29** | **0.39** | **2.30** |

Lower gap = more robust. Only the two purpose-built defenses separate from the vanilla-GCN level.

## Setup

`.venv` is a `--system-site-packages` venv reusing the system PyTorch (2.9.1+cu128); all compute runs on
**CPU** for these small graphs. DeepRobust 0.2.9 is installed around a broken `gensim` pin.

```bash
python -m venv --system-site-packages .venv
./.venv/bin/pip install torch_geometric
./.venv/bin/pip install "deeprobust==0.2.9" --no-deps
./.venv/bin/pip install gensim
./.venv/bin/pip install torch_sparse torch_scatter -f https://data.pyg.org/whl/torch-2.9.0+cu128.html
```

## Reproduce

```bash
# 0. GCN baseline (sanity)
./.venv/bin/python -m scripts.run_cora_baseline --dataset Cora --runs 5

# 1. Precompute + cache the Metattack poisoned graphs (slow; once per dataset)
./.venv/bin/python -m scripts.precompute_attacks --name cora     --ptb 0.05 0.10
./.venv/bin/python -m scripts.precompute_attacks --name citeseer --ptb 0.05 0.10

# 2. Headline grid (6 methods)  ->  results/grid_cora.json
./.venv/bin/python -m scripts.run_grid --methods gcn dropedge jaccard fosr proxydelete svd \
    --ptb 0.05 0.10 --seeds 0 1 2 --svd_k 50
./.venv/bin/python -m scripts.run_grid --name citeseer --methods gcn dropedge jaccard fosr proxydelete svd \
    --ptb 0.05 0.10 --seeds 0 1 2 --svd_k 50 --out results/grid_citeseer.json

# 3. Analyses
./.venv/bin/python -m scripts.edge_overlap          --ptb 0.05 0.10 --budget 643   # mechanism (recall)
./.venv/bin/python -m scripts.spectral_gap_analysis --ptb 0.05 --budget 253        # gap sanity
./.venv/bin/python -m scripts.run_probe             --layers 6 --ptb 0.10          # Jacobian probe (RQ1)
./.venv/bin/python -m scripts.adaptive_evasion      --budgets 253 503              # defense-aware attack
./.venv/bin/python -m scripts.run_grid --methods gcn dropedge jaccard fosr proxydelete svd \
    --remove_budget_abs 643 --out results/grid_cora_budget643.json                 # budget-matched control

# 4. Figures
./.venv/bin/python -m scripts.make_figures --grid results/grid_cora.json
```

## Layout

```
src/
  data.py, models.py, train.py, eval.py   # PyG GCN baseline (Task 0)
  attacks.py                              # Metattack + on-disk cache
  methods.py                              # method registry (adj -> adj): the add/remove axis
  defenses.py                             # GCN-Jaccard pruning + GCN-SVD trainer
  dr_train.py                             # DeepRobust GCN trainer (uniform across methods)
  probe.py                                # Jacobian over-squashing sensitivity (RQ1/RQ3)
  rewiring/ fosr.py (add) pruning.py (ProxyDelete remove) dropedge.py (random)
scripts/  run_cora_baseline, precompute_attacks, run_grid, run_probe,
          edge_overlap, spectral_gap_analysis, adaptive_evasion, make_figures
results/  *.json (every reported number) + cache/ (poisoned graphs)
final_session/
  report_acl/   ACL-format final paper  (main.pdf)  <- submission format
  report/       NeurIPS-style paper      (main.pdf)
  slides/       Beamer deck (slides.pdf, slides_rev1.pdf)
  final_slide/  designed deck (Deep Learning_slide.pdf)
  slide_pngs/   every figure/table as PNG
  presentation_script.md                 # ~10-min talk script
PROJECT_BRIEF.md                          # design doc & decisions
```

## Methods, attack, data

- **Methods:** GCN (baseline); FoSR (add) & ProxyDelete spectral pruning (remove) — over-squashing arms;
  GCN-Jaccard (similarity) & GCN-SVD (low-rank) — defenses; DropEdge (random) — control.
- **Attack:** Metattack (global poisoning) via DeepRobust, ptb ∈ {0.05, 0.10}.
- **Data:** Cora & Citeseer (DeepRobust `nettack` split, largest connected component).
- **Pipeline:** `clean → Metattack → method → train GCN → eval`; poisoned graphs cached and reused.

## References for the ported code
FoSR — Karhadkar et al., ICLR 2023 (`github.com/kedar2/FoSR`). ProxyDelete — Jamadandi et al., NeurIPS
2024 (`github.com/RelationalML/SpectralPruningBraess`). DeepRobust — Li et al., 2020.

## Team
Supanut Kompayak (st126055), Dechathon Niamsa-Ard (st126235) — Deep Learning, Asian Institute of Technology.
