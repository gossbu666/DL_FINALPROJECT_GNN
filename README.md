# GNN Over-Squashing × Adversarial Robustness

Course project (Deep Learning, AIT). Studies whether over-squashing fixes (add-based **FoSR** vs
remove-based **ProxyDelete** spectral pruning) also help adversarial robustness, through the
**"add vs remove edge"** lens, against a purpose-built defense (**GCN-Jaccard**) and an undefended GCN.

See [PROJECT_BRIEF.md](PROJECT_BRIEF.md) — the canonical design doc (read its DECISIONS and OPEN
QUESTIONS before changing anything). **All reported numbers must be real and reproducible.**

## Setup

`.venv` is a `--system-site-packages` venv that reuses the system torch (2.9.1+cu128) and adds
PyTorch Geometric (2.8.0). Compute runs on **CPU** for the small Planetoid graphs (`device='cpu'`)
to sidestep Blackwell sm_120 / DeepRobust-CUDA issues.

```bash
python -m venv --system-site-packages .venv
./.venv/bin/pip install torch_geometric
```

## Run

```bash
# Task 0 — GCN baseline on Cora (5 seeds, CPU)
./.venv/bin/python -m scripts.run_cora_baseline --dataset Cora --runs 5 --device cpu
```

Current baseline: **Cora GCN clean test acc = 80.30 ± 0.47** (5 seeds). See `results/cora_baseline.json`.

## Layout

```
src/
  data.py     # Planetoid loaders (Cora/Citeseer) + load-time stat check
  models.py   # GCN (primary); GAT/GraphSAGE = stretch
  train.py    # transductive training loop (best-val model selection)
  eval.py     # accuracy (clean); robust acc / gap added in Task 2
  rewiring/   # fosr.py (add), pruning.py (ProxyDelete remove), dropedge.py — added per task
scripts/
  run_cora_baseline.py   # Task 0
results/      # json results, tables, plots
```

## Status

- [x] Task 0 — GCN on Cora
- [ ] Task 1 — DeepRobust clean-env spike (Metattack 0.05 + GCN-Jaccard) ← next
- [ ] Task 2 — core pipeline + headline table
- [ ] Task 3 — FoSR (add) for node tasks
- [ ] Task 4 — Jacobian over-squashing probe (scoped)
- [ ] Task 5 — Citeseer + ProxyDelete integration
- [ ] Task 6 — slides + progress summary
