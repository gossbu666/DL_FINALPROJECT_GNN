# PROJECT BRIEF — Over-Squashing × Adversarial Robustness in GNNs

> **Course project (Deep Learning, AIT).** This is a **learning project**, not a novelty
> contribution. The goal is to *investigate* a question and *understand the mechanism*, using
> existing methods as tools. Do not frame anything as "novel" or "no prior work does this."
>
> **For the AI agent reading this:** Read the `DECISIONS` and `OPEN QUESTIONS` sections before
> writing code. If you disagree with a decision, say so — don't silently override it. **Report
> real numbers only.** Negative or weak results (e.g. "rewiring did not help robustness") are
> fully acceptable outcomes and should be written up honestly. Never fabricate or "clean up"
> results.

---

## 1. TL;DR + STATUS

We study whether methods that fix **over-squashing** (a long-range information bottleneck in
GNNs) also help **adversarial robustness** (resistance to a few maliciously edited edges),
through the lens of **"add vs remove edge"**.

- **Status (28 Jun):** Task 0 **done** — GCN on Cora running, **80.30 ± 0.47** test acc over 5 seeds
  (CPU). Remove-arm code (Jamadandi ProxyDelete) **confirmed usable**. Repo scaffolded (`src/`,
  `scripts/run_cora_baseline.py`, results saved).
- **Immediate next action:** **Task 1 — DeepRobust clean-env spike** (Metattack 0.05 + GCN-Jaccard on
  Cora, end-to-end once). This is the gating risk; it must pass before building the grid.
- **Hard deadlines:**
  - **Phase 2 — Midway Report / Experiment Protocol:** Tue **30 Jun 2026, 07:45** (a *written plan*,
    no presentation). ~70% is adapted from the existing proposal.
  - **Final presentation:** Tue **7 Jul 2026** — this is the **FINAL** presentation (complete project,
    NOT a progress update). Bar is higher: need *complete* results — the full headline table
    (GCN / FoSR / ProxyDelete / GCN-Jaccard × {clean, 0.05, 0.10}, clean+robust+gap, multi-seed) +
    Jacobian probe results + ideally Citeseer — all real numbers, with a clear narrative answering RQ2.
    **Deliverable = presentation (slides PDF) + final report/paper.** The final paper grows out of the
    midway report (`midway_report/main.tex`): add a Results section + Conclusion, update the abstract.

---

## 2. THE CORE IDEA (so the agent understands *why*)

GNNs pass messages hop-by-hop along edges. Two known problems both come from **how information
flows through graph structure**:

- **Over-squashing** — to reach distant nodes you need many layers, but the (exponentially growing)
  long-range information gets compressed through structural **bottlenecks** into a fixed-size vector
  and is lost. → output is **not sensitive enough** to distant signal.
- **Adversarial fragility** — an attacker adds/removes a few edges; message passing pulls in the
  bad signal and flips the prediction. → output is **too sensitive** to malicious structural change.

Both are about *how sensitive a node's output is to structural changes elsewhere*. This is the
conceptual bridge we *investigate* (not a claim we prove).

### The sharper framing we actually use: **add vs remove edge**

This is the spine of the project:

- Over-squashing rewiring methods mostly **ADD** edges (FoSR, SDRF — to raise the spectral gap /
  relieve bottlenecks).
- Adversarial **attacks ADD** edges (typically between dissimilar nodes).
- Adversarial **defenses REMOVE** edges (GCN-Jaccard, Pro-GNN prune suspicious edges).

So adding edges to fix over-squashing might *hurt* robustness (more edges for an attacker to
exploit), while a *remove*-based over-squashing method (Jamadandi's spectral pruning) might transfer
to robustness *better* because it removes edges like a defense does. **"Add vs remove: which helps
robustness, and why?"** is a better learning question than "are they the same mechanism."

---

## 3. EXPERIMENTAL LOGIC (the method table = the whole design)

| Method | Targets | Edge op | Role in this study |
|---|---|---|---|
| **GCN** (vanilla) | — | none | baseline |
| **FoSR** | over-squashing | **add** | the "add" arm |
| **Spectral pruning (Jamadandi, "ProxyDelete")** | over-squashing | **remove** | the "remove" arm — **code confirmed usable** (`RelationalML/SpectralPruningBraess`, pure networkx/scipy, CPU-ok) |
| **GCN-Jaccard** | adversarial (defense) | **remove** | purpose-built defense reference |

What each comparison teaches:
- **FoSR vs Jamadandi** → does the add/remove *direction* matter among over-squashing fixes?
- **Jamadandi vs GCN-Jaccard** → is "remove for over-squashing" as good as "remove for defense"?
- **all vs vanilla GCN** → does any of this help robustness at all, and at what cost to clean accuracy?

### Pipeline order (poisoning → method → train → eval) — the operational spine

Metattack is a **poisoning** attack (edits the graph *before* training), so operation order is part of
the design, not an implementation detail:

```
clean graph ──A(Metattack)──► poisoned G' ──M──► G'' ──train GCN──► eval
                                          M ∈ {none, FoSR-add, ProxyDelete-remove, GCN-Jaccard}
```

- **A before train** — by definition of poisoning.
- **M after A** — mirrors deployment: receive a (possibly poisoned) graph → preprocess/defend → train.
  Matches how DeepRobust evaluates GCN-Jaccard.
- Attacker mostly **adds** dissimilar edges; Jaccard **removes** them (why "remove" defends); FoSR
  **adds** more (does adding hurt robustness?); ProxyDelete **removes** by spectral criterion (does
  over-squashing-motivated removal transfer to robustness?).

**Grid = 4 graph conditions per method** (a method changes the graph even with no attack, so its
clean-accuracy cost must be measured separately):

| graph fed to training | measures |
|---|---|
| clean | vanilla clean acc |
| clean + M | clean-accuracy cost of the method itself |
| A (poisoned, no M) | undefended robust acc |
| A + M | robust acc with the method |

Headline table: rows = {GCN, FoSR, ProxyDelete, GCN-Jaccard, (DropEdge)}; cols = {clean-acc,
robust@0.05, robust@0.10, gap = clean − robust}.

**Caveat — non-adaptive attacker:** Metattack attacks a vanilla GCN surrogate and is *unaware* of the
defense applied afterward (DeepRobust standard). This flatters defenses vs an adaptive attacker — note
as a limitation; adaptive attack is stretch.

**Cost control:** generate the poisoned adjacency *once* per ptb_rate, cache to disk, reuse across all
methods (Metattack is expensive — meta-gradient bilevel).

---

## 4. RESEARCH QUESTIONS (reframed: *investigate*, not *contribute*)

- **RQ2 (LEAD):** Do over-squashing mitigations (add-based FoSR vs remove-based pruning) improve
  adversarial robustness relative to (a) undefended GCN and (b) a purpose-built defense (GCN-Jaccard)
  — and at what cost to clean accuracy?
- **RQ1 (secondary, nearly free once the probe exists):** Does structural perturbation measurably
  worsen the over-squashing measure? Is there a relationship between perturbation strength and
  long-range information loss?
- **RQ3 (stretch):** Can a single measure (Jacobian sensitivity) capture both over-squashing severity
  and adversarial fragility on the same scale?

---

## 5. DATASETS

| Dataset | Source | Task | Size | Role |
|---|---|---|---|---|
| **Cora** | PyG Planetoid | node classification | 2,708 nodes / 5,429 edges / 7 classes / 1,433 feat | **primary** — node-level + standard adversarial-attack testbed |
| **Citeseer** | PyG Planetoid | node classification | 3,327 / 4,732 / 6 / 3,703 | second node-level setting (cheap swap) |
| Peptides-func | LRGB | graph classification (multilabel) | 15,535 graphs, ~150 avg nodes; metric = AP | *stretch* — long-range/over-squashing probe |
| ZINC (subset) | PyG | graph regression | 12,000 graphs, ~23 avg nodes; metric = MAE | *stretch* — graph-level extension |

Confirm all stats at `load` time. **Cora is the workhorse; everything else is optional.**

---

## 6. METHODS, ATTACKS, METRICS (concrete)

**Backbones:** GCN (primary). GAT, GraphSAGE = stretch (IV3).

**Attack:** **Metattack** (global poisoning meta-attack; standard for *robust accuracy* benchmarks),
ptb_rate ∈ {0.05, 0.10} core, {0.20} stretch. Nettack (targeted) = stretch. Both in DeepRobust.

**Defense reference:** **GCN-Jaccard** (in DeepRobust) — prunes edges between dissimilar nodes.

**Rewiring:**
- **FoSR** (add) — reference code at `github.com/kedar2/FoSR` (but written for *graph* classification;
  must be adapted to node tasks — extract the edge-adding/spectral-gap routine and apply to Cora).
- **Spectral pruning (Jamadandi, "ProxyDelete" / `proxydelmin`)** (remove) — primary remove-arm.
  **Code confirmed usable (28 Jun):** `github.com/RelationalML/SpectralPruningBraess`,
  `NodeClassification/rewiring/{MinGapKupdates,spectral_utils}.py`. Deletes edges to maximize the
  spectral gap; pure networkx/scipy/numpy (no torch/DGL/CUDA), operates on a NetworkX graph so it
  accepts a poisoned Cora adjacency directly. Extract `min_and_update_edges(...)` + a thin
  `prune_spectral(adj, num_delete) -> adj` wrapper; drop their NMI/community return bookkeeping.
- **DropEdge** (optional sanity baseline, *not* the fallback) — random edge dropping, ~10 lines. Now
  used to contrast *principled spectral removal (ProxyDelete) vs random removal (DropEdge)*.

**Metrics:**
- **Clean accuracy** (no perturbation)
- **Robust accuracy** (under Metattack)
- **Robustness gap = clean − robust** ← **headline number**
- **Over-squashing measure (Jacobian sensitivity)** — ∂(node v embedding)/∂(node u input) for distant
  (u,v) pairs, via autograd. **Scoped version first** (sample a few far pairs, get the concept), then
  expand. This is the core *learning* artifact — see Di Giovanni et al. 2023 for the exact definition
  before coding.

---

## 7. DECISIONS ALREADY MADE (do not re-litigate)

1. **Framing = learning project**, lens = "add vs remove edge". No novelty claims.
2. **Lead with RQ2.** RQ1 is a near-free add-on after the probe; RQ3 is stretch.
3. **Run on CPU** for Cora/Citeseer. They're tiny; this sidesteps the Blackwell GPU/CUDA issues.
4. **Pin DeepRobust to 0.2.9**; install and smoke-test in a clean env before relying on it.
5. **Remove-arm = Jamadandi ProxyDelete** (code confirmed usable, 28 Jun). DropEdge demoted to an
   optional random-removal sanity baseline.
6. **Jacobian probe = scoped first**, expand later. Don't sink the whole sprint into full theory.
7. For FoSR on node tasks, a reasonable simplification is to run **plain GCN on the rewired graph**
   (FoSR's paper pairs it with a relational R-GCN to curb over-smoothing; that's optional for a
   learning project — note the simplification in the writeup).

---

## 8. ENVIRONMENT & TOOLING NOTES

- **Hardware:** Mac M4 Pro; Windows RTX 5070 (Blackwell, sm_120) + WSL2 + CUDA 12.8.
- **GPU caveat:** RTX 50-series needs cu128 wheels (PyTorch ≥2.7). DeepRobust (last release 0.2.9,
  Nov 2023) may not play nicely with bleeding-edge PyTorch/PyG. **→ Run CPU for the small graphs;
  only fight the GPU stack if scaling up later.**
- **DeepRobust:** `github.com/DSE-MSU/DeepRobust`, `pip install deeprobust==0.2.9`. Has Metattack,
  Nettack, PRBCD, GCN-Jaccard with Cora examples. Verified fix for Metattack OOM on newer PyTorch.
  If deps are painful, `setup_empty.py` installs DeepRobust without forcing its dependency versions.
- **PyG (PyTorch Geometric):** datasets (Planetoid → Cora/Citeseer), layers (GCNConv, GATConv,
  SAGEConv), converts to/from DeepRobust format.
- **FoSR:** `github.com/kedar2/FoSR` (conda `environment.yml`, runner is graph-classification).

---

## 9. PROPOSED REPO STRUCTURE

```
gnn-oversquash-robustness/
├── README.md
├── PROJECT_BRIEF.md            # this file (can also be referenced from CLAUDE.md)
├── requirements.txt
├── src/
│   ├── data.py                 # Planetoid loaders (Cora/Citeseer) + perturbation hooks
│   ├── models.py               # GCN (primary), GAT, GraphSAGE
│   ├── rewiring/
│   │   ├── fosr.py             # adapted FoSR (add edges) for node tasks
│   │   ├── pruning.py          # Jamadandi spectral pruning (remove) — TBD
│   │   └── dropedge.py         # fallback remove-arm
│   ├── attacks.py              # DeepRobust wrappers: Metattack (+ Nettack)
│   ├── defenses.py             # GCN-Jaccard wrapper (DeepRobust)
│   ├── probe.py                # Jacobian over-squashing sensitivity (scoped → full)
│   ├── metrics.py              # clean/robust accuracy, robustness gap
│   ├── train.py
│   └── eval.py
├── scripts/
│   └── run_cora_baseline.py    # Task 0: GCN on Cora
├── configs/                    # one yaml per experiment setting
├── results/                    # tables, logs, plots
└── notebooks/                  # exploration
```

---

## 10. KICKOFF PLAN (task order + day map)

**Build order — do not jump ahead; each task de-risks the next:**

- [x] **Task 0 — GCN on Cora (DONE 28 Jun).** PyG Planetoid → GCN → **80.30 ± 0.47** test acc (CPU,
      5 seeds). Baseline + clean-accuracy number for the protocol secured. Code: `src/`,
      `scripts/run_cora_baseline.py`; result in `results/cora_baseline.json`.
- [ ] **Task 1 — DeepRobust spike (gating risk).** In a clean env, run Metattack (ptb 0.05) **and**
      GCN-Jaccard on Cora, end-to-end, once. If GPU breaks → switch to CPU immediately. **Must pass
      before building the grid.**
- [ ] **Task 2 — Core pipeline + headline table.** GCN + DropEdge + GCN-Jaccard, conditions
      {clean, adv 0.05, adv 0.10} → table of clean / robust / gap. **This table is the July-7
      deliverable backbone.**
- [ ] **Task 3 — Adapt FoSR to node tasks** (the "add" arm).
- [ ] **Task 4 — Jacobian probe (scoped)** via autograd on a few far node-pairs.
- [ ] **Task 5 — Add Citeseer** (cheap swap) + check Jamadandi code; integrate if practical.
- [ ] **Task 6 — Slides + Summary of Progress.**

**Day map:**

| Date | Focus | Gate |
|---|---|---|
| Sun 28 Jun | Task 0 (GCN on Cora) | |
| Mon 29 Jun | Task 1 (DeepRobust spike) + start writing protocol | |
| Tue 30 Jun | Finish & submit Phase 2 protocol (07:45) | **Gate 1** |
| Wed 1 – Thu 2 Jul | Task 2 (core table) | |
| Fri 3 Jul | Task 3 (FoSR adapt) | |
| Sat 4 Jul | Task 4 (Jacobian scoped) | |
| Sun 5 Jul | Task 5 (Citeseer + Jamadandi) | |
| Mon 6 Jul | Task 6 (slides + progress) | |
| Tue 7 Jul | Present | **Gate 2** |

**What's blocked on what:**
- *Do now (nothing blocks):* Task 0, write protocol (it's a plan), DropEdge, read Di Giovanni for the
  Jacobian definition.
- *Spike before trusting:* DeepRobust install (Task 1), FoSR node-adapt, Jacobian probe.
- *Waits on others:* Jamadandi (needs code check / after FoSR), Citeseer/Peptides (after Cora
  pipeline), full RQ1/RQ3 (after probe).

---

## 11. OPEN QUESTIONS / TO VERIFY (before/while building)

1. ~~**Jamadandi spectral-pruning code**~~ — **RESOLVED (28 Jun): usable.**
   `github.com/RelationalML/SpectralPruningBraess`, ProxyDelete (`proxydelmin`) in
   `NodeClassification/rewiring/`. Pure networkx/scipy, CPU-ok, runs on Cora node classification.
   Remove-arm is now principled spectral pruning, not DropEdge.
2. **Jacobian over-squashing measure** — pull the exact definition from **Di Giovanni et al. 2023**
   (arXiv 2302.02941) before coding `probe.py`. Don't guess the formula.
3. **Attack choice** — Metattack (global poisoning) is the default for robust-accuracy benchmarks;
   confirm this matches what we want to report.
4. **Dataset stats** — **Cora confirmed at load (28 Jun):** 2708 nodes, **5278 undirected edges**
   (10556 directed), 7 classes, 1433 feat, split 140/500/1000. *Note: §5 cites 5,429 (commonly-quoted
   raw figure); PyG's processed Planetoid Cora yields 5,278 undirected — use 5,278 in the writeup.*
   Citeseer to confirm when added.

---

## 12. KNOWN RISKS

- **DeepRobust × Blackwell/CUDA** (highest) → mitigated by CPU + clean-env smoke test on Day 1.
- **FoSR repo is graph-classification** → must extract and adapt the rewiring routine to node tasks.
- **Jacobian probe is the hardest code** → scope it small first; it can be the limiting factor.

---

## 13. GLOSSARY (use these terms correctly)

- **MPNN / message passing** — GNN framework: each layer, every node aggregates neighbor vectors and
  updates its own. GCN/GAT/GraphSAGE are instances.
- **Over-squashing** — long-range info compressed through bottlenecks into a fixed-size vector and lost.
- **Over-smoothing** — different problem: too many layers make all node vectors converge/indistinguishable.
- **Bottleneck** — a narrow structural point all information must funnel through.
- **Rewiring** — editing graph edges (add/remove) to change information flow.
- **Spectral gap** — scalar measuring how well-connected / fast-mixing a graph is. Higher = fewer
  bottlenecks. Both FoSR (add) and Jamadandi (remove) aim to raise it — this is why the add/remove
  axis is coherent.
- **Ricci curvature** — per-edge geometric criterion; negatively curved edges are bottlenecks (used by SDRF).
- **Jacobian sensitivity** — ∂(node v embedding)/∂(node u input); ≈0 for distant u signals over-squashing.
- **Inductive bias** — built-in assumption; GNNs assume graph distance is meaningful. Transformers drop this.

---

## 14. REFERENCES

- Alon & Yahav, *On the Bottleneck of GNNs* (over-squashing named), ICLR 2021 — arXiv 2006.05205
- Topping et al., *Understanding Over-Squashing… via Curvature* (SDRF), ICLR 2022 — arXiv 2111.14522
- Di Giovanni et al., *On Over-Squashing in MPNNs: Width, Depth, Topology* (Jacobian sensitivity),
  ICML 2023 — arXiv 2302.02941  ← **read for the probe definition**
- Karhadkar et al., *FoSR: First-Order Spectral Rewiring*, ICLR 2023 — arXiv 2210.11790;
  code `github.com/kedar2/FoSR`
- Jamadandi et al., *Spectral Graph Pruning Against Over-Squashing and Over-Smoothing*, NeurIPS 2024
  — arXiv 2404.04612
- Dwivedi et al., *Long Range Graph Benchmark (LRGB)*, NeurIPS 2022 — arXiv 2206.08164
- Zügner & Günnemann, *Metattack* (meta-learning poisoning), ICLR 2019
- Zügner et al., *Nettack* (targeted attack), KDD 2018
- Wu et al., *GCN-Jaccard* (defense via Jaccard edge pruning), IJCAI 2019
- Jin et al., *Pro-GNN* (graph structure learning defense), KDD 2020
- Zhang & Zitnik, *GNNGuard* (defense), NeurIPS 2020
- Li et al., *DeepRobust* (library), AAAI 2021 — `github.com/DSE-MSU/DeepRobust` (v0.2.9)
- *Adversarial Robustness of Graph Transformers* — arXiv 2407.11764 ← positioning read (covers the
  transformer side of the bridge)
- Position paper on over-smoothing/over-squashing (Jan 2026) — arXiv 2601.07419 ← positioning read

---

## 15. TEAM & CONTEXT

- Supanut Kompayak (st126055), Dechathon Niamsa-Ard (st126235).
- Deep Learning course project, AIT.
- Reminder: prior version of this project had an integrity issue (AI-generated PDFs with fabricated
  results). This build is clean from scratch. **All reported numbers must be real and reproducible.**
