Title of Project
-Over-Squashing vs. Adversarial Robustness in GNNs: An "Add vs. Remove Edge" Investigation
What good at
"- Sharpest problem framing in the batch: unifies over-squashing and adversarial fragility as one ""sensitivity to structural change"" question via an add-vs-remove-edge axis.
 - Genuinely rigorous experimental design: methods placed on the add/remove axis, pairwise comparisons each isolate one factor, correct poisoning pipeline order, clean/poisoned x method conditions to separate clean-accuracy cost.
 - Honest scientific posture: reports only real numbers, accepts negative results, states the non-adaptive-attacker limitation."

Novelty check
Contribution lives in the research question, not in any single method (FoSR, spectral pruning, GCN-Jaccard, Metattack all exist). Whether over-squashing mitigation transfers to adversarial robustness is still flagged as open in recent surveys, so an "add-vs-remove-edge" bridge study between the two subfields is a real and worthwhile contribution. Strengthen it by finishing the full grid so the question actually gets answered, and by positioning clearly against those surveys.

20260702 Comment (P2) based on progress's submission
1. Strongest conceptual proposal - keep the framing and the honest reporting.
 2. Execution risk is the main concern: the headline 4-method x perturbation grid (RQ2) is still to-do; only smoke-test numbers exist. Confirm the full tuned grid with >=3 seeds can be finished before the deadline.
 3. Clarify the two baselines: Planetoid-split GCN (80.30) vs DeepRobust attack-split GCN (84.05) must be comparable within one table before contrasting clean vs robust.
 4. The Jacobian over-squashing probe (RQ1/RQ3) is the linchpin but hardest and not built. If time is tight, RQ2 alone is a complete contribution - declare RQ1/RQ3 as stretch. Keep Cora primary."
