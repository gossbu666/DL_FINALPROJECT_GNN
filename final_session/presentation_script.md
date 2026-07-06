# Presentation script — ~10 min, 12 slides
*Over-Squashing vs. Adversarial Robustness in GNNs: An "Add vs. Remove Edge" Investigation*

Delivery notes: speak in short sentences, pause on the **bold** lines, point at the figure when one is on screen.

---

## Slide 1 — Title (~20s)
"Good morning. I'm Supanut, and this is Dechathon. Our project is titled *Over-Squashing versus Adversarial Robustness in Graph Neural Networks — an Add-versus-Remove-Edge Investigation.* In one sentence: we ask whether a method that fixes one known GNN problem accidentally fixes — or hurts — a completely different one."

## Slide 2 — Two structural failure modes (~60s)
"Graph neural networks classify a node by passing information along the edges of the graph. Two well-known problems both come from that graph structure.
On the left is **over-squashing**: to reach a distant node, information has to travel through many hops, and it gets squeezed through bottlenecks and lost. So the node is *not sensitive enough* to far-away signal.
On the right is **adversarial fragility**: an attacker adds or removes just a few edges, and the prediction flips. So the node is *too sensitive* to malicious structural change.
The key observation — bottom of the slide — is that these are two sides of the *same* thing: **how sensitive a node's output is to structural change elsewhere in the graph.** That shared mechanism is what motivates the whole project."

## Slide 3 — The methods manipulate structure (~55s)
"Now, the methods people use for these two problems all edit edges — and that's the connection.
To fix over-squashing you rewire the graph: FoSR **adds** edges, spectral pruning **removes** edges — both to raise the spectral gap.
On the robustness side, the attack **adds** dissimilar edges, and the classic defense, GCN-Jaccard, **removes** them.
So everything lives on one axis — add versus remove. That gives us a clean **hypothesis**: maybe *adding* edges to fix over-squashing hurts robustness — because you give the attacker more edges — while *removing* edges transfers to robustness, because it behaves like a defense. That's what we set out to test."

## Slide 4 — Research questions (~45s)
"We split that into three questions.
**RQ2 is our main one:** do over-squashing fixes — add-based FoSR and remove-based pruning — actually improve robustness, compared to an undefended GCN and to a purpose-built defense, and at what cost to clean accuracy?
**RQ1**, secondary: does the attack itself make over-squashing worse?
And **RQ3**, a stretch: can a single measure capture both problems at once?
We lead with RQ2; the other two come almost for free from the same probe."

## Slide 5 — Methods (the design) (~60s)
"Here's the design, and it's really a controlled ablation. Each method sits on the add/remove axis.
GCN is the plain baseline. FoSR is the *add* arm, ProxyDelete is the *remove* arm — both target over-squashing. GCN-Jaccard and GCN-SVD are two purpose-built *defenses*. And DropEdge, which removes edges at random, is a sanity control.
The point is what each comparison isolates: **FoSR versus ProxyDelete changes only the direction** — add versus remove. **ProxyDelete versus the defenses changes only the criterion** — which edges you remove. So every pair turns exactly one knob."

## Slide 6 — Setup (~40s)
"The setup: two standard citation graphs, Cora and Citeseer, using the standard attack split. The attack is Metattack, a poisoning attack, at two strengths. The pipeline is: take the clean graph, poison it, apply the method, train a GCN, evaluate. We generate the attack once and cache it, so every method sees the exact same poisoned graph. And our headline metric is the **robustness gap** — clean accuracy minus robust accuracy — smaller is better."

## Slide 7 — Results (~70s)  ⭐ key slide
"Here's the main result on Cora. Each pair of bars is one method — robust accuracy at the two attack strengths. The dotted line at the top is clean accuracy, 83.5%.
Look at the four grey bars — the undefended GCN, random removal, FoSR, and ProxyDelete. They all sit at the *same* level, around 77 and 72 percent. The over-squashing methods do nothing for robustness.
Now the two coloured bars — Jaccard in red, SVD in orange — the purpose-built defenses. **They're the only ones that stand up.** Jaccard is best at the low attack strength; SVD is the most attack-immune at the high one.
So the first big message: **the add/remove direction does not decide robustness. Only the two real defenses separate from the pack.**"

## Slide 8 — Why does Jaccard win? (~70s)  ⭐ key slide
"So *why* do the defenses win? We looked directly at the edges.
First — Metattack is almost purely edge-*addition*: at the high strength it adds 503 edges and removes only 3. So defense means: can you delete the edges the attacker added?
The table answers that. 'Recall' is the fraction of the attacker's edges that a method deletes. **Jaccard deletes about half — 0.50 — roughly four times chance.** It literally undoes the attack.
ProxyDelete, the spectral method, is at 0.19 — barely above the chance line of 0.12. Random DropEdge is exactly at chance. And FoSR removes none, because it only adds.
So this is the mechanism, and it's the heart of the project: **robustness comes from deleting the attacker's specific edges — and only the similarity criterion does that.**"

## Slide 9 — Over-squashing probe, RQ1 (~45s)
"This slide answers RQ1. We measure over-squashing directly with a Jacobian sensitivity — how much a node's output depends on a node d hops away.
The blue line drops about five-fold per hop — that's over-squashing, and it confirms the phenomenon is real.
The orange line is the same measurement on the poisoned graph. Notice the two lines are *parallel* — poisoning shifts the level but does **not** change the decay rate. So the attack does not actually make over-squashing worse. The two problems are decoupled — which is completely consistent with our main result."

## Slide 10 — Adaptive attacker (~50s)
"A fair question: our attacker didn't know about the defense. What if it does?
So we built a defense-aware attacker: it adds edges between *similar* nodes, which Jaccard cannot remove. Two things happen. One — the evasion works: Jaccard removes zero percent of those edges. But two — the attack becomes weak: it drops accuracy by less than two points, versus six to twelve for the real attack.
And that's the interesting part: **the edges that actually fool the model are the dissimilar ones — which are exactly the ones Jaccard removes.** Attack strength and detectability are coupled. This is a heuristic attack, so a fully optimized adaptive attack is future work."

## Slide 11 — Findings (ablation summary) (~55s)
"Here's the whole study on one slide — every knob we turned and what it taught us.
Vary the *direction* — add versus remove — both fail: direction is not what matters.
Vary the *criterion* — only the attack-aware defenses work: **the criterion is the key.**
Give the spectral method a bigger budget — it still fails, so it's not a budget effect. Switch datasets — same ranking, it generalizes. Make the attacker adaptive — it evades but turns weak. And the over-squashing measure itself is unchanged by the attack — the two phenomena are decoupled.
Every row points the same way."

## Slide 12 — Conclusion (~50s)
"To conclude. Over-squashing mitigations — whether they add or remove edges — do **not** improve adversarial robustness. Only attack-aware defenses do, and robustness comes from targeting the attacker's edges, not from optimizing the spectral gap.
This is an honest negative result: on this testbed, the bridge from over-squashing to robustness is not supported. And it's robust — it holds at a larger budget, it replicates on a second dataset, and it survives a defense-aware attacker.
As a learning project, the contribution is the controlled question and its answer. Thank you — we're happy to take questions."

---
### Timing summary (~10 min)
20 + 60 + 55 + 45 + 60 + 40 + 70 + 70 + 45 + 50 + 55 + 50 ≈ **620 s ≈ 10.3 min**

### Three things to remember if you blank out
1. **Only the two defenses (Jaccard, SVD) improve robustness — the over-squashing methods don't.**
2. **Why: Jaccard deletes ~half the attacker's edges; the others are at chance.**
3. **So the *criterion* (which edges you edit), not the add/remove *direction*, decides robustness.**
