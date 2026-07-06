# Presentation script — ~10.5 min, 13 slides
*Over-Squashing vs. Adversarial Robustness in GNNs: An "Add vs. Remove Edge" Investigation*

Delivery notes: speak in short sentences, pause on the **bold** lines, point at the figure when one is on screen.

---

## Slide 1 — Title (~20s)
"Good morning. I'm Supanut, and this is Dechathon. Our project is titled *Over-Squashing versus Adversarial Robustness in Graph Neural Networks — an Add-versus-Remove-Edge Investigation.* In one sentence: we ask whether a method that fixes one known GNN problem accidentally fixes — or hurts — a completely different one."

## Slide 2 — Two structural failure modes (~60s)
"Graph neural networks classify a node by passing information along the edges of the graph. Two well-known problems both come from that graph structure. On the left is **over-squashing**: to reach a distant node, information travels through many hops and gets squeezed through bottlenecks and lost — so the node is *not sensitive enough* to far-away signal. On the right is **adversarial fragility**: an attacker adds or removes just a few edges, and the prediction flips — the node is *too sensitive* to malicious change. The key point: these are two sides of the *same* thing — **how sensitive a node's output is to structural change elsewhere.** That shared mechanism motivates the whole project."

## Slide 3 — The methods manipulate structure (~55s)
"The methods for both problems all edit edges — that's the connection. To fix over-squashing you rewire: FoSR **adds** edges, spectral pruning **removes** edges. On the robustness side, the attack **adds** dissimilar edges and the defense, GCN-Jaccard, **removes** them. So everything lives on one axis — add versus remove. That gives a clean **hypothesis**: maybe *adding* edges to fix over-squashing hurts robustness — more edges for the attacker — while *removing* transfers to robustness, because it acts like a defense. That's what we test."

## Slide 4 — Research questions (~45s)
"Three questions. **RQ2, our main one:** do over-squashing fixes improve robustness versus an undefended GCN and versus a purpose-built defense, and at what cost to clean accuracy? **RQ1**: does the attack make over-squashing worse? **RQ3**, a stretch: can one measure capture both problems? We lead with RQ2; the others come almost for free."

## Slide 5 — Related work & positioning (~30s)
"A quick word on where this sits. Everything we use already exists — the rewiring methods, the attack, and the defense. We do *not* propose a new method. What's missing is the *connection* between the two: to our knowledge, the over-squashing literature doesn't discuss adversarial robustness, and the robustness literature doesn't discuss over-squashing. So our contribution is simply the controlled study that connects them — reporting only real, reproducible numbers."

## Slide 6 — Methods (the design) (~60s)
"The design is really a controlled ablation. Each method sits on the add/remove axis. GCN is the baseline. FoSR is the *add* arm, ProxyDelete the *remove* arm — both target over-squashing. Jaccard and SVD are two real defenses. DropEdge removes edges at random — a sanity control. The point is what each pair isolates: **FoSR versus ProxyDelete changes only the direction; ProxyDelete versus the defenses changes only the criterion** — which edges you remove. Every pair turns exactly one knob."

## Slide 7 — Setup (~40s)
"Two citation graphs, Cora and Citeseer, standard attack split. The attack is Metattack, a poisoning attack, at two strengths. Pipeline: clean graph → poison it → apply the method → train a GCN → evaluate. We cache the attack so every method sees the same poisoned graph. Headline metric: the **robustness gap** — clean minus robust — smaller is better."

## Slide 8 — Results (~70s)  ⭐ key slide
"Main result on Cora. Each pair of bars is one method's robust accuracy at the two attack strengths; the dotted line is clean accuracy, 83.5%. Look at the four grey bars — GCN, random removal, FoSR, ProxyDelete — they all sit at the *same* level, about 77 and 72 percent. The over-squashing methods do nothing. Now the two coloured bars — Jaccard and SVD, the real defenses — **they're the only ones that stand up.** So the first message: **the add/remove direction does not decide robustness. Only the defenses separate.**"

## Slide 9 — Why does Jaccard win? (~70s)  ⭐ key slide
"*Why* do the defenses win? We looked at the edges directly. Metattack is almost purely edge-*addition* — at high strength it adds 503 edges and removes only 3. So defense means: can you delete what the attacker added? 'Recall' is the fraction of attacker edges a method deletes. **Jaccard deletes about half — 0.50 — four times chance.** It undoes the attack. ProxyDelete is 0.19, barely above the 0.12 chance line; random is exactly chance; FoSR removes none. This is the mechanism, the heart of the project: **robustness comes from deleting the attacker's specific edges — and only the similarity criterion does that.**"

## Slide 10 — Over-squashing probe, RQ1 (~45s)
"This slide answers RQ1. We measure over-squashing with a Jacobian sensitivity — how much a node depends on a node d hops away. The blue line drops about five-fold per hop — that's over-squashing, confirmed. The orange line is the poisoned graph — the two are *parallel*: poisoning shifts the level but not the decay rate. So the attack does not make over-squashing worse. The two problems are decoupled — consistent with our main result."

## Slide 11 — Adaptive attacker (~50s)
"A fair question: our attacker didn't know the defense. So we built one that does — it adds edges between *similar* nodes, which Jaccard can't remove. Two things: the evasion works — Jaccard removes zero percent of them. But the attack becomes weak — it drops accuracy less than two points, versus six to twelve for the real attack. The interesting part: **the edges that fool the model are the dissimilar ones — exactly the ones Jaccard removes.** Strength and detectability are coupled. This is heuristic, so an optimal adaptive attack is future work."

## Slide 12 — Findings (ablation summary) (~55s)
"The whole study on one slide — every knob and its lesson. Vary the *direction* — both fail: direction isn't what matters. Vary the *criterion* — only attack-aware defenses work: **the criterion is the key.** Bigger budget — spectral still fails, not a budget effect. Switch datasets — same ranking, it generalizes. Adaptive attacker — evades but turns weak. Over-squashing measure — unchanged by the attack, decoupled. Every row points the same way."

## Slide 13 — Conclusion (~50s)
"To conclude: over-squashing mitigations — add or remove — do **not** improve adversarial robustness. Only attack-aware defenses do, and robustness comes from targeting the attacker's edges, not optimizing the spectral gap. An honest negative result: the over-squashing-to-robustness bridge is not supported here — and it holds at a larger budget, replicates on a second dataset, and survives a defense-aware attacker. As a learning project, the contribution is the controlled question and its answer. Thank you — we're happy to take questions."

---
### Timing summary (~10.5 min)
20 + 60 + 55 + 45 + 30 + 60 + 40 + 70 + 70 + 45 + 50 + 55 + 50 ≈ **650 s ≈ 10.8 min**
*(If you need a strict 10 min: shorten Related Work to two sentences and trim Setup — that recovers ~1 min.)*

### Three things to remember if you blank out
1. **Only the two defenses (Jaccard, SVD) improve robustness — the over-squashing methods don't.**
2. **Why: Jaccard deletes ~half the attacker's edges; the others are at chance.**
3. **So the *criterion* (which edges you edit), not the add/remove *direction*, decides robustness.**
