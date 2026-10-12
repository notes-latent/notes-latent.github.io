---
layout: post
math: true
title: "From Semantic IDs to Intent-Aware Retrieval Sets"
description: "A personal research reflection on multi-intent planning, independent SID sampling, and budget-aware retrieval set utility."
---

*Why I stopped treating SID prediction as the entire retrieval problem—and started thinking about intent descriptions, independent SID sampling, and set-level optimization.*

**Status:** Research reflection · Proposed framework, not experimentally validated  
**Research thread:** Semantic IDs → user-interest representations → multi-positive training → retrieval coverage → intent-conditioned set sampling.

> **Core thesis.** A Semantic ID (SID) is a useful *retrieval unit*, but the system ultimately delivers a **set of candidates** under a finite budget. Predicting a good SID, or even raising the likelihood of several positive SIDs, does not necessarily produce a good *set*. I want to explicitly model the user's plausible intents, independently sample SIDs conditioned on a compact **multi-intent description**, and evaluate the resulting set for relevance, unique-intent coverage, and budget allocation. Set-level policy optimization is a possible final step—not the prerequisite.

## 1. How the papers connect: from representation to retrieval utility

I began by asking **why use SIDs at all**, given that dense embeddings already express semantic similarity. Several papers suggest that SID is a reusable interface rather than merely a unique item identifier. But their contributions operate at *different* levels:

| Work | What its use of SID or tokens establishes | What it leaves open for my question |
|---|---|---|
| [TIGER (2023)](https://arxiv.org/abs/2305.05065) | Item embeddings are quantized into hierarchical SIDs; autoregressive models can generate them for retrieval. | One-item identification is not always necessary for retrieval; compare against strong content-aware dense ANN. |
| [Better Generalization with Semantic IDs (2023/2024)](https://arxiv.org/abs/2306.08121) and [PLUM (2025)](https://arxiv.org/abs/2510.07784) | SID pieces can support ranking features; domain/task adaptation can make pretrained models useful for recommendation. | Representation gains, extra supervision, fine-tuning, and SID-specific gains should be disentangled. |
| [STATIC (2026)](https://arxiv.org/abs/2602.22647) and [Beyond Item IDs (2026)](https://arxiv.org/abs/2606.07546) | The SID interface can support constrained decoding and efficient long-history modeling. | Lower serving cost does not by itself establish better multi-intent coverage. |
| [TokenMinds (2026)](https://arxiv.org/abs/2606.25147) | Multi-target future-interest supervision and reusable user tokens/embeddings give a concrete reason to predict *more than the immediate next action*. | Future-window user representation is not the same as query-conditioned retrieval; the output set still needs direct evaluation. |
| [Alibaba CQ-SID + Progressive SFT (2026)](https://arxiv.org/abs/2605.14434) | A cluster-level SID can retrieve several items; a 0.5B LM is progressively trained on Item2SID → Query2SID → User+Query2SID; an optional EG-GRPO stage follows. | It lacks a direct progressive-versus-direct SFT ablation, a matched strong dense-retrieval baseline, and an explicit relevant-intent-coverage metric. |
| [Google Token Factory (2026)](https://arxiv.org/abs/2606.19635) | **Adjacent, not equivalent to SID generation:** compress heterogeneous inputs into soft tokens. | Could help shorten user-history input, but will not automatically reduce the cost of producing many SID candidates. |
| [Kuaishou Once Generated, Ranked / OGR (2026)](https://arxiv.org/abs/2608.17613) | Treats an **ordered recommendation slate** as the object of planning and preference alignment, including diversity and novelty signals. | Its logged-slate alignment is not the same as online sampling of unordered, query-conditioned retrieval sets; it does not establish our proposed two-level intent/SID sampling. |

**My synthesis:** The recurring value of SIDs is not proof that quantization is intrinsically better than dense retrieval. It is that a shared discrete vocabulary can connect item semantics, user behavior, generative prediction, and systems optimizations. My open problem lies *above* the ID representation: **what should the retrieved SID set cover, and how much budget should each covered intent receive?**

## 2. The missing objective: a good SID is not necessarily a good SID set

Most next-SID or multi-positive SFT objectives reward *individual* targets. In practice, retrieval must provide a finite collection of candidates to downstream ranking. The marginal benefit of one more candidate depends on what has *already been covered*, even if SID generation itself need not be sequential.

Take a query for running shoes and three relevant intent groups: road, trail, and racing. Assume each returned SID is individually relevant. At a budget of three clusters:

- **Set A:** three road-running clusters → 3 individually relevant clusters, **1 distinct intent**.
- **Set B:** one road, one trail, one racing cluster → 3 individually relevant clusters, **3 distinct intents**.

An additive relevance score may treat A and B equally. A relevance-aware set objective should prefer B *if* the user plausibly has all three intents. Importantly, it should **not** reward unrelated categories just to increase diversity.

One simple intent-aware utility is:

```math
C(S\mid x)=\sum_{m=1}^{M} w_m(x)\left[1-\prod_{s\in \mathrm{unique}(S)}(1-c_m(s,x))\right],
```

where `x=(q,H_u)` is the current query and visible user history, `w_m` is the estimated importance of intent `m`, and `c_m(s,x)\in[0,1]` measures how well SID `s` covers that intent. The product introduces **diminishing returns**, with exact duplicates counted only once: ten near-identical clusters should not be worth ten independent intent discoveries. This is a *proposed* metric, not one already validated by these papers.

If the intent weights in the example are `(0.5,0.3,0.2)`, a perfect binary cluster-to-intent mapping gives Set A a coverage score of **0.5** and Set B **1.0**, despite identical counts of individually relevant SIDs. That is exactly the distinction I want to quantify.

There is also a sampling argument. Suppose a single-SID policy assigns probabilities `p` and `1-p` to two useful intents and independently samples two SIDs. The probability of covering *both* is `2p(1-p)`: **18% at `p=0.9`** versus **50% at `p=0.5`**. Improving single-SID likelihood for the dominant mode can therefore be misaligned with coverage of plausible alternative modes. This is not a defect of probability estimation; it is a **different downstream utility**.

**Why the set matters:** relevance, redundancy, intent coverage, long-tail access, and candidate-budget efficiency are properties of the *joint output*, not merely sums of isolated SID scores.

## 3. My proposed framework: describe intents, then independently sample SIDs

I do **not** want an autoregressive model that generates item 1, conditions item 2 on item 1, then generates item 3. That would impose an unnecessary item order on a retrieval set and increase sequential decoding work. **Each inner rollout should still generate exactly one complete SID.** Only the tokens *within* that SID are autoregressive.

The new element is a compact **multi-intent description** (or structured intent plan) sampled *before* SID generation. This description should express several plausible interpretations of the same query and user context—not a single intent per rollout and not a long natural-language chain of thought.

```text
Observed condition x = current query + user-history context
                          |
          Outer sampling: G plausible multi-intent plans z_g
                          |
       For each plan z_g: independently sample K complete SIDs
          one rollout -> one SID; no SID-to-SID dependency
                          |
          Unordered sampled SID set S_g = {s_g1 ... s_gK}
                          |
          SID clusters -> item candidates + allocation ratios
                          |
       Score the *whole set* on relevance, intent coverage,
                    redundancy, and fixed-budget utility
                          |
       Optional: compare multiple set rewards and optimize policy
```

More formally:

```math
z_g\sim p_\phi(z\mid q,H_u),\qquad
s_{g,k}\overset{\mathrm{iid}}\sim p_\theta(s\mid q,H_u,z_g),\qquad
S_g=\{s_{g,1},\ldots,s_{g,K}\}.
```

The repeated SID samples are conditionally independent **given the same intent description**. That is deliberate: shared `z_g` can coordinate the *distribution* of interests without sequentially conditioning one item on another. Duplicate draws are possible and must be handled and measured.

### Why an explicit intent description?

Pure independent sampling from `p(SID\mid q,H_u)` can produce different sets, but the difference may be sampling noise or a popular-intent collapse. Sampling a *structured, multi-intent hypothesis* gives each group a coherent explanation of which interests are relevant and how heavily to cover them. Two outer samples can represent genuinely different but defensible interpretations of ambiguous behavior. This also makes failures auditable: was the intent interpretation wrong, or did SID sampling fail under a reasonable interpretation?

A short plan might read: `road running: 0.5; trail: 0.3; racing: 0.2`. These are **hypothetical importance weights**, not observed ground truth. If verbalized descriptions prove slow or unfaithful, the same abstraction could be learned as compact intent tokens or a small planner head; interpretability and added computation are testable trade-offs.

### Importance is not the same as allocation

For each selected SID `s`, a lightweight mapping can predict a context-dependent allocation weight `a(s\mid x,z)`, normalized across the distinct sampled SIDs. Under a total item budget `N`, these weights determine how many items to retrieve from each SID cluster, subject to cluster size, duplicate elimination, and quality constraints.

This separates three ideas that are often conflated:

1. **Presence:** Is an intent/SID plausible at all?
2. **Importance:** How much user value does this intent represent?
3. **Marginal allocation:** How many *additional unique, relevant items* are worth retrieving from this cluster given what other clusters already supply?

High SID sampling frequency need not imply the same fraction of the final item budget. The allocation head should ideally learn **marginal set utility**, not just mimic historical clicks or a raw SID probability.

## 4. Training: multi-positive SFT first, optional nested set-level RL later

**Stage A — Offline representation.** Construct item-to-cluster SIDs using an existing competent quantizer; keep the SID vocabulary stable during the initial LM training. Unique per-item IDs are not required for retrieval. Test clustering quality rather than assuming category-aware quantization is always superior.

**Stage B — Query-conditioned multi-positive SFT.** Train on positives associated with `q+H_u`, including *future relevant interactions within a carefully defined query/session window*. The horizon must reflect the current search task; a future click on an unrelated topic must not become a positive for today's query. Crucially, future behavior is used **only as an offline label**, never as online input. Compare direct personalized SFT with the Item2SID → Query2SID → Personalized curriculum rather than assuming progressive training is needed.

**Stage C — Intent-conditioned sets.** Learn a compact multi-intent plan `z`, then independently sample SIDs within that plan. Initially measure whether intent descriptions and smarter allocations improve the set **without any RL**. Otherwise, RL would obscure the simpler contribution.

**Stage D — Set-level group-relative optimization (optional).** For the *same* `x`, sample multiple outer plans and associated SID sets. Define a set reward, for example:

```math
R(S,x)=\alpha\,\mathrm{RelevantRecall@N}
+\beta\,\mathrm{WeightedIntentCoverage@N}
-\gamma\,\mathrm{Redundancy@N}
-\delta\,\mathrm{CostPenalty}.
```

Normalize rewards **between complete sets**, not between individual SIDs. For one outer sampled plan and its `K` independent SID draws, an ordered-rollout representation has log probability:

```math
\log p(z,s_{1:K}\mid x)=\log p_\phi(z\mid x)+\sum_{k=1}^{K}\log p_\theta(s_k\mid x,z).
```

This supports a set-level policy-gradient construction in principle. It does **not** automatically solve credit assignment: one good set may contain redundant members, and a high score may reflect lucky inner sampling rather than a better intent plan. A useful diagnostic is to sample **multiple independent SID sets per fixed plan** and compare the plans' *mean* rewards, though this adds training cost. An optional marginal-contribution signal can refine credit assignment later. **Step-level partial-set rewards are neither required nor claimed here.**

Unlike Kuaishou OGR's ordered slate and logged-exposure preference calibration, this idea targets **unordered, query-conditioned recall sets**. Unexposed sampled sets do not have real observed engagement; initial reward therefore must use valid offline future positives, independent relevance judgments, or a carefully validated reward model—not invented counterfactual click labels.

## 5. The decisive tests: quantify set value before claiming a framework win

| Question | Minimum experiment | What would count as meaningful evidence |
|---|---|---|
| Does the *set* objective matter? | Compare equal-size sets with matched individual relevance but different covered intents; report marginal utility as clusters are added. | Higher unique relevant-intent coverage at the **same item-candidate budget and relevance floor**. |
| Is intent description genuinely useful? | Single-SID sampling vs. compact intent-conditioned sampling; include empty, shuffled, and randomly sampled intent-plan controls. | A coverage lift attributable to **informative** plans, not extra tokens or sampling entropy. |
| Is allocation independently useful? | Equal quota vs. SID-frequency quota vs. learned allocation vs. oracle/greedy marginal-coverage allocation. | Better item Recall@N / intent coverage **after SID-to-item expansion**, not merely more distinct SID codes. |
| Does RL add anything beyond SFT? | Multi-positive SFT; + independent sampling; + intent plan; + allocation; finally + set-reward GRPO. | Incremental set utility after the strong no-RL baseline; stable training across seeds. |
| Is SID necessary? | Compare with a strong query+history **dense ANN retriever** trained on the same positives, and matched total candidates, latency, and training resources. | Evidence that structured generation earns its systems and modeling complexity. |
| Can a small LM serve it? | Vary LM size, number of outer plans, SID samples, plan-token count, and history compression. Record P50/P95 latency, QPS, GPU cost, and failures. | A useful relevance/coverage–cost frontier, not just a reported average. |

**Primary evaluation unit:** the **expanded item candidate set under a fixed budget `N`**; SID-level metrics are diagnostics. Report relevant Intent Coverage@N, weighted future-positive Recall@N, duplicate rate, marginal coverage gain, popularity-stratified recall, and cost. Use α-nDCG only where item order is actually evaluated. Simply increasing category diversity is insufficient: it can reward irrelevant candidates. Segment analyses by query ambiguity and the number/entropy of validated future intents. A narrow, unambiguous query may correctly need only one intent.

## 6. Evidence ledger and retrospective checkpoint

**Supported by published work:** SIDs can be used for generative retrieval and production user modeling; Alibaba demonstrates a practical cluster-level search-recall channel and progressive training pipeline; TokenMinds gives evidence for multiple future targets; OGR demonstrates the relevance of whole-slate objectives for **ordered recommendation**. [TIGER](https://arxiv.org/abs/2305.05065), [CQ-SID](https://arxiv.org/abs/2605.14434), [TokenMinds](https://arxiv.org/abs/2606.25147), [OGR](https://arxiv.org/abs/2608.17613).

**My interpretation, not proven by these papers:** Cluster-level SID construction and a new identifier scheme are not, by themselves, the most interesting research contribution for retrieval. The more compelling question is whether **explicit multi-intent planning plus budget-aware set utility** leads to better coverage than independently generated high-likelihood SIDs or competitive dense retrievers. Alibaba's EG-GRPO gain over strong SFT is small, and its Click/Exposure metrics do not directly validate relevant intent diversity. Its published user-history prompt and history length are also underspecified. Progressive SFT is worth trying, but its advantage over direct personalized SFT is still unestablished.

**Open risks / TBD:** how to obtain trustworthy query-conditioned future-intent labels; whether intent descriptions are identifiable, stable, and not just differently worded plans; whether cluster quantization hides valuable item attributes; duplicate/invalid SID handling; how to learn allocation without overfitting exposure bias; planner-vs-inner-sampler credit assignment; reproducibility and serving costs.

**Why this felt like a resolution after several days of reading:** I was separately questioning (i) SID uniqueness and quantization, (ii) why multi-positive likelihood did not guarantee diverse Top-K results, (iii) why pointwise GRPO added so little, (iv) how to use a retrieval budget across heterogeneous interests, and (v) whether a sequence model needed to emit an *ordered list* of items. The unifying answer is to **separate the unit of generation from the unit of utility**. Generate **one SID at a time, independently within a shared intent plan**; evaluate and allocate resources at the **set** level. The added intent-description layer makes the competing sets structurally meaningful. This is a coherent hypothesis—not yet evidence that the proposed implementation will outperform simpler sampling, diversification, or dense baselines.

### Six-month revisit checklist

- Can I show that the same individual-SID quality yields measurably different **set-level intent coverage**?
- Does a *real* intent plan beat a shuffled plan, with the **same K, N, and latency**?
- Does learned allocation beat equal quotas and a simple greedy coverage heuristic?
- Is set-level RL still useful after multi-positive SFT + intent-conditioned sampling?
- Does the improvement hold over a strong dense retrieval baseline, especially on ambiguous queries?
- Can this be served without a long reasoning trace or expensive multi-set online decoding?

**Decision as of 2026-10-11:** Follow up. Start with the **smallest falsifiable no-RL prototype**; add hierarchical set-level GRPO only if a measurable residual set-coverage gap remains.
