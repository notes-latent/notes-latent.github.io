---
layout: post
title: "Multi-Target Future Interest Modeling"
description: "A living technical note on adaptive multi-interest embeddings, with a question–evidence–experiment agenda and controlled retrieval ablations."
---

*Part of: [LLM-enhanced Recommendation — From Sequence Encoding to Grounded Generation]({{ '/notes/llm-enhanced-recommendation/' | relative_url }})*

**Status:** Living technical note · v0.3 · Personal hypotheses + evidence ledger
**Tags:** `user-modeling` `multi-positive` `multi-interest` `dense-retrieval` `semantic-ID` `listwise` `retrieval-budget`

> **Working thesis.** If our retrieval objective is to cover *multiple plausible future user interests*, a single user vector may be an unnecessary bottleneck. I want to explore a **variable-size set of interest embeddings**, with learned presence and importance, trained using stable set-level assignment and listwise retrieval objectives. Crucially, I have **not yet validated** that soft matching + listwise training outperforms hard matching.

## 1. Why I started thinking about this

I have recently been reading work on **Semantic IDs (SIDs)** and generative recommendation. The original [TIGER / *Recommender Systems with Generative Retrieval*](https://arxiv.org/abs/2305.05065) appeared in **2023**, introducing autoregressive next-item SID generation. The idea has since expanded beyond generating a single item ID: SIDs can provide hierarchical item representations, retrieval candidates, and ranking features.

A particularly interesting 2026 example is Google's [TokenMinds](https://arxiv.org/abs/2606.25147). It generates **multiple coarse-grained future-interest SID prefixes** via a decoder *and* a **dense user embedding** via an encoder. In the paper, their most detailed production evaluation is for **downstream ranking**, not a direct demonstration that the generated prefixes are themselves a complete item-retrieval solution.

Two design choices caught my attention:

1. **Future-window supervision rather than next-item-only supervision.** TokenMinds samples up to 15 watched targets in a 24-hour look-ahead window. Rather than hinging everything on the immediate next action, the model is encouraged to represent a broader range of near-future interests. My interpretation is that this may reduce sensitivity to one-off/noisy observations; it does **not** prove that labels themselves become less noisy.
2. **Amortized heavy user-model computation.** The heavy representation generation runs asynchronously with a 24-hour refresh cadence; downstream ranking consumes cached outputs. This makes larger user models practical at scale. **The look-ahead window and refresh cadence are separate decisions**—multi-target supervision does not itself reduce compute.

Together, these motivate the broader question: **Can a relatively infrequently updated user representation anticipate multiple future interests and be reused by retrieval and ranking?**

This entry focuses on **dense user representations**. I will add a separate section on **autoregressive generation of multiple Semantic IDs** later.

## 2. Two dense-representation approaches

| Approach | Representative papers | What the model emits | How multiple future positives are supervised | Main potential limitation |
|---|---|---|---|---|
| **A. Single-vector, multi-positive** | [PinnerFormer](https://arxiv.org/abs/2205.04507) (Pinterest, 2022); [PinFM](https://arxiv.org/abs/2507.12704) (Pinterest, 2025) | One user embedding for a particular serving context / prefix | Several future positives contribute contrastive or sampled-softmax loss against the **same** vector | A single vector may compromise among dissimilar future-interest modes; no explicit per-interest retrieval allocation |
| **B. Multi-vector, multi-interest** | [MIND](https://arxiv.org/abs/1904.08030), [ComiRec](https://arxiv.org/abs/2005.09347); [Pinterest DCM](https://arxiv.org/abs/2506.23060) (2025); [SetMIR](https://arxiv.org/abs/2608.30251) (Snap, 2026) | Several interest embeddings, sometimes with presence/importance | Each positive is associated with one or more interest slots; different matching strategies determine where gradients go | Assignment noise, slot collapse, unknown effective number of interests, online query/ANN budget allocation |

### A. Single embedding + multiple positives

Let `h(H)` be the current user embedding and `v_j` the embedding of future positive item `j`. A simplified objective is:

```text
\mathcal{L}_{\mathrm{multi-positive}} = \frac{1}{M}\sum_{j=1}^{M}\ell_{\mathrm{InfoNCE}}(h(H),v_j).
```

- **Why it is attractive:** simple retrieval infrastructure; reuse one ANN query vector; long-horizon supervision may produce a more stable user embedding when representations are refreshed infrequently.
- **My concern:** if future positives represent several semantically separated modes (e.g., cooking, basketball, ML), gradients need to pass through one vector. It may bias retrieval toward dominant interests, even if average Recall@K improves.
- **Important qualification:** This is a potential **capacity/geometry trade-off**, **not** proof that single-embedding models perform poorly. PinnerFormer explicitly improved on next-action baselines using future-action supervision. In PinFM, multi-token/future-token objectives supervise sequence hidden states, but do not automatically produce a set of independent interest-specific retrieval vectors.

An experiment should stratify by **future-interest diversity/entropy**. The prediction is not necessarily that multi-vector models always win; it is that their relative advantage may grow for users with genuinely multimodal future behavior.

### B. Multiple embeddings + positive assignment

Let an encoder/decoder emit `K` interest vectors `U(H)=\{u_1,\dots,u_K\}`. A common hard assignment is:

```text
k^*(j) = \arg\max_k\, u_k^\top v_j.
```

Pinterest's 2025 DCM associates positives with interest-conditioned output embeddings using an argmax operator. It also adds clustering/routing design choices to mitigate collapse; it is **not** simply an unprotected random-slot argmax model.

**My concerns:**

1. **Discontinuous gradient routing.** If two slots have nearly equal scores, a tiny parameter change can flip which slot receives a positive's direct retrieval gradient. This **does not necessarily imply a large change in loss value**, but can make assignments unstable and starve weaker slots.
2. **Unknown interest cardinality.** A user may have one salient interest or many. Always producing 30 active vectors wastes retrieval budget and can encourage duplication. The number of distinct *future items* is also not the number of distinct *future interests*.
3. **No native contribution estimate.** Even if 5 interest vectors are useful, should they receive equal ANN depth? Historical activity proportions may not equal future interest importance or marginal recall utility.

[SetMIR](https://arxiv.org/abs/2608.30251) is especially relevant: it treats the output as a set, uses learnable queries, Hungarian one-to-one matching, a presence head, and diversity regularization. Its ablations give credible evidence of argmax-based **interest collapse** in the evaluated setup. Still, Hungarian matching is discrete, and per-item matching does not guarantee that slots correspond one-to-one with true semantic interest modes.

## 3. My proposed direction — adaptive multi-interest embeddings

**Architecture hypothesis (not an existing claimed result):**

`User history → Transformer history encoder → up to 30 learnable interest queries → per-query embedding + presence + importance → active ANN queries → weighted candidate aggregation / listwise training`

The model may emit up to `K_{\max}=30` slots, but it should only use as many as are warranted for each user. For slot `k`:

```text
u_k=\mathrm{normalize}(W_e z_k),\quad p_k=\sigma(g_p(z_k)),\quad a_k=g_w(z_k).
```

- `u_k` is a retrieval vector.
- `p_k` is an *activity/presence* score used for gating.
- `a_k` is an *importance* logit used to compare active interests and help allocate retrieval budget.
- Presence and importance are **different concepts**: an interest may exist but deserve few marginal retrieval candidates.

**A potential loss:**

```text
\mathcal{L}=\mathcal{L}_{\mathrm{retrieval}}+\lambda_s\mathcal{L}_{\mathrm{soft-set}}+\lambda_l\mathcal{L}_{\mathrm{listwise}}+\lambda_p\mathcal{L}_{\mathrm{presence}}+\lambda_d\mathcal{L}_{\mathrm{diversity}}+\lambda_b\mathcal{L}_{\mathrm{budget}}.
```

These are *proposed components*, not a published joint objective:

- **Soft set assignment:** experiment with entropy-regularized / unbalanced optimal transport (possibly Sinkhorn with dustbin/inactive slots). Allow one interest group to contain multiple future items; compare against per-item assignments and Hungarian matching. Smooth matching could reduce gradient-routing discontinuities, but may also blur specialization.
- **Contrastive retrieval:** matched or softly responsible slots pull relevant future items closer and separate negatives. Negatives, sampling corrections, and shared item towers must be held fixed across ablations.
- **Listwise ranking:** score a future candidate slate with both similarity and interest importance, then optimize its positive relevance distribution. For example, a *candidate design* is `s_i=\tau\log\sum_k\alpha_k\exp(u_k^\top v_i/\tau)`, followed by listwise CE. This allows smooth contribution from multiple slots. **Listwise CE by itself does not remove hard argmax instability** if an argmax is still used earlier.
- **Presence + diversity:** inactive slots should not consume ANN queries; active slots should not all predict the same dominant interest. A diversity penalty must be limited to genuinely distinct active modes—not indiscriminately force all 30 vectors to be orthogonal.
- **Importance/budget:** test whether learned future-demand weights can be improved by a proxy for **marginal unique relevant candidates per additional ANN fetch**, accounting for overlap and a fixed global retrieval budget.

**Important refinement to my earlier intuition about “fixing each embedding's position”:** interest slots represent an *unordered set*. Fixing Slot 1 to the most important future item, Slot 2 to the next one, etc., creates an unstable arbitrary ordering if relevance ranks change. The stronger hypothesis is **permutation-aware soft matching plus smooth listwise supervision**, not rigid index-to-target matching.

## 4. Research Questions & Validation Plan

*Updated 2026-10-11. This section is both my **open-question log** and my **experiment plan**—not two separate to-do lists. I will update the evidence and decision for each question as I read more papers or run experiments.*

**North-star question:** Given the same user history and multiple plausible future engagements, can *adaptively activated interest embeddings*, trained with more stable assignment and slate-aware objectives, cover more useful future interests **at the same retrieval cost** than a strong single-vector or hard-routing baseline?

I want to test three interventions **separately** before combining them: **soft set assignment**, **listwise supervision**, and **learned retrieval allocation**. Existing work demonstrates benefits from multi-future supervision, multiple interest queries, anti-collapse matching, and query gating; it does **not** establish that my proposed combination is superior.

**Evidence standard:** *Direct* means an in-paper controlled ablation supports the narrower claim; *Adjacent* means a published finding motivates my hypothesis, but does not directly test it; *Open* means the experiment I need has not been established in the papers reviewed. Results from different papers/datasets are **not directly comparable**.

### 4.1. Research questions, evidence, and what remains unknown

| Research question / my hypothesis | What current papers actually establish | What remains **unverified** | My decisive test |
|---|---|---|---|
| **RQ1 · Is a single dense embedding a bottleneck for multimodal future interests?** A multi-slot representation should help most when future interests are far apart. | **[PinnerFormer (2022), Table 3 / §4.2.3](https://arxiv.org/html/2205.04507v1):** Next Action Recall@10 **0.186** → All Action 28d **0.224** → Dense All Action **0.229**; discusses averaging when many positives backpropagate through the same last-step vector. **[PinFM (2025), §3.1 / Table 3](https://arxiv.org/html/2507.12704v3):** adds Multi-Token / Future-Token supervision. *Counterpoint:* PinnerFormer **0.229** vs older PinnerSage 20-cluster **0.046** in Table 1, but architectures/training differ. | More targets help supervision; they do **not** prove multiple output embeddings beat a well-trained single vector. | **A vs B/D**, controlling capacity and total ANN budget; segment the gain by **future-interest entropy**, not just overall Recall@K. **Evidence: direct for multi-target learning; open for representation bottleneck.** |
| **RQ2 · Can assignment become smoother *without* collapsing the interest slots?** Hard argmax may switch the receiving slot after a small parameter update. | **[Pinterest DCM (2025), §3.2.3–3.2.4](https://arxiv.org/html/2506.23060v1):** discusses winner-takes-all with naive hard assignment and uses initialization/routing protections; some alternative models use Straight-Through Gumbel-Softmax. **[SetMIR (2026), Table 2](https://arxiv.org/abs/2608.30251):** replacing Hungarian with per-target argmax yields **63.5%** of full-model R@10 and **2.16** rather than **6.70** active queries. | These results support the **collapse/slot-utilization problem**, not the claim that loss values fluctuate dramatically or that soft OT beats Hungarian. Hard routing changes **direct gradient recipients** discontinuously; soft assignments could instead blur specialization. | **B → D → E** (argmax → one-to-one Hungarian → soft set assignment); track assignment switches *after slot alignment*, per-slot gradient share, active-slot diversity, recall, and loss variance. **Evidence: direct for anti-collapse need; soft matching remains open.** |
| **RQ3 · Does listwise supervision improve future-interest coverage, beyond any assignment change?** A slate-aware loss may align training better with retrieval utility. | **[PinnerFormer (2022), Table 3](https://arxiv.org/html/2205.04507v1):** Recall@10 improves with All Action, while reported **global** P90 Coverage@10 changes from **0.050** (Next Action) to **0.028** (All Action 28d), or **0.042** (Dense All Action). **[TokenMinds (2026), Table 4](https://arxiv.org/html/2606.25147v1):** dense-only / SID-only / combined features give SFV ranking engagement **+0.05% / +0.40% / +0.62%**. | Neither paper shows that listwise CE makes interest-slot assignment smoother. **Global item coverage is not per-user future-interest coverage.** An upstream argmax remains discontinuous even if the downstream loss is listwise. | **B vs C** with identical hard routing; **E vs F** with identical soft routing. A listwise gain must be shown separately on **item ranking**, **unique-interest coverage**, and **gradient stability**. **Evidence: adjacent only.** |
| **RQ4 · How many interest embeddings should be active?** Learn effective cardinality instead of confusing positive-item count with interest count. | **[Pinterest DCM (2025), Table 6 / §4.4](https://arxiv.org/html/2506.23060v1):** HR@100 **0.176 (K=2)** → **0.185 (K=7)** in its setting, but an adaptive-cluster-count variant reports **−0.12% HF Repins** vs fixed K. **[SetMIR (2026), Tables 2, 3, 6](https://arxiv.org/abs/2608.30251):** K sweep saturates (relative R@10 **154.7% / 155.0% / 155.1%** at K=7/10/15); presence/diversity ablations matter; query gating/NMS reduces average ANN calls **10 → 6.70** in its dispatch setup. | An active-slot label derived from *matched items* need not represent a distinct semantic interest, and an unobserved future interest is not necessarily absent. Dynamic K is **not** automatically better than fixed K. | Compare **per-item vs semantic-interest-group supervision**, plus **all-K vs presence-gated** serving. Report active-slot count, item recall, group coverage, duplicated queries, and ANN cost, with sensitivity to grouping noise. **Evidence: direct for gating/capacity tradeoff; open for semantic cardinality.** |
| **RQ5 · What should each interest's retrieval weight mean?** Learned *future demand* may be useful, but optimal quota should consider *marginal unique relevant retrievals*. | **[MIP (ICML 2023), §4.3 / Fig. 3](https://proceedings.mlr.press/v202/shi23b/shi23b.pdf):** evaluates learned, equal, and time-decay preference weights; learned is **not universally superior** across its displayed Pinterest metrics. **[Pinterest DCM (2025), §3.5](https://arxiv.org/html/2506.23060v1):** allocates ANN depth by historical routing mass; reports p90 serving latency **150 → 205 ms**. **[SetMIR (2026), Table 6](https://arxiv.org/abs/2608.30251)** supports query gating, not directly learned per-query depth. | Neither a learned preference weight nor predicted engagement proportion is automatically a **calibrated estimate of marginal retrieval utility**; overlapping interests may waste quota. | **F vs G**, fixed total candidates and ANN budget: equal quota vs routing-mass vs learned future-share vs marginal-utility-aware quota. Measure **incremental unique relevant items per extra ANN fetch** and tail-interest coverage. **Evidence: direct for importance of allocation; optimal quota remains open.** |

**Related questions to revisit after the core dense ablations (not a second open-question list):**

- **Representation freshness and reuse:** [PinnerFormer](https://arxiv.org/abs/2205.04507) and [TokenMinds](https://arxiv.org/abs/2606.25147) motivate low-frequency caching, but do not establish the best cadence for my proposed dynamic slots. Later compare **1h / 6h / 24h refresh** with a lightweight online short-term-interest adjustment, measuring task quality per unit cost.
- **Dense vs generative SID:** [TIGER](https://arxiv.org/abs/2305.05065) generates item IDs; TokenMinds generates coarse future-interest prefixes for downstream ranking. I need a **matched-budget, matched-target** comparison before concluding that one representation family is preferable. The SID-specific objective/decoding experiments belong in **Section 5**.

### 4.2. Controlled A–G ablation sequence

*This matrix is the **implementation plan** for the research questions above; it does not repeat the supporting paper evidence.*

| Run | Controlled change | Comparison / decision criterion |
|---|---|---|
| **A · Single-vector baseline** | One user vector + multi-future InfoNCE / sampled softmax | Strong reference for RQ1; performance by future-interest entropy |
| **B · Hard-routing multi-vector** | Fixed K, per-positive argmax, sampled softmax; **equal per-slot ANN quota for isolation** | **B vs A:** benefit/cost of multiple vectors; unlike deployed DCM, quota is equal here only to isolate representation |
| **C · Hard routing + listwise** | Keep B's assignment/K; add or replace with candidate-slate loss | **C vs B:** what listwise adds **without** smoother matching (RQ3) |
| **D · Set-style matching** | Hungarian one-to-one, presence head, active-only diversity; equal active-slot quota | **D vs B:** global matching / anti-collapse baseline (RQ2/RQ4) |
| **E · Soft set matching** | Change **only** D's matching to soft or unbalanced OT; keep other losses and quota | **E vs D:** smoother responsibility, **and** maintained/improved recall and specialization (RQ2) |
| **F · Soft matching + listwise** | Keep E's assignment; add listwise candidate-slate supervision | **F vs E:** slate/interest coverage at unchanged serving compute; stability is a *separate* measurement (RQ3) |
| **G · Learned retrieval allocation** | Hold F's trained representations fixed; compare equal, routing-mass, learned-share, utility-aware budgets | **G vs F:** unique relevant recall **per unit ANN work**; verify simpler quota baselines are beaten (RQ5) |

**Control and interpretation rules.** Keep the item tower, history cutoff, future-label window, positive/negative pools, sampled-softmax corrections, ANN index, dedup/merge logic, and total retrieval budget constant. Where possible match parameter count/FLOPs; otherwise report them. **Do not** attribute A→B solely to matching, because vector count/model capacity also change. Compare assignment switch rates only after aligning unordered slot identities; never arbitrarily fix Slot 1 to the most important future interest. Pinterest's negative-sampling ablations are a reminder that sampling changes can masquerade as representation gains.

### 4.3. Evaluation and falsification criteria

| Measurement | Main metrics | What would challenge my thesis? |
|---|---|---|
| **Quality** | Item Recall@K / HR@K / NDCG@K; engagement-weighted recall | A multi-slot gain disappears at **equal candidate/ANN budget** |
| **Actual future-interest coverage** | Per-user unique semantic-interest Coverage@K; tail-mode and high-entropy user breakdown | Only global catalog diversity improves, not *per-user* future-interest coverage |
| **Optimization and specialization** | Gradient share per slot; post-alignment assignment switch rate; active slot use; pairwise similarity; loss variance | Softer routing reduces switching but duplicates slots or lowers recall |
| **Variable cardinality** | Active-query count, coverage versus active-K, grouping-granularity sensitivity | Presence thresholds save little cost or remove legitimate minority interests |
| **Serving / allocation** | ANN calls, examined candidates, unique post-dedup candidates, p50/p90 latency, quality-vs-cost curve | Learned weights fail to beat equal/routing-mass allocation after controlling compute |
| **Freshness / transfer (later)** | Quality by refresh interval, recent intent shift, cold-start, downstream task | Heavy cached interests become stale faster than lightweight online adaptation can compensate |

**Current decision:** The strongest *published* support is for **multi-future training** (PinnerFormer/PinFM), **anti-collapse matching and active-slot gating** (DCM/SetMIR), and the importance of **retrieval allocation** (MIP/DCM). My full proposal—**permutation-aware soft set matching + listwise supervision + learned budget allocation**—remains an **untested combination**. I should validate each component independently before claiming an overall solution.

## 5. Next entry (to be added): generating multiple Semantic IDs

I want to keep **dense retrieval** and **autoregressive SID generation** as two distinct branches. The SID-specific entry will investigate:

- For one history with multiple valid future SIDs, should we optimize independent positive likelihoods, a chained future sequence, or a set/list-level objective?
- How much of top-K error comes from **prefix beam pruning**, vs. generation probability mass concentrating on a few near-duplicate interests?
- Can coverage-aware decoding or listwise reward align multi-target training with the final *set* of generated SIDs?
- When are SIDs acting as **unique item identifiers** (TIGER) vs **coarse user-interest features** (TokenMinds)? Are dense representations an appropriate baseline in both cases?

Carry forward previously discussed ideas on item-level dense input vs flattened SID input, quantization depth and collision tokens, and learned continuous/soft-token compression. These belong to the broader **Representation-first vs Generation-first** note, not to the dense-loss ablation above.

### Revision history

- **2026-10-11 · v0.3:** Unified the former Section 4 validation plan and Section 7 open questions into a single **question → paper evidence/counter-evidence → uncertainty → decisive experiment** research agenda. Kept A–G as an implementation matrix; retained freshness and SID-vs-dense comparisons as scoped follow-ups.
- **2026-10-11 · v0.2:** Added primary-paper evidence, counter-evidence, and explicit controlled ablations. Soft matching + listwise + learned utility was not yet validated.
- **2026-10-10 · v0.1:** Initial personal hypotheses on multi-future supervision, single- vs multi-vector representations, matching instability, variable slot cardinality, and weighted retrieval.
