---
layout: post
math: true
title: "Multi-Target Future Interests & Generative Semantic IDs"
description: "Living research notes on dense multi-interest representations, generative semantic IDs, and training-to-decoding alignment."
---

*Part of: [LLM-enhanced Recommendation — From Sequence Encoding to Grounded Generation]({{ '/notes/llm-enhanced-recommendation/' | relative_url }})*

**Status:** Living technical note · v0.5 · Personal hypotheses + evidence ledger
**Tags:** `user-modeling` `multi-positive` `multi-interest` `dense-retrieval` `semantic-ID` `listwise` `retrieval-budget` `generative-retrieval` `beam-search` `training-decoding-alignment`

> **Working thesis.** If our retrieval objective is to cover *multiple plausible future user interests*, a single user vector may be an unnecessary bottleneck. I want to explore a **variable-size set of interest embeddings**, with learned presence and importance, trained using stable set-level assignment and listwise retrieval objectives. Crucially, I have **not yet validated** that soft matching + listwise training outperforms hard matching.

## 1. Why I started thinking about this

I have recently been reading work on **Semantic IDs (SIDs)** and generative recommendation. The original [TIGER / *Recommender Systems with Generative Retrieval*](https://arxiv.org/abs/2305.05065) appeared in **2023**, introducing autoregressive next-item SID generation. The idea has since expanded beyond generating a single item ID: SIDs can provide hierarchical item representations, retrieval candidates, and ranking features.

A particularly interesting 2026 example is Google's [TokenMinds](https://arxiv.org/abs/2606.25147). It generates **multiple coarse-grained future-interest SID prefixes** via a decoder *and* a **dense user embedding** via an encoder. In the paper, their most detailed production evaluation is for **downstream ranking**, not a direct demonstration that the generated prefixes are themselves a complete item-retrieval solution.

Two design choices caught my attention:

1. **Future-window supervision rather than next-item-only supervision.** TokenMinds samples up to 15 watched targets in a 24-hour look-ahead window. Rather than hinging everything on the immediate next action, the model is encouraged to represent a broader range of near-future interests. My interpretation is that this may reduce sensitivity to one-off/noisy observations; it does **not** prove that labels themselves become less noisy.
2. **Amortized heavy user-model computation.** The heavy representation generation runs asynchronously with a 24-hour refresh cadence; downstream ranking consumes cached outputs. This makes larger user models practical at scale. **The look-ahead window and refresh cadence are separate decisions**—multi-target supervision does not itself reduce compute.

Together, these motivate the broader question: **Can a relatively infrequently updated user representation anticipate multiple future interests and be reused by retrieval and ranking?**

The first entry (Sections 1–4) focuses on **dense user representations**, including whether dense multi-positive training has its **own** top-K/set-coverage mismatch (§2.3). The second entry (Section 5) tracks my **Generative SID / retrieval** readings and the unresolved multi-target-to-top-K decoding problem.

## 2. Two dense-representation approaches

| Approach | Representative papers | What the model emits | How multiple future positives are supervised | Main potential limitation |
|---|---|---|---|---|
| **A. Single-vector, multi-positive** | [PinnerFormer](https://arxiv.org/abs/2205.04507) (Pinterest, 2022); [PinFM](https://arxiv.org/abs/2507.12704) (Pinterest, 2025) | One user embedding for a particular serving context / prefix | Several future positives contribute contrastive or sampled-softmax loss against the **same** vector | A single vector may compromise among dissimilar future-interest modes; no explicit per-interest retrieval allocation |
| **B. Multi-vector, multi-interest** | [MIND](https://arxiv.org/abs/1904.08030), [ComiRec](https://arxiv.org/abs/2005.09347); [Pinterest DCM](https://arxiv.org/abs/2506.23060) (2025); [SetMIR](https://arxiv.org/abs/2608.30251) (Snap, 2026) | Several interest embeddings, sometimes with presence/importance | Each positive is associated with one or more interest slots; different matching strategies determine where gradients go | Assignment noise, slot collapse, unknown effective number of interests, online query/ANN budget allocation |

### A. Single embedding + multiple positives

Let `h(H)` be the current user embedding and `v_j` the embedding of future positive item `j`. A simplified objective is:

```math
\mathcal{L}_{\mathrm{multi-positive}} = \frac{1}{M}\sum_{j=1}^{M}\ell_{\mathrm{InfoNCE}}(h(H),v_j).
```

- **Why it is attractive:** simple retrieval infrastructure; reuse one ANN query vector; long-horizon supervision may produce a more stable user embedding when representations are refreshed infrequently.
- **My concern:** if future positives represent several semantically separated modes (e.g., cooking, basketball, ML), gradients need to pass through one vector. It may bias retrieval toward dominant interests, even if average Recall@K improves.
- **Important qualification:** This is a potential **capacity/geometry trade-off**, **not** proof that single-embedding models perform poorly. PinnerFormer explicitly improved on next-action baselines using future-action supervision. In PinFM, multi-token/future-token objectives supervise sequence hidden states, but do not automatically produce a set of independent interest-specific retrieval vectors.

An experiment should stratify by **future-interest diversity/entropy**. The prediction is not necessarily that multi-vector models always win; it is that their relative advantage may grow for users with genuinely multimodal future behavior.

### B. Multiple embeddings + positive assignment

Let an encoder/decoder emit `K` interest vectors `U(H)=\{u_1,\dots,u_K\}`. A common hard assignment is:

```math
k^*(j) = \arg\max_k\, u_k^\top v_j.
```

Pinterest's 2025 DCM associates positives with interest-conditioned output embeddings using an argmax operator. It also adds clustering/routing design choices to mitigate collapse; it is **not** simply an unprotected random-slot argmax model.

**My concerns:**

1. **Discontinuous gradient routing.** If two slots have nearly equal scores, a tiny parameter change can flip which slot receives a positive's direct retrieval gradient. This **does not necessarily imply a large change in loss value**, but can make assignments unstable and starve weaker slots.
2. **Unknown interest cardinality.** A user may have one salient interest or many. Always producing 30 active vectors wastes retrieval budget and can encourage duplication. The number of distinct *future items* is also not the number of distinct *future interests*.
3. **No native contribution estimate.** Even if 5 interest vectors are useful, should they receive equal ANN depth? Historical activity proportions may not equal future interest importance or marginal recall utility.

[SetMIR](https://arxiv.org/abs/2608.30251) is especially relevant: it treats the output as a set, uses learnable queries, Hungarian one-to-one matching, a presence head, and diversity regularization. Its ablations give credible evidence of argmax-based **interest collapse** in the evaluated setup. Still, Hungarian matching is discrete, and per-item matching does not guarantee that slots correspond one-to-one with true semantic interest modes.

### 2.3. Does dense multi-positive prediction have the **same** training-to-top-K mismatch?

**Short answer: yes for set-level relevance/coverage, but *not* for autoregressive decoding.** This distinction matters before claiming that the same problem exists in dense and generative retrieval.

A common single-vector training setup, suppressing the details of sampled negatives, is:

```math
\mathcal L_{\rm dense}=-\sum_{j\in\mathcal P(H)}w_j\log\frac{e^{s(H,v_j)/\tau}}{\sum_{i\in\mathcal C}e^{s(H,v_i)/\tau}},\qquad s(H,v)=h(H)^\top v.
```

At serving, the model performs ANN search to approximate the highest-scoring items. **If** the learned scores exactly rank calibrated probabilities of *one* future sampled action, **and** retrieval is exact, score Top-K and probability Top-K are aligned for expected *single-event Hit@K*. The issue is **not** that top-K nearest neighbor search is inherently inconsistent with a softmax or InfoNCE objective. Approximate ANN can introduce search error, but there is no token-prefix pruning in the dense path.

The **shared mismatch** appears when the desired target is instead *a set of distinct interests over a time window*, or a slate-level utility with diminishing returns from redundant items. Multi-positive CE/InfoNCE increases the probability of observed future actions; it does **not** explicitly maximize distinct-interest coverage under a fixed number of candidates.

**A tiny example worth keeping:** Suppose the learned *single next-action* distribution is Sports item A=0.4, Sports item B=0.4, Cooking item C=0.2. Exact Top-2 returns A+B, which is **optimal for expected next-action hit probability (0.8)**. But it only covers **one of two observed interest groups**; A+C covers both, albeit with a lower next-action hit probability (0.6). Thus a *set coverage* optimization might prefer A+C without contradicting the correctness of the original model's probability ranking.

| Mechanism | Dense embedding retrieval | Generative SID retrieval | My evidence / status |
|---|---|---|---|
| **Multi-positive supervision vs interest-set utility** | Multiple future positives train one or several vectors, usually with pointwise contrastive / sampled-softmax losses; duplicates or dominant modes can saturate Top-K | Multiple targets train SID likelihood, not necessarily interest-set coverage | **Common unresolved objective question**. PinnerFormer Table 3 supports benefits of broader supervision, **not** a solved set-coverage objective. TokenMinds Fig. 5 examines diversity but not weighted ground-truth set coverage. |
| **Capacity / geometry** | A single dot-product vector may not rank several distant semantic modes well; multi-vector routing can improve expressivity | Encoder/dense branch may have the same bottleneck; a beam of SID hypotheses offers another representation format | **Dense-specific hypothesis**. PinnerFormer's strong single-vector results are a caution against assuming multiple is always better. |
| **Training versus inference search** | Exact Top-K over the same score preserves ordering; approximate ANN may miss high-ranked candidates | Beam pruning can remove a promising full SID path due to a weak early prefix | **Different failure modes**. BEAR/APAO study the second; they do **not** directly fix the first. |
| **Score-to-dispatch mismatch** | With multiple interest vectors, training may optimize a `max` or `log-sum-exp` over slots, while serving retrieves fixed per-slot ANN depths and deduplicates | Generated top-K prefixes may all belong to dominant modes, with fixed beam/candidate budgets | **Common budget/aggregation mismatch**, but due to different mechanisms. Compare exact global score Top-K vs per-slot quota ANN. |
| **Training labels / negatives** | Negative sampling distribution, false negatives, temporal exposure and popularity bias may distort the estimated item distribution | Exposure, target sampling and engagement weights can likewise bias future SID targets | **Separate statistical issue**, not a proof of decoding mismatch. PinnerFormer reports sensitivity to negative-source choice and sampling probability correction (Table 4). |
| **Refresh cadence** | Cached user vectors may miss session-local drift | Cached dense + discrete user features may miss the same drift | **Shared staleness issue**, independent of output format; test a lightweight online recent-intent branch. |

**My current conclusion:** Do **not** use the sentence “dense multi-positive has the same beam-search training/inference mismatch.” Instead write: *Both families can learn future-item likelihood without directly optimizing set-level interest coverage; SID additionally has token-prefix search errors, while dense multi-vector retrieval additionally has slot-assignment and ANN-quota errors.*

**Dense experiments to append to §4 (not an extra disconnected research agenda):**

1. **Objective test (RQ1/RQ3):** Single-vector next-item CE versus multi-positive CE versus an explicit slate/group-coverage objective, using the **same item tower, negatives, refresh cutoff and retrieval budget**. Compare next-item Recall@K against **unique future-interest-group Coverage@K**; stratify by future-interest entropy. These metrics can prefer different systems legitimately.
2. **Search test (RQ5):** For fixed multi-slot embeddings, compare (a) exact catalog Top-K under the *trained aggregation score*, (b) exact per-slot Top-N + merge at a fixed total candidate count, and (c) actual ANN. This separates **objective error vs dispatch/quota error vs ANN approximation error**.
3. **Assignment test (RQ2/RQ4):** Hard routing versus Hungarian versus soft matching, matching effective active-K and compute; record per-slot gradient share, aligned assignment switch rate and duplication. Do **not** infer that soft matching stabilizes overall training from smoother gradients alone.
4. **Allocation test (RQ5):** Equal quotas versus historical routing-mass weights versus learned future-interest weights versus *marginal unique-recall utility* weights. A high predicted engagement probability does not automatically mean another candidate from that slot adds value.

**Evidence anchors:** [PinnerFormer, Tables 3–4](https://arxiv.org/abs/2205.04507); [PinFM, Multi-Token/Future-Token training](https://arxiv.org/abs/2507.12704) (**TBD deep read**); [Pinterest multi-embedding/DCM](https://arxiv.org/abs/2506.23060) (**discussed, full reading unconfirmed**); [SetMIR](https://arxiv.org/abs/2608.30251) (**discussed, full reading unconfirmed**); [TokenMinds](https://arxiv.org/abs/2606.25147) (**read and critiqued**). These are *different evidence levels*, not an apples-to-apples leaderboard.

---

## 3. My proposed direction — adaptive multi-interest embeddings

**Architecture hypothesis (not an existing claimed result):**

`User history → Transformer history encoder → up to 30 learnable interest queries → per-query embedding + presence + importance → active ANN queries → weighted candidate aggregation / listwise training`

The model may emit up to `K_{\max}=30` slots, but it should only use as many as are warranted for each user. For slot `k`:

```math
u_k=\mathrm{normalize}(W_e z_k),\quad p_k=\sigma(g_p(z_k)),\quad a_k=g_w(z_k).
```

- `u_k` is a retrieval vector.
- `p_k` is an *activity/presence* score used for gating.
- `a_k` is an *importance* logit used to compare active interests and help allocate retrieval budget.
- Presence and importance are **different concepts**: an interest may exist but deserve few marginal retrieval candidates.

**A potential loss:**

```math
\begin{aligned}
\mathcal{L}={}&\mathcal{L}_{\mathrm{retrieval}}+\lambda_s\mathcal{L}_{\mathrm{soft-set}}\\
&+\lambda_l\mathcal{L}_{\mathrm{listwise}}+\lambda_p\mathcal{L}_{\mathrm{presence}}\\
&+\lambda_d\mathcal{L}_{\mathrm{diversity}}+\lambda_b\mathcal{L}_{\mathrm{budget}}.
\end{aligned}
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

**Experiment labels:** A–G refer to the configurations defined in [§4.2](#dense-ablation-configurations).

| Research question / my hypothesis | What current papers actually establish | What remains **unverified** | My decisive test |
|---|---|---|---|
| **RQ1 · Is a single dense embedding a bottleneck for multimodal future interests?** A multi-slot representation should help most when future interests are far apart. | **[PinnerFormer (2022), Table 3 / §4.2.3](https://arxiv.org/html/2205.04507v1):** Next Action Recall@10 **0.186** → All Action 28d **0.224** → Dense All Action **0.229**; discusses averaging when many positives backpropagate through the same last-step vector. **[PinFM (2025), §3.1 / Table 3](https://arxiv.org/html/2507.12704v3):** adds Multi-Token / Future-Token supervision. *Counterpoint:* PinnerFormer **0.229** vs older PinnerSage 20-cluster **0.046** in Table 1, but architectures/training differ. | More targets help supervision; they do **not** prove multiple output embeddings beat a well-trained single vector. | **Single-vector baseline (A) vs hard-routing multi-vector (B) / set-style matching (D)**, controlling capacity and total ANN budget; segment the gain by **future-interest entropy**, not just overall Recall@K. **Evidence: direct for multi-target learning; open for representation bottleneck.** |
| **RQ2 · Can assignment become smoother *without* collapsing the interest slots?** Hard argmax may switch the receiving slot after a small parameter update. | **[Pinterest DCM (2025), §3.2.3–3.2.4](https://arxiv.org/html/2506.23060v1):** discusses winner-takes-all with naive hard assignment and uses initialization/routing protections; some alternative models use Straight-Through Gumbel-Softmax. **[SetMIR (2026), Table 2](https://arxiv.org/abs/2608.30251):** replacing Hungarian with per-target argmax yields **63.5%** of full-model R@10 and **2.16** rather than **6.70** active queries. | These results support the **collapse/slot-utilization problem**, not the claim that loss values fluctuate dramatically or that soft OT beats Hungarian. Hard routing changes **direct gradient recipients** discontinuously; soft assignments could instead blur specialization. | **B → D → E** (argmax → one-to-one Hungarian → soft set assignment); track assignment switches *after slot alignment*, per-slot gradient share, active-slot diversity, recall, and loss variance. **Evidence: direct for anti-collapse need; soft matching remains open.** |
| **RQ3 · Does listwise supervision improve future-interest coverage, beyond any assignment change?** A slate-aware loss may align training better with retrieval utility. | **[PinnerFormer (2022), Table 3](https://arxiv.org/html/2205.04507v1):** Recall@10 improves with All Action, while reported **global** P90 Coverage@10 changes from **0.050** (Next Action) to **0.028** (All Action 28d), or **0.042** (Dense All Action). **[TokenMinds (2026), Table 4](https://arxiv.org/html/2606.25147v1):** dense-only / SID-only / combined features give SFV ranking engagement **+0.05% / +0.40% / +0.62%**. | Neither paper shows that listwise CE makes interest-slot assignment smoother. **Global item coverage is not per-user future-interest coverage.** An upstream argmax remains discontinuous even if the downstream loss is listwise. | **B vs C** with identical hard routing; **E vs F** with identical soft routing. A listwise gain must be shown separately on **item ranking**, **unique-interest coverage**, and **gradient stability**. **Evidence: adjacent only.** |
| **RQ4 · How many interest embeddings should be active?** Learn effective cardinality instead of confusing positive-item count with interest count. | **[Pinterest DCM (2025), Table 6 / §4.4](https://arxiv.org/html/2506.23060v1):** HR@100 **0.176 (K=2)** → **0.185 (K=7)** in its setting, but an adaptive-cluster-count variant reports **−0.12% HF Repins** vs fixed K. **[SetMIR (2026), Tables 2, 3, 6](https://arxiv.org/abs/2608.30251):** K sweep saturates (relative R@10 **154.7% / 155.0% / 155.1%** at K=7/10/15); presence/diversity ablations matter; query gating/NMS reduces average ANN calls **10 → 6.70** in its dispatch setup. | An active-slot label derived from *matched items* need not represent a distinct semantic interest, and an unobserved future interest is not necessarily absent. Dynamic K is **not** automatically better than fixed K. | Compare **per-item vs semantic-interest-group supervision**, plus **all-K vs presence-gated** serving. Report active-slot count, item recall, group coverage, duplicated queries, and ANN cost, with sensitivity to grouping noise. **Evidence: direct for gating/capacity tradeoff; open for semantic cardinality.** |
| **RQ5 · What should each interest's retrieval weight mean?** Learned *future demand* may be useful, but optimal quota should consider *marginal unique relevant retrievals*. | **[MIP (ICML 2023), §4.3 / Fig. 3](https://proceedings.mlr.press/v202/shi23b/shi23b.pdf):** evaluates learned, equal, and time-decay preference weights; learned is **not universally superior** across its displayed Pinterest metrics. **[Pinterest DCM (2025), §3.5](https://arxiv.org/html/2506.23060v1):** allocates ANN depth by historical routing mass; reports p90 serving latency **150 → 205 ms**. **[SetMIR (2026), Table 6](https://arxiv.org/abs/2608.30251)** supports query gating, not directly learned per-query depth. | Neither a learned preference weight nor predicted engagement proportion is automatically a **calibrated estimate of marginal retrieval utility**; overlapping interests may waste quota. | **F vs G**, fixed total candidates and ANN budget: equal quota vs routing-mass vs learned future-share vs marginal-utility-aware quota. Measure **incremental unique relevant items per extra ANN fetch** and tail-interest coverage. **Evidence: direct for importance of allocation; optimal quota remains open.** |

**Dense train→retrieval alignment note:** §2.3 distinguishes **single-event Top-K optimality** from **future-interest set coverage** and separates **exact search**, **per-slot ANN dispatch**, and **approximate-search** errors. This supplies the interpretation checks for RQ1, RQ3 and RQ5 without adding a redundant experiment list here.

**Related questions to revisit after the core dense ablations (not a second open-question list):**

- **Representation freshness and reuse:** [PinnerFormer](https://arxiv.org/abs/2205.04507) and [TokenMinds](https://arxiv.org/abs/2606.25147) motivate low-frequency caching, but do not establish the best cadence for my proposed dynamic slots. Later compare **1h / 6h / 24h refresh** with a lightweight online short-term-interest adjustment, measuring task quality per unit cost.
- **Dense vs generative SID:** [TIGER](https://arxiv.org/abs/2305.05065) generates item IDs; TokenMinds generates coarse future-interest prefixes for downstream ranking. I need a **matched-budget, matched-target** comparison before concluding that one representation family is preferable. The SID-specific objective/decoding experiments belong in **Section 5**.

<a id="dense-ablation-configurations"></a>

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

## 5. Research Entry — Generative Retrieval with Semantic IDs

**Added:** 2026-10-11
**Scope:** *My* reading history, interpretations, outstanding evidence, and a concrete next-reading agenda. This is deliberately **not** a generic SID survey. It distinguishes **generating item identifiers**, **using SIDs as ranking features**, **generating coarse user-interest tokens**, **compressing features as soft tokens**, and **optimizing inference**.

> **Working thesis.** Turning recommendation into autoregressive SID generation makes it possible to model structured candidate paths. But **learning likelihood for several valid future target IDs is not the same as retrieving a diverse, useful top-K set**. The loss, SID tokenization, decoding algorithm, and downstream use must be evaluated together. The **dense** branch has an analogous set-utility mismatch, but **not** beam-prefix pruning (§2.3). The size of the observed online uplift cannot be attributed to this mismatch without a controlled experiment.

### 5.1. Papers I actually read and critiqued — original mechanisms and *my* doubts

**Status rule:** Entries in this table have **directly documented personal technical questions or critiques** from our conversations, including PLUM. “Read and discussed” means we examined specific paper mechanisms, not that every section has been audited. Papers introduced only by the assistant remain **TBD**. The other dense papers may be *discussed in detail* without a confirmed independent full-paper reading.

| Paper · source | What it actually does / strongest reported evidence | My observation / unresolved critique | Where I should update next |
|---|---|---|---|
| **[TIGER — *Recommender Systems with Generative Retrieval*](https://arxiv.org/abs/2305.05065) (2023)** · **Studied** | RQ-VAE learns item SID code sequences; seq2seq model predicts next item SID autoregressively; an extra collision-resolution token identifies otherwise indistinguishable items. Evaluates generative retrieval and cold-start behavior. | **(1)** The fundamental benefit is turning item retrieval into conditional token generation—not proof that SID content semantics alone outperform a strong dense ANN baseline. **(2)** Cold-start may already be served by content embeddings / semantic KNN; I want matched baselines. **(3)** The last collision token is necessary only if the output must identify a *unique item*, not when a prefix stands for a coarse interest. **(4)** Lower RQ levels reduce residual error, but are not guaranteed interpretable taxonomy labels. | Re-read SID semantic-analysis/cold-start and collision-token ablations; compare retrieval output and quantization tradeoffs. |
| **[Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations](https://arxiv.org/abs/2306.08121) (Google, 2023; revised 2024)** · **Studied** | Uses SIDs as **ranking model features** through learnable n-gram / SentencePiece subpieces. Reports generalization benefits for fresh/tail content in YouTube ranking, not autoregressive item retrieval. | The gains could come from **learnable collaborative adaptation / parameter sharing** on top of semantic content, rather than quantization itself. The decisive comparison would be original content embeddings **with a parameter-matched trainable adapter**, controlled lookup capacity, and item-density slices. Do not infer that a SID token carries universally readable semantic meaning. | Write a *matched-budget dense vs SID-as-feature* ablation proposal. |
| **[PLUM — *Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations*](https://arxiv.org/abs/2510.07784) (Google/YouTube, 2025)** · **Read and discussed** | Adapts a pretrained LM to SID item generation through SID tokenization, continued pretraining (CPT), and recommendation-task fine-tuning. It is **full item-SID generative retrieval**, unlike TokenMinds' coarse user-interest features. | I questioned **(1)** whether expanding each behavior into several SID tokens increases context length and token-level attention noise compared with *item-level* history embeddings; **(2)** whether using **pre-quantization dense item embeddings as input** would be a better or simpler baseline while keeping SID as output; **(3)** whether gains are attributable to LM pretrained knowledge/CPT, SID vocabulary, or richer metadata with unequal feature budgets; and **(4)** whether a fully verbalized, matched-input/matched-compute baseline is missing. These were **my critiques**, not facts proved by the paper. | Add an input-format × target-format ablation: dense item history vs SID-token history vs verbalized metadata, all predicting the same SID outputs, matched context/FLOPs and feature coverage; record cold-start / tail + latency. |
| **[Beyond Item IDs: Scaling Short-Form-Video Recommendation via Semantic-Native Long Sequence Modeling](https://arxiv.org/abs/2606.07546) (Google, 2026)** · **Studied** | Retains a **2-level SID prefix**, flattens it into a compact sequence ID, compresses ultra-long behavior histories by temporal folding, processes compressed tokens with global query tokens, and produces a pooled user representation for downstream recommendation. **Not** SID item autoregressive decoding. | I see strong **systems/scale validation**, but relatively limited algorithmic novelty: truncation + flattening/concatenation + Transformer/global tokens + masked pooling are familiar components. My open question is how much of the gain comes from longer histories vs the SID representation. The two-level code does not imply four independent hierarchy levels. | Decompose ablations for `SID only`, `sequence length only`, `folding/global-token only`; separate empirical industrial value from theoretical novelty. |
| **[STATIC — *Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators*](https://arxiv.org/abs/2602.22647) (Google/YouTube, 2026)** · **Studied** | Encodes a valid-SID prefix trie as an accelerator-friendly CSR transition structure; masks invalid next codes without CPU–GPU/TPU traversal overhead. Paper reports **0.033 ms per step**, **0.25%** of inference time for the constraint mechanism in its measured setup. | Most convincing contribution is **accelerator-native inference engineering** rather than new user/SID learning. My extension: if early SID codes separate corpora or constraints well, could a smaller beam or fewer invalid expansions be sufficient? **Not demonstrated by STATIC**; beam survival, quality, and branching must be measured. | Track valid-prefix branching, mask selectivity, beam width, actual kernel time and item recall; test *early-corpus-separation* only as a hypothesis. |
| **[Token Factory: Efficiently Integrating Diverse Signals into Large Recommendation Models](https://arxiv.org/abs/2606.19635) (Google, 2026)** · **Studied; adjacent, NOT SID generative retrieval** | Converts heterogeneous ranking/user/query/candidate features into compressed **soft input tokens** for large ranking models; addresses input-context and serving-compute costs. | My concern was whether token-maker MLPs are co-trained with the ranking model or could be trained separately, what features can be offline-precomputed vs online, and whether the paper fairly compares against stronger long-context baselines at matching latency/throughput. **Do not classify its soft token output as a generated item SID.** | Keep under **Representation-first / Soft-token feature compression**; add a cost accounting table (offline vs online precompute, token length, ranking quality). |
| **[TokenMinds: Pretrained User Tokens and Embeddings](https://arxiv.org/abs/2606.25147) (Google/YouTube, 2026)** · **Studied** | Encoder → **1,152D dense user embedding**; decoder → **40 length-4 SID prefixes** (20 LFV / 20 SFV). Train with up to **15 future watches in a 24h window** using autoregressive SID CE; cache representations on a **24h refresh**. Focuses its presented production results on ranking; SFV satisfied engagement **+0.05% dense-only**, **+0.40% token-only**, **+0.62% combined** (Table 4). | The paper explicitly learns *multiple future targets*, but does **not** optimize whether the 40 beams match the *set* of future interests. Its Table 1 reports **single held-out SID-prefix Recall@10**; Fig. 5 studies **diversity/duplicate rates**, not set-level coverage. Dense user embedding also lacks its own direct representation loss (gradients come through decoder cross-attention). Gains of SID-over-dense may reflect training alignment and downstream learnable embedding adaptation, not inherent superiority of discrete tokens. | Main **mismatch experiment** (§5.3–5.4), plus `dense independent-supervision` and `candidate-aware use of 40 tokens` ablations. |

**Another modeling distinction I raised:** Generating **multiple interest SID prefixes** is not the same as predicting **multiple SID quantization levels independently**. Within one SID, residual codes are *conditional on previous codes*; treating levels as independent outputs can create invalid or meaningless combinations. For multi-interest coverage, I want K **separate valid prefix paths**, not independent choices at each level. This matters for training-factorization and beam validity ablations.

**My distinction to preserve:** TIGER = **predict unique item SID for retrieval**. Better Generalization / Beyond Item IDs = **use SID in ranking/long-sequence user encoding**. TokenMinds = **predict multiple coarse SID interests to augment rankers**. STATIC = **constrained SID decoding systems**. Token Factory = **soft input token compression**. Similar terminology does not make their output spaces, losses, or deployment paths equivalent.

### 5.2. My recurring SID questions (independent of the multi-target gap)

| Question I have raised | What I think so far | Evidence that would change my view |
|---|---|---|
| **Does SID add semantic information beyond the original dense item embedding?** | The quantization maps from dense features and **cannot create semantic content ex nihilo**. Benefits could come from shared parameters, small structured vocabularies, behavior-trained token lookups, and autoregressive conditional modeling—not intrinsically better input semantics. | Strong dense retriever + trainable item adapter + multi-vector dense baselines with matched compute and training signals. Relevant: Better Generalization, [LIGER](https://arxiv.org/abs/2411.18814). |
| **Is a residual codebook really a human-interpretable coarse→fine hierarchy?** | Coarse prefixes can group semantically related items; deeper code stages primarily reduce residual/reconstruction error. Token indices at the same depth need not mean the same category without their prefix. | Prefix purity, mutual information with human categories, quantization distortion, and next-interaction transition tests; compare RQ-VAE with taxonomy- or behavior-aligned quantizers. |
| **Why does TIGER need a final disambiguation ID, while TokenMinds does not?** | The former must map generated IDs back to **distinct catalog items**; TokenMinds intentionally uses **coarse prefixes as interest labels**, so collision can be a feature rather than a bug. | Compare full-item identifier vs coarse-cluster objective under the same target/serving metric; study [Don't Waste the Last Token](https://github.com/sb-ai-lab/sid-collisions). |
| **How much should we credit production engineering as research contribution?** | Beyond Item IDs and STATIC may be primarily system design. That is valuable industrial evidence, but should not be mistaken for proof of better generative modeling / better learned semantics. | Breakdown by quality gain, representational benefit, scalability, compute/memory and serving benchmarks, with causal ablations. |

### 5.3. The multi-target **training → top-K decoding mismatch** I am most interested in

**Setup.** Let history be `H`. The user has multiple legitimate future watches `W_1,\ldots,W_M`, mapped to SID prefixes `s_1,\ldots,s_M`. At inference I want a **set** of `K` relevant future-interest tokens, not necessarily their exact temporal order.

**A. Independent multi-positive training** — same context, several possible outputs (closer to training LMs on alternative correct responses as separate examples):

```math
\mathcal L_{\mathrm{ind}}=-\sum_{i=1}^M w_i\log P_\theta(s_i\mid H).
```

**B. TokenMinds-style chained multi-target training** — decoder factors the sampled future sequence, also conditioning on previously selected targets:

```math
\mathcal L_{\mathrm{chain}}=-\sum_{i=1}^M w_i\log P_\theta(s_i\mid H,s_{<i}),
```

where each `P_\theta(s_i\mid\cdots)` is itself a product of per-code autoregressive probabilities. This is a schematic form of **TokenMinds Eq. (1)**, omitting the SID-token index. **A and B are not mathematically equivalent** because B models additional conditional dependencies. Only *independent* same-`H` training examples with the same weights/normalization are equivalent to batching independent multi-positives.

**C. Actual inference** — beam-search returns `\hat G_K(H)=\operatorname{BeamTopK}_s P_\theta(s\mid H)`, in TokenMinds with **four-code future-interest prefixes**. It does **not** decode a 15-item ordered future watch sequence and compare it to the whole ground-truth set.

**My mismatch claim:** Optimizing A or B does not directly optimize the usefulness or coverage of `\hat G_K(H)` for the **entire future-interest set** `G(H)`. A can learn an accurate marginal probability distribution and still allocate most top-K paths to dominant, closely related modes. B additionally learns future-target ordering dependencies that may not be used when inference requests individual SID prefixes. These are **hypotheses about alignment**, not proven explanations of TokenMinds' ranking lift.

**Separate the gaps:**

| Gap type | Example | Does existing research address it? | Measurement I need |
|---|---|---|---|
| **G1 · Target/set objective** | 15 valid future targets, but loss never directly measures which distinct interests are covered by 40 outputs. | TokenMinds ablates multi-target/temporal sampling; **not** a direct set-level coverage loss. [F-GRPO](https://arxiv.org/abs/2605.12995) shows coverage rewards in a neighboring LLM candidate-generation/list framework, **not exactly SID-prefix user modeling**. | Engagement-weighted **future unique-interest Coverage@K**; precision/utility vs diversity; target grouping robustness. |
| **G2 · Chained training vs independent marginal interest prediction** | Training target `i` conditions on prior sampled future targets; serving top-K standalone prefixes from `H`. | TokenMinds Eq. (1) reveals the conditioning; no controlled independent-vs-chained set-coverage comparison in its reported results. | Identical architecture/data; compare independent multi-positive CE vs chained AR CE under identical beam decoder. |
| **G3 · Prefix pruning/search error (generative-specific)** | A good full SID path is dropped at an early decoding layer. | [BEAR](https://arxiv.org/abs/2601.22925) adds beam-survival-aware loss; [APAO](https://arxiv.org/abs/2603.02730) adds adaptive prefix-ranking/worst-prefix optimization. | Oracle/exact search vs beam; positive-prefix survival by depth; Recall@K across beam widths. |
| **G4 · Top-K distribution concentration** | Many distinct beams, but most describe the same dominant interest (or duplicate many nearby prefixes). | [Autoregressive Generation Strategies for Top-K Sequential Recommendations](https://arxiv.org/abs/2409.17730) explores temperature sampling + rank/relevance aggregation. TokenMinds Fig. 5 measures prefix duplication/collision—not relevance-weighted set coverage. | Per-user mode coverage/entropy, unique ground-truth-group recall and downstream ranker lift for beam vs diverse/stochastic generation. |
| **G5 · Feature/downstream scoring mismatch** | Generated SID list is pooled into a single user feature; ranker may need candidate-specific selection. | TokenMinds §3.4 discusses learnable SID embeddings, pooling and optional cross-attention, but has no decisive feature-use control comparing all combinations. | Pooled user token vs candidate-aware token attention, matched ranker capacity; test SFV/LFV and tail-interest slices. |
| **G6 · Serving/corpus constraints** | Beam spends effort on invalid/ineligible codes; SID prefix layout controls validity masks, branching, and practical latency. | STATIC demonstrates accelerator-native constrained decoding, **not** a new end-to-end set-coverage objective. | Valid-beam fraction, item eligibility, beam size, latency/quality vs GPU utilization, corpus prefix entropy. |

**Two subtleties I do not want to lose:**

1. A same-context **categorical next-interest distribution** sums to one, so it cannot assign probability 0.9 to every mutually exclusive next interest. But *future 24h presence probabilities* are **multi-label**, and several interest modes can simultaneously have high probability. These are different training targets; a set-aware loss must say which one it means.
2. **Diversity ≠ future-ground-truth coverage.** Producing 40 different SIDs is insufficient if none describes a low-frequency but genuinely important future interest. Likewise, an online ranking increase of +0.62% is not inherently “small” for an already strong ranker; proving *why* it has that magnitude requires intervention, not speculation.

**What the TokenMinds evidence actually says** (source: [paper Eq. (1), Table 1, Figure 5, Table 4](https://arxiv.org/html/2606.25147v1)):

- Multi-target + look-ahead + prefix truncation help **individual held-out prefix Recall@10**: full **0.291**, no multiple targets **0.265**, no look-ahead **0.278** (session metric). This is *support* for broad future supervision, not for set-level optimum.
- Figure 5 reports beam diversity comparable to observed-watch diversity under the paper's collision/duplication metrics. This weakens a **simple beam-collapse** claim; it does not settle relevant-mode recall.
- Online SFV satisfied engagement goes **+0.05% embedding**, **+0.40% SID tokens**, **+0.62% both**, relative to its production baseline. Encoder embedding is *indirectly* trained through decoder CE; LE token embeddings adapt end-to-end in the downstream ranker. Hence there are multiple possible explanations for the difference.
- The 24-hour **future look-ahead** and 24-hour **refresh cadence** are separate modeling and infrastructure decisions. Neither establishes the best freshness horizon for my use case.

### 5.4. Next decisive experiments — generative branch (separate from dense A–G)

| ID | Change relative to controlled baseline | What it isolates | Success/falsification criterion | Evidence/reading anchor |
|---|---|---|---|---|
| **SID-G0 · Baseline** | Reproduce next-item SID CE + standard beam; include strong dense ANN baseline | SID quantization/generation vs dense retrieval under the *same task and cost* | Report item Recall/NDCG, coverage, p95 latency, tail/cold-start slices; do not mix user-token ranking gains with item retrieval gains | TIGER, LIGER |
| **SID-G1 · Independent vs chained targets** | Same history, targets, weights, SID tokenizer and model; vary only independent multi-positive CE vs TokenMinds-style chained targets | **G2** conditioning objective | Which gives higher ground-truth *set* coverage and downstream quality at fixed compute? | TokenMinds Eq. (1), Table 1 |
| **SID-G2 · Change only decoding** | Standard beam vs stochastic/temperature + aggregation vs diverse beam; fixed total decoded candidate budget | **G4** top-K concentration | Improve weighted unique-interest recall **without excessive relevance loss** | Top-K generation strategies (2024) |
| **SID-G3 · Change only prefix survival loss** | Base CE vs BEAR vs APAO on same decoder and beam | **G3** prefix pruning | Higher positive-prefix survival and item Recall@K for same beam width/latency | BEAR, APAO |
| **SID-G4 · Set/list-aware objective** | Base CE vs differentiable coverage surrogate or sequence-level reward; keep decoder policy and budget controlled | **G1** future-interest utility | Set coverage and ranking utility gains; check under-coverage of rare interests and instability | F-GRPO, Listwise Generative Retrieval (adjacent) |
| **SID-G5 · Downstream representation** | Pool generated SID embeddings vs candidate-aware attention; learnable vs frozen original-prefix embedding; add dense-only comparator | **G5** feature adaptation | Are gains explained by user-interest coverage, learnable token features, or larger ranker capacity? | TokenMinds Table 3/4; Better Generalization |
| **SID-G6 · Serving/SID structure** | Vary prefix depth, collision handling, constraint trie, and beam width under fixed workload | **G6** validity/efficiency | Accuracy-quality/latency Pareto vs item identifier needs; test *early corpus separation* only as an extension | TIGER, Beyond Item IDs, STATIC |

**Metric caution:** compute set coverage on **distinct future-interest groups** as well as exact SID prefixes, with popularity/engagement weighting and within-group items. A prefix may collide across distinct items; arbitrary ground-truth item count is not ground-truth interest count. Report beam valid-path rate and **oracle-retrieval upper bound** to separate representational, search, and catalog limits.

### 5.5. Where I will update the note next

| Next update location | Task, in my own research terms | Read first / next artifact | Progress |
|---|---|---|---|
| **§2.3 / §4 RQ1/RQ3/RQ5** | Check dense multi-positive likelihood vs interest-group coverage, without conflating it with beam pruning | PinnerFormer Table 3/4 → PinFM / Pinterest DCM / SetMIR (**deep-read TBD where marked**); add exact-TopK-vs-quota experiment | ☐ Derived distinction written; empirical test TBD |
| **§5.3 G3/G4** | Identify whether *beam pruning* or *top-K concentration* dominates on an actual SID corpus | BEAR → APAO → Top-K Generation Strategies; add prefix survival plots | ☐ Not tested |
| **§5.3 G1/G2** | Write down precisely how independent multi-positive vs chained targets change the modeled distribution and top-K utility | TokenMinds Eq. (1) + independent CE ablation proposal; add small synthetic example | ☐ Hypothesis only |
| **§5.2 SID semantics** | Determine when prefix depth, uniqueness token, and RQ hierarchy help over strong dense embedding | TIGER + Better Generalization + Don't Waste Last Token + 2026 SID survey | ☐ Needs controlled baseline |
| **§5.4 SID-G5** | Test whether multiple predicted coarse SIDs are more useful to candidate-aware ranking than pooled user features | TokenMinds §3.4/Table 3–4; revisit PLUM CPT/pretraining ablations (previously discussed) | ☐ Reading queued |
| **§5.4 SID-G6** | Profile valid-path sparsity, beam width and accelerator costs; verify whether early prefix partitioning helps | STATIC + NVIDIA SID-GR serving examples | ☐ Extension untested |

**How I will decide whether the mismatch theory is useful:** It must win against **decoding-only fixes** and **strong independent future-positive baselines** at equal model/data/beam/candidate budgets. If beam-diversity changes alone close the gap, I should not introduce an expensive custom set loss. If direct user-token listwise objectives improve only per-user diversity but not retrieval/ranking utility, I should not call it a successful recommendation improvement.
