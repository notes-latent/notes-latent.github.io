---
layout: post
math: true
title: "Constructing Good Semantic IDs for LLM Prediction"
description: "Research hypotheses on semantic purity, multilingual query quantization, LLM output predictability, and stable SID vocabularies."
---

**Status:** Research note · Working hypotheses, not production claims
**Focus:** Query/text embedding quantization

## 1. Scope and central question

**How should I construct a high-quality Semantic ID (SID) vocabulary from text/query embeddings so that a pretrained LLM can reliably predict useful SIDs, while preserving the semantic distinctions that matter for retrieval?**

This is a **tokenizer and output-space design** note, not an end-to-end query recommendation proposal. I am interested in using the LLM's pretrained language/world knowledge to predict **target SIDs as outputs**. I am **not** studying how to construct trajectories, how to put SID histories into the LLM *input*, or how to train a business ranking policy in this note. Those are separate downstream projects. I mention **within-bucket expression ranking** only to clarify the architectural boundary: semantic SID construction should not bake in changeable engagement/advertising objectives. Future purpose-specific use cases—e.g., refinement or complementary inspiration—matter only insofar as they impose requirements on the SID vocabulary.

**Current thesis.** A good SID for this setting should be **semantically coherent within each bucket, language-neutral where appropriate, sufficiently expressive, efficient to predict, and stable enough to serve as a versioned indexing contract**. It need not force every related concept into a shared long prefix. A pretrained LLM can, in principle, predict multiple distinct SIDs for a complex intent; whether it does so effectively requires a separate downstream experiment.

**A second, deliberate design principle is semantic–business decoupling.** The embedding and SID tokenizer should define **what a query means**, not **which wording the platform currently prefers for engagement or ad outcomes**. Once an LLM predicts a semantic SID, a separate, potentially precomputed **within-bucket expression ranking** can choose among semantically valid query formulations using product-specific objectives. Engagement and monetization targets can evolve without continually redefining the SID codebook. This is a proposed architectural benefit, not a locally validated causal lift.

Evidence is separated into two classes throughout:

- **Published:** a finding or method reported by a paper; not proof that it carries over to my query distribution.
- **Hypothesis / TBD:** a design intuition that still needs controlled tests.

## 2. Why SIDs, and what does the LLM actually need?

[TIGER (2023)](https://arxiv.org/abs/2305.05065) demonstrated the appeal of encoding items as short residual-quantized token sequences rather than unrelated atomic item IDs. While quantization can provide storage and vocabulary benefits, **the more compelling reason for my project is to expose a constrained, grounded, learnable output space to a pretrained LLM**.

For an input query embedding `x`, a quantizer produces an SID such as `(c1, c2, c3)`. A generator can then model:

```math
P(SID\mid \text{natural-language query, available context})
=\prod_{l=1}^{L}P(c_l\mid \text{text, context},c_{<l}).
```

There are two different questions:

1. **Can the quantizer assign coherent semantic regions?** This is a property of the input embeddings, normalization, clustering objective, buckets, and codebook.
2. **Can an LLM learn to predict those regions?** This depends on the label space, training support, pretrained knowledge, text-to-code alignment, and generation constraints. It is *not* established by low reconstruction error or balanced bucket sizes alone.

A numeric code is not intrinsically meaningful to a pretrained language model. My goal is not necessarily to make each code **human-readable**; it is to make each target sufficiently **consistent and learnable** from language, while its ground truth stays tied to a real query/content region.

### The representation property I now prioritize

My main requirement is **one-way semantic purity**:

> **If two queries map to the same full SID bucket, they should represent the same or sufficiently compatible retrieval intent, without silently adding contradictory or unrelated meaning.**

The inverse is **not required**: semantically related queries may live in different buckets. A complex query such as *red dress + blue clutch* may eventually require multiple target SIDs, but this does **not** imply the quantizer must encode both concepts inside one prefix path. Forcing all pairwise cosine similarities to translate into proportionally long shared prefixes would impose a stronger—and sometimes geometrically impossible—tree constraint than the downstream generator actually needs.

Thus I differentiate:

- **Bucket purity:** low risk of unrelated / conflicting meanings sharing a full SID (**primary requirement**).
- **Neighborhood preservation:** cosine-nearest queries remain retrievable after quantization (**useful but not sufficient**).
- **Prefix structure:** early codes offer meaningful coarse partitions and later codes refine them (**useful when it improves purity, routing, and predictability; not an end in itself**).
- **Learnability:** conditional SID prediction is feasible with an LLM (**must be evaluated separately**).

## 3. How this project differs from adjacent SID research

The important distinction is **what each line of work optimizes and whether its reported downstream gains isolate tokenizer quality**.

| Work / direction | Main optimization target | Relevance to my work | What I am **not** adopting or assuming |
| --- | --- | --- | --- |
| TIGER (2023) | RQ-VAE item SIDs for generative recommendation | Foundational discrete-output representation | Its reconstruction objective alone guarantees pure query buckets or easy LLM prediction |
| [CoST (2024)](https://arxiv.org/abs/2404.14774) | Contrastive quantization / neighborhood preservation | Useful diagnostic for semantic damage during quantization | Perfect neighborhood similarity is necessary, or InfoNCE directly enforces semantic equivalence within a full SID |
| [LETTER (2024)](https://arxiv.org/abs/2405.07314) | Collaborative alignment and code assignment diversity | A reminder that reconstruction is not the sole objective | CF alignment: **out of scope**, since my immediate representation is query text semantics |
| [TopoTok (2026)](https://arxiv.org/abs/2607.18600) | Layer-wise topology preservation | Can inform quantizer evaluation at different depths | Better reconstructed-neighbor overlap necessarily means higher **bucket semantic purity** or stronger multi-intent prediction |
| [OneRec (2025)](https://arxiv.org/abs/2506.13695) and [OneSearch (2026)](https://proceedings.mlr.press/v306/chen26al.html) | Industrial SID-based generative retrieval; RQ-KMeans, codebook allocation, downstream hit rates | Strong engineering references for post-hoc quantization and L1 capacity | Their click/order retrieval improvements are equivalent to my embedding-oracle recall; I am **not** designing an all-in-one ranker here |
| [PRQ-KMeans (2026)](https://arxiv.org/abs/2608.24207) | Improve post-hoc residual quantization (common-component removal, centroid refinement, projection residual) | **Most immediately actionable quantizer baseline** for already-trained embeddings | Industrial HitRate@50 gains prove bucket equivalence or conditional LLM predictability |
| [ReSID (2026)](https://arxiv.org/abs/2602.02338) | Improve SID sequential predictability together with representation design | Relevant idea: generation-aware code assignment | Its collaborative/structured-feature representation pipeline directly transfers to my text-only query encoder |
| [ETEGRec (2025)](https://arxiv.org/abs/2409.05546) / [DIGER (2026)](https://github.com/junchen-fu/DIGER) | Couple downstream recommendation gradients with tokenizer training | A possible later research direction | Joint training is required upfront; frequent code reassignment would also impose retraining/migration costs |
| [SIDReasoner (2026)](https://arxiv.org/abs/2603.23183) | Teach a model to understand/reason about **fixed** SIDs | Demonstrates that language-to-code alignment is distinct from codebook construction | SIDReasoner directly repairs SID centroids or reassigns each query/item during reasoning training |
| [GRAM/GRLM Term IDs (2026)](https://aclanthology.org/2026.findings-acl.984/) | Use standardized natural-language terms as output identifiers | Meaningful **alternative output-space design** using native language tokens | Text terms automatically solve ambiguity, fine-grained tail coverage, or stable exact grounding |
| [GEAR (2026)](https://arxiv.org/html/2609.39327) | Prefix-aware quantization, collapse/collision handling, industrial generation and reranking | Relevant for prefix organization, collision and future lifecycle | Its full ad-serving architecture or item-level reranker is the goal of this note |

**Interpretation:** The field spans **compression/reconstruction**, **neighborhood geometry**, **collaborative relevance**, **downstream task coupling**, **predictability**, **language grounding**, and **online-system efficiency**. My niche is narrower: **construct a clean, multilingual query SID output vocabulary that an LLM can learn without requiring SID tokens as part of the input, without collaborative or business-reward objectives in embedding/tokenizer training, and without jointly training the tokenizer and LLM at the outset**.

**Distinct architectural choice:** Unlike lines of work that push interaction, collaborative, or downstream recommendation signals into the representation/tokenizer (e.g., LETTER, ETEGRec, DIGER), I want to keep the **semantic partition** independent of the **business preference among equivalent surface forms**. That separation creates a stable indexing/LLM-output contract and leaves engagement, advertising, and other OKRs to downstream ranking. This is a difference in optimization *scope*, not evidence that those methods are generally inferior: task-coupled representations may improve their own end-to-end objectives, and semantic-only IDs must still prove downstream usefulness.

Paper-reported Recall/HitRate lifts generally involve retraining a generative recommender, and should not be described as tokenizer-only gains. In particular, the reported PRQ-KMeans industrial evaluation uses clicked/ordered item retrieval, whereas the oracle defined in §5 uses *cosine similarity of original query embeddings*. These evaluate different things.

## 4. What makes a *good* SID for LLM prediction?

### A. Semantic purity before enforced similarity geometry

For a complete SID bucket `b`, the desired property is high:

```math
P(\text{retrieval-intent compatible}(q_i,q_j)\mid SID(q_i)=SID(q_j)=b).
```

A useful *error* is **false semantic merging**: a bucket contains contradictory attributes, different entity types, or an unrequested extra intent. For example, `red dress` and `red formal dress` can be cosine-close but not interchangeable in an intent-preserving task; `red dress` and `blue handbag` should not become equivalent simply because they co-occur in a session.

A bucket may intentionally represent a **coarse** concept at L1, while the full SID is more specific. I should audit semantic purity at multiple prefix depths against the *appropriate level of specificity*, rather than insist that every broad parent bucket contain exact paraphrases.

### B. Language neutrality without erasing real constraints

Equivalent queries in different languages should not waste L1 capacity in separate language-only clusters; however color, season, year, product version, brand, and purpose can be semantically essential. My normalization must preserve these real distinctions. A low language-probe accuracy by itself is not proof of good multilingual semantic alignment.

### C. Good bucket capacity and traffic distribution

Avoid code collapse and extreme traffic concentration; keep sufficient effective coverage across tail intents. But **balanced marginal assignments do not prove conditional predictability**. An almost uniform SID prior can have higher entropy than a highly concentrated one. What matters is whether natural language reliably disambiguates the correct code *beyond a popularity-only predictor*.

### D. Useful levels, but no unnecessary prefix dogma

A large L1 may better isolate broad semantic regions and shorten the code sequence. Later codes should contribute meaningful finer distinctions, not encode language noise or redundant residual energy. However an RQ code at L2 is not automatically a human-interpretable subcategory unless the training procedure actually enforces parent-conditional partitions. The role of each level should be demonstrated with data.

### E. Predictable, valid and stable output labels

For a generator, useful diagnostics include per-level conditional NLL/accuracy, full-path validity, held-out query-to-SID accuracy, and reachable SID coverage. These require training or evaluating an actual predictor; they cannot be inferred from code entropy alone. **Stable label-to-bucket identity** matters because changing centroids or renumbering codes can invalidate an LLM's learned output mapping.

### F. Support for future multiple outputs

A single query may have multiple meaningful target intents. The SID space should allow the LLM to predict **multiple existing, coherent codes**, not require all complementary relations to be encoded as one shared prefix. This is an **output-space requirement**, not a proposal to put SIDs into the model input or to build trajectory training data here.

### G. Semantic IDs as stable buckets for downstream expression ranking

There is an additional product capability beyond generating a valid SID: **multiple query surface forms can express the same retrieval intent while producing different engagement or ad outcomes on a specific platform**. For example, if both `red dresses` and `dresses in red` pass a strict semantic-equivalence check and map to the same bucket, a downstream component may learn which phrasing attracts more qualified engagement or better monetization in that context. In contrast, `red summer wedding dress` must **not** be treated as interchangeable with `red dress`: it adds season and occasion constraints and belongs to a distinct intent/target SID (or an appropriately distinct finer bucket).

The intended separation is:

```text
Text/query embedding -> language calibration -> semantic quantizer -> stable SID
                                                              |
                                     LLM predicts target semantic SID(s)
                                                              |
                                          SID -> valid candidate expressions
                                                              |
                     Independent static/precomputed expression ranking (or contextual policy)
                              engagement / ads / marketplace OKRs + guardrails
```

Conceptually, for a predicted bucket `b`, choose only among expressions that preserve its intent:

```math
e^*(b,c)=\arg\max_{e\in E(b):\,\mathrm{valid}(e,b)} S_{\mathrm{business}}(e\mid b,c).
```

Here `E(b)` is the set of grounded surface forms in a bucket, `c` is optional product context, and `S_{\mathrm{business}}` is **not** part of the embedding or codebook-training loss. Scores can be **precomputed/static per bucket** for an inexpensive initial implementation, and changed much more frequently than the SID assignment. Context-dependent reranking is a possible future extension, not this note's focus. The LLM still uses SIDs as **outputs**, not as required input features.

**Why this separation is attractive (design hypotheses, not demonstrated lifts):**

- **Cleaner invariants:** `same SID` means compatible semantic intent, not `same monetization value`. Engagement can vary across valid phrasings without forcing the quantizer to separate them.
- **Independent optimization opportunities:** expression choice, relevance, user engagement, and advertising yield can be monitored and optimized as downstream product/OKR layers instead of contaminating the semantic embedding space.
- **Lower SID churn:** policy, advertiser demand, and ranking-prior changes can be handled by refreshing expression scores rather than rerunning quantization or retraining an SID-predicting LLM. Genuine new semantic distinctions still warrant embedding/codebook review.
- **Better diagnosis:** if a query expression underperforms, distinguish `wrong semantic bucket`, `LLM predicted wrong SID`, and `right SID but poor expression ranking`.

**Risks and boundaries:** bucket identity must be genuinely compatible with the phrasing being ranked; engagement/ads estimates can be confounded by position, prior exposure, or traffic mix; a revenue-optimized phrase must not introduce misleading constraints or undermine user relevance. Evaluate business policies on qualified engagement, user value, and monetization guardrails, not raw CTR/revenue alone. These **downstream performance tests are intentionally outside the scope** of tokenizer construction.

## 5. Evaluation methodology: reproducible definitions

A SID tokenizer needs three independent kinds of evidence: **semantic fidelity**, **code-space diagnostics**, and **LLM output predictability**. For an interpretable benchmark, keep the text encoder, query corpus, normalization, quantization budget, retrieval policy, and candidate budget fixed when comparing tokenizers. Unless otherwise specified, exclude the source query itself from neighbors and evaluate on held-out queries.

| Evaluation layer | Exact question | Metrics / method |
| --- | --- | --- |
| **Embedding-neighborhood preservation** | Can SID-based lookup recover the cosine neighbors of the original text embedding? | Cosine-oracle Neighbor Recall@K under a fixed candidate budget N; report by SID depth, language, and query-frequency cohort. |
| **Prefix occupancy** | How much of the *valid* hierarchical code space is occupied? | Distinct used prefixes B_l, available prefix capacity M_l, utilization U_l = B_l / M_l, and conditional child fan-out. |
| **Bucket distribution** | Are semantic buckets excessively concentrated by *query count* or *search traffic*? | Occupied-bucket size histogram, Gini, normalized entropy, effective bucket count; calculate once for unique queries and separately for search-volume weights. |
| **Semantic purity** | Do queries sharing a full SID express compatible retrieval intent without contradictory or extra constraints? | Independent equivalence/compatibility labels on sampled in-bucket pairs; micro- and macro-averaged agreement, false-merge and attribute-conflict rates. |
| **Within-bucket expression validity** | Are there multiple safe surface forms available for independent downstream selection? | Count of independently validated equivalent phrasings per SID, proportion of query traffic in buckets with 2+ valid forms, rate of added or contradictory constraints. |
| **Cross-language consistency** | Do equivalent translations share buckets without deleting meaningful attributes? | Same-SID / same-prefix rate on trusted parallel or paraphrase pairs, plus constraint-preservation/conflict rate by language pair. |
| **LLM output predictability** | Can natural language condition a model to predict a valid SID? | Per-level conditional NLL, top-k code accuracy, full-path exact-match, valid-path and tail-path coverage; compare to unconditional or prefix-only popularity baselines. |
| **Temporal robustness** | Will a frozen codebook still represent future terms and unseen product years? | Time-split neighbor recall and semantic purity, new-path assignment/validity, label churn across separately versioned codebooks. |

### 5.1 Cosine-oracle Neighbor Recall@K — what it does and does not measure

Let `G_K(q)` be the top-K nearest *other* queries to `q` using cosine similarity of the **original, unquantized embedding**. Let `R_N(q)` be up to N candidates returned by a precisely specified **SID retrieval** procedure (e.g., exact full-SID bucket, longest-prefix backoff, or multi-prefix probing).

```math
\operatorname{NeighborRecall@K\mid N}
=\frac{1}{|Q|}\sum_{q\in Q}\frac{|G_K(q)\cap R_N(q)|}{K}.
```

**Procedure.** Freeze the query corpus and embeddings; construct cosine top-K neighbors; assign SIDs with each candidate tokenizer; build its index; run the *same* SID lookup, tie-breaking and N-candidate truncation policy; average recalls. Report a curve across N rather than only one operating point. Also report results for individual depths and head/torso/tail query-frequency cohorts. A traffic-weighted version averages per-query recall with weights proportional to search volume.

**Caveat.** This is an *embedding-fidelity oracle*, **not behavioral relevance**. An exact cosine index is optimal for recovering cosine-defined neighbors. Do not confuse SID *prefix lookup* with a different evaluation that reconstructs quantized vectors and compares the top-K neighbors in two continuous spaces (as in some topology-preservation studies).

### 5.2 Prefix occupancy and distribution balance — beyond bucket size

At depth `l`, let `B_l` be the number of observed unique prefixes `(c1,...,cl)`, and let `M_l` be the number of *available valid* prefixes at that depth. Under **global independent codebook alphabets**, `M_l = ∏_{j=1}^l K_j`, so:

```math
U_l=B_l/M_l.
```

For parent-conditioned clustering or restricted prefix tries, count **valid reachable prefixes**, not the naïve product. Report both the numerator and denominator. A second useful statistic is each occupied parent's number of occupied children (fan-out distribution); this exposes collapsed branches hidden by an aggregate U_l.

For each complete bucket `b`, define `n_b` = number of distinct queries assigned to it; define `w_b` = sum of their observed search counts. Compute **two distinct distributions**:

```math
p_b^{\mathrm{query}}=n_b/\sum_j n_j,\qquad
p_b^{\mathrm{traffic}}=w_b/\sum_j w_j.
```

On either distribution, report:

- **Gini** of bucket counts or weights: `0` means equally sized occupied buckets; larger values indicate more concentration. State whether empty buckets are included (prefer occupied-only Gini alongside utilization).
- **Entropy** `H(p) = -∑_b p_b log p_b`, and **normalized entropy** `H(p)/log B` using `B` occupied buckets. `exp(H)` is the **effective number of buckets**; always include `B`, since a high normalized entropy over only a few used buckets can hide severe under-utilization.
- **Percentiles** of `n_b` / `w_b`, top-1%-bucket traffic share, and counts of empty/near-empty prefixes.

These are capacity and load diagnostics. **Higher balance does not necessarily reduce text-conditioned SID NLL**: a uniform marginal SID distribution has higher prior entropy, while semantic purity can make the correct SID easier to infer *given the query*.

### 5.3 Semantic bucket purity — independent relevance supervision

Cosine-neighbor recall does not establish whether all queries within one bucket express the **same retrieval intent**. For each complete SID bucket, sample pairs using a documented protocol (e.g., random, boundary/hard pairs, and frequency-stratified pairs), then obtain human-reviewed or independently calibrated teacher labels:

- **Equivalent / compatible:** preserves item category and required constraints such as color, size, occasion, year, and product version.
- **Different specificity:** one query adds a constraint, e.g., `red dress` vs. `red wedding dress`; it may be a useful *refinement*, but not necessarily interchangeable within an exact semantic bucket.
- **Contradictory / unrelated:** incompatible attributes or different product intents.

If `e_ij` is `1` only when sampled pair `(i,j)` is semantically interchangeable for the designated retrieval task, the basic **pairwise purity** is:

```math
\operatorname{Purity}=\frac{\sum_{(i,j)\in S}e_{ij}}{|S|},
\qquad \operatorname{FalseMergeRate}=1-\operatorname{Purity}.
```

Report **macro-average** (equal weight per evaluated bucket) as well as **micro-/traffic-weighted average** (weight pairs or buckets by clearly stated sampling probabilities). Also report the rate of *specific* attribute contradictions; a mere high cosine similarity must never be used as its own semantic-equivalence label. Audit at L1/L2 using depth-appropriate coarse-grained compatibility, not a full-SID equivalence requirement.

For **within-bucket expression validity**, deduplicate surface forms, validate semantic equivalence, and count distinct safe expressions per complete SID. This tests whether independent expression ranking has a **valid candidate pool**; actual engagement or ads lift would require a *separate downstream evaluation* and should not enter SID quantizer training.

### 5.4 Cross-lingual and LLM-prediction diagnostics

**Cross-language test.** Construct a held-out set of trusted translations/paraphrases, including difficult negatives differing in year, color, brand, or intended product. Compute (1) fraction sharing a full SID, (2) fraction sharing the first `l` codes, and (3) constraint-conflict rate within multilingual buckets. Slice by language pair, query frequency, and specificity. Merely suppressing language identity in embeddings is insufficient if it also loses real product attributes.

**LLM learnability test.** Freeze one SID codebook. Use the same train/test split and fixed text-to-SID predictor capacity across tokenizers. For SID `c_1,...,c_L`, evaluate per-depth **teacher-forced** negative log-likelihood:

```math
\operatorname{NLL}_l
=-\frac{1}{|Q|}\sum_{q\in Q}
\log P_\theta(c_l(q)\mid q,c_{<l}(q)).
```

Add top-k code accuracy, full-path exact match and valid-generated-path rate under the same constrained decoding policy. Report performance on rare codes and **never-before-observed paths** separately: a valid path might still be unreachable for the model. To avoid mistaking memorized popularity for semantic understanding, compare against a code-frequency baseline such as `P(c_l | c_<l)`; also note that vocabularies with different widths have different chance levels and NLL scales. An intent-conditioned target-SID evaluation belongs to a separate downstream study.

### 5.5 Freshness and evaluation controls

Freeze embedding calibration and a tokenizer at time `t0`; evaluate on **strictly later** queries/new product terms/years, without leaking future data into centroid fitting. Monitor semantic purity and cosine-oracle recall for new terms, proportion assigned to reachable SID paths, and changes in assignments when comparing version `v` vs. `v+1` on a fixed anchor set. Measure exact-path churn and semantic compatibility separately: mere code-number changes can break a deployed generator even if bucket meaning is unchanged.

**Cross-paper metric warning.** PRQ-KMeans/OneSearch report behavioral click/order HitRate or MRR after retraining a generative retriever; TopoTok reports reconstructed-embedding neighbor overlap as well as end-to-end recommendation recall. Those are **not** equivalent to the cosine-neighbor SID lookup above. Keep the ground-truth source (embedding cosine vs. judged relevance vs. logged behavior) explicit in every result.

## 6. Temporal freshness and SID lifecycle

A codebook trained today must remain useful when queries mention a new model year, newly launched products, new brands, emerging slang, or changed associations. Two distinct failures are possible:

1. **Representation / partition failure:** the frozen encoder + codebook places the new concept in a bucket that adds or confuses meaning.
2. **Generator reachability failure:** a new query is correctly assigned an existing or novel path, but the LLM has not learned when to predict that path.

These require different remedies. **New-content assignment does not require codebook retraining for each item**, and new facts / new language-to-code associations may be updated without changing centroids. **Changes in expression-level engagement or advertising preferences are a third, separate kind of drift:** update the downstream ranking/score table, not the semantic codebook, as long as the intent represented by each bucket is still valid. However, truly new semantic distinctions or systematically impure buckets may justify refreshing the codebook.

| Lifecycle layer | Typical change | Principle (not a published universal update frequency) |
| --- | --- | --- |
| **Query/content admission and SID index** | New embeddings assigned to current centroids; index additions/removals | Can be refreshed frequently according to serving SLAs |
| **Output-label grounding / predictor adaptation** | New vocabulary, emerging paths, updated query distribution | Update independently where possible, while retaining SID meaning |
| **Embedding calibration and SID codebook** | New geometry, language shifts, persistent bucket conflict and representation drift | Refresh **based on measured semantic quality drift**; version codebook and coordinate migration |
| **Within-bucket expression preferences (downstream, not quantizer)** | Engagement/ads/marketplace scoring of semantically equivalent expressions changes | Recompute static ranking scores or change policy independently; no SID retraining merely for business-prior shifts |

Published systems illustrate why frequencies should be separated: GEAR (2026) discusses nearline index refresh and dynamic model training, but **minute-level index updates do not mean minute-level full K-Means retraining**. [SID staleness alignment (2026)](https://arxiv.org/abs/2604.13273) studies compatibility between changed SIDs and prior retrieval models; its reported computational savings are not an assumed SLA for my text tokenizer.

A robust evaluation should train or freeze a tokenizer at time `t_0`, then evaluate **strictly later** query cohorts (new-year expressions, fresh product terms, first-seen concepts). Measure both bucket semantics and LLM path reachability, while separately tracking code-assignment churn between versions. Potential safeguards include anchor queries, versioned label maps, backward-compatible mapping where feasible, and fallback retrieval for poorly represented new semantics.

## 7. Future research directions

These are **research questions**, not a sequential implementation plan or claims of completed evaluation. §5 contains the metric definitions; they are not repeated here.

| Direction | Hypothesis to test | Discriminating experiment |
| --- | --- | --- |
| **Semantic purity vs. neighborhood geometry** | Low false-semantic-merge rates may matter more to an LLM output space than perfect cosine-neighbor preservation. | Compare tokenizers on independently judged bucket equivalence *and* cosine-oracle recall, under matched index budgets. |
| **Multilingual calibration** | Removing language-dependent shifts saves coarse SID capacity without erasing genuine distinctions. | Benchmark trusted cross-language paraphrases and year/color/version minimal pairs before/after calibration. |
| **Codebook structure and residual design** | Front-loaded L1 capacity, parent-aware subdivisions, or projection residuals can improve label quality. | Budget-matched `1024–256–256` vs `256–1024–256`; compare hierarchical/global RQ and PRQ-style residual updates. |
| **LLM-facing predictability** | Semantically coherent bucket assignments can be easier to learn from natural language than purely balanced or low-MSE assignments. | Freeze each quantizer and compare a common text-to-SID predictor's conditional NLL, path accuracy, and rare-path reachability. |
| **Temporal stability** | Index admission and language-to-SID alignment can refresh faster than the underlying codebook. | Evaluate first-seen years/concepts with time-split data; estimate when quality drift warrants versioned retraining. |
| **Semantic–business decoupling** | Valid interchangeable expressions can be ranked for independent downstream objectives without changing SID meaning. | First verify equivalence and expression-pool coverage; assess engagement/ads outcomes only in a separate product study. |

**Ablation comparability note:** `1024–256–256` vs. `256–1024–256` matches token count and global-codebook vector count, but **not** cost when a separate child codebook is trained for each parent. The PRQ projection-residual comparison should use identical multilingual calibration and code widths.

## Closing thesis

A useful SID output vocabulary should be **semantically consistent, multilingual where appropriate, predictable from natural language, and stable across business-objective changes**. It is not necessary for the tokenizer to encode every query-to-query relationship or maximize engagement. The key design choice is to **learn the semantic partitions independently, let the LLM predict them, and leave preference among valid surface forms to downstream systems**. Their downstream value remains a hypothesis until tested independently.

## References

- [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065) (2023).
- [CoST: Contrastive Quantization based Semantic Tokenization for Generative Recommendation](https://arxiv.org/abs/2404.14774) (2024).
- [Learnable Item Tokenization for Generative Recommendation](https://arxiv.org/abs/2405.07314) (2024).
- [Topology-Aware Tokenization for Generative Recommendation](https://arxiv.org/abs/2607.18600) (2026).
- [OneRec Technical Report](https://arxiv.org/abs/2506.13695) (2025).
- [OneSearch: A Preliminary Exploration of the Unified End-to-End Generative Framework for E-commerce Search](https://proceedings.mlr.press/v306/chen26al.html) (2026).
- [PRQ-KMeans: Projection Residual Quantization for Semantic ID Tokenization](https://arxiv.org/abs/2608.24207) (2026).
- [Rethinking Generative Recommender Tokenizer: Recsys-Native Encoding and Semantic Quantization Beyond LLMs](https://arxiv.org/abs/2602.02338) (2026).
- [Generative Recommender with End-to-End Learnable Item Tokenization](https://arxiv.org/abs/2409.05546) (2025).
- [DIGER: Differentiable Semantic ID for Generative Recommendation — official repository](https://github.com/junchen-fu/DIGER) (2026).
- [Reasoning over Semantic IDs Enhances Generative Recommendation](https://arxiv.org/abs/2603.23183) (2026).
- [Unleashing the Native Recommendation Potential: LLM-Based Generative Recommendation via Structured Term Identifiers](https://aclanthology.org/2026.findings-acl.984/) (2026).
- [Generative End-to-end Ad Retrieval at Douyin](https://arxiv.org/html/2609.39327) (2026).
- [Mitigating Collaborative Semantic ID Staleness in Generative Retrieval](https://arxiv.org/abs/2604.13273) (2026).
