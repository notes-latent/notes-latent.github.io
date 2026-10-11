---
layout: post
title: "Multi-Target Future Interest Modeling"
description: "Personal research hypotheses on multi-positive supervision, adaptive multi-interest embeddings, soft matching, and learned retrieval allocation."
---

*Part of: [LLM-enhanced Recommendation — From Sequence Encoding to Grounded Generation]({{ '/notes/llm-enhanced-recommendation/' | relative_url }})*

**Status:** Personal research notes — hypotheses to validate

## 1. Motivation: Beyond Next-Item Prediction

I have recently been exploring Semantic ID (SID)-based recommendation, starting from Google's [TIGER](https://arxiv.org/abs/2305.05065) (2023) and more recent work on generative retrieval, ranking, and user representation learning.

One particularly interesting paper is [TokenMinds](https://arxiv.org/abs/2606.25147) (Google, 2026), which generates both dense user embeddings and multiple SID-based user-interest tokens to enhance downstream ranking.

Two ideas caught my attention.

**First, predicting multiple future interests rather than only the immediate next item.**

The next interaction can be noisy and may not represent the user's broader preferences. Predicting multiple interactions within a future time window could make representations less sensitive to isolated observations and capture a wider range of interests.

**Second, amortizing expensive user-model computation.**

Rather than repeatedly running a large model for every recommendation request, we can periodically generate reusable user representations and cache them for multiple downstream tasks.

The training prediction horizon and representation refresh cadence are distinct design decisions, but they can complement each other.

This motivates an interesting research question:

**How can we efficiently learn a reusable user representation that captures multiple future interests rather than compressing everything into a single embedding?**

I see two main directions: dense multi-interest representations and generative SID-based representations. This entry focuses on the dense branch.

## 2. Dense User Representation: Two Approaches

| Approach | Representative Work | Key Idea | Potential Problem |
|---|---|---|---|
| Single embedding + multiple future positives | [PinnerFormer](https://arxiv.org/abs/2205.04507), [PinFM](https://arxiv.org/abs/2507.12704) | One contextual embedding predicts multiple future positive items | Different interest modes must be represented by the same vector |
| Multiple embeddings + future positives | [Pinterest DCM](https://arxiv.org/abs/2506.23060), [SetMIR](https://arxiv.org/abs/2608.30251) | Generate multiple interest-specific user vectors | Target assignment, interest collapse, variable cardinality, retrieval allocation |

### 2.1 Single Embedding with Multi-Positive Supervision

PinnerFormer introduces long-term future-action supervision instead of focusing entirely on next-item prediction.

A simplified objective is:

`L = Average_j InfoNCE(user_embedding, future_positive_j)`

Multiple future positive examples contribute gradients to the same user embedding.

**My observation:** This improves the amount and quality of future-interest supervision. However, if the positives represent very different semantic directions, a single embedding may need to compromise between them.

This may create a representation bottleneck, particularly for users with highly diverse interests.

This is not a claim that single-vector models necessarily perform poorly. PinnerFormer demonstrates clear improvements over next-action baselines.

**Hypothesis H1:** The relative advantage of multi-interest representations should increase with the multimodality of a user's future-interest distribution.

I would evaluate this by segmenting users according to future-interest entropy rather than relying only on aggregate Recall@K.

### 2.2 Multiple Embeddings with Hard Interest Assignment

Pinterest's 2025 multi-embedding retrieval framework uses several interest-conditioned user embeddings.

For each positive item, the training process selects the most relevant interest representation using an argmax assignment.

`selected_slot = argmax_k similarity(user_interest_k, positive_item)`

This allows different embeddings to specialize in different user interests.

However, I see three potential limitations.

**1. Assignment instability**

A small model update can change the selected interest slot. The loss gradient is routed to a different embedding even if the corresponding similarity scores change only slightly.

This may introduce unstable supervision and winner-takes-all behavior. It does not necessarily imply that the loss value itself changes dramatically.

**2. Fixed number of interest embeddings**

Different users may have different numbers of active interests. Fixing the number of meaningful output embeddings may introduce redundancy or insufficient coverage.

The number of future positive items is also not necessarily equal to the number of distinct future interests.

**3. Retrieval contribution**

Even when the model learns multiple useful interest embeddings, they should not necessarily receive equal retrieval budgets.

Some interests may have stronger expected future engagement, while others may produce more marginally useful candidates.

Pinterest addresses parts of the collapse problem using clustering initialization and routing. Snap's SetMIR further introduces one-to-one Hungarian matching, presence prediction, and diversity regularization.

However, the joint problem of stable assignment, variable interest count, and learned retrieval allocation remains interesting.

## 3. My Proposed Direction: Adaptive Multi-Interest Embeddings

I want to explore a model that directly outputs a variable-size set of interest embeddings, each associated with an importance score.

### Architecture

`User history → Transformer encoder → Learnable interest queries → Interest embeddings + presence + importance`

For example, the model can support up to 30 interest slots.

For each interest slot, it predicts:

- **Interest embedding:** the retrieval representation.
- **Presence probability:** whether the interest should be active.
- **Importance weight:** the relative contribution that interest should make to retrieval.

The maximum number of slots does not imply that all 30 should be active for every user.

I prefer learned interest queries over simply taking the final 30 sequence-position embeddings. Different historical timestamps do not necessarily correspond to distinct interests.

### Training: Soft Matching + Listwise Supervision

My current hypothesis is that **soft set matching combined with listwise supervision** may be more stable than independent hard argmax assignment.

The intended design has several components.

**Soft matching**

Instead of assigning every positive item to exactly one interest through an unstable local argmax, use a differentiable or approximately differentiable soft assignment mechanism.

Potential approaches include entropy-regularized optimal transport or soft set matching with inactive slots.

The matching should ideally be permutation-aware rather than forcing a specific interest into a fixed output position.

**Contrastive retrieval objective**

Encourage the corresponding interest embeddings to match relevant future positives and separate them from negatives.

**Listwise ranking objective**

Construct a candidate slate and train the model to rank multiple future positives above negatives.

The candidate score should consider both embedding similarity and learned interest importance.

The hypothesis is that smooth scoring and listwise supervision could reduce abrupt gradient changes, but **listwise loss alone will not fix hard assignment instability** if the hard routing remains elsewhere in the pipeline.

**Diversity regularization**

Penalize excessive similarity between active interest embeddings so that all slots do not collapse onto the dominant user interest.

This should not indiscriminately force every output embedding to be orthogonal.

**Presence and importance learning**

Presence determines whether to issue a retrieval query. Importance helps determine how much retrieval budget to allocate to that interest.

An important distinction is that expected engagement frequency is not necessarily equivalent to expected marginal retrieval utility.

## 4. What I Want to Validate

| Experiment | Representation | Assignment / Loss | Question |
|---|---|---|---|
| A | Single embedding | Multi-positive InfoNCE | Baseline |
| B | Fixed multi-embedding | Hard argmax + sampled softmax | Standard multi-interest approach |
| C | Fixed multi-embedding | Hard argmax + listwise | Does listwise alone help? |
| D | Dynamic active slots | Hungarian + presence + diversity | SetMIR-style baseline |
| E | Dynamic active slots | Soft matching + contrastive | Can smoother assignment improve stability? |
| F | Dynamic active slots | Soft matching + listwise + diversity | Does joint training improve interest coverage? |
| G | Dynamic active slots | Same as F + learned retrieval weights | Does adaptive quota improve fixed-budget retrieval? |

Key evaluation metrics should include Recall@K, NDCG, future-interest coverage, slot utilization, embedding redundancy, gradient concentration, assignment switching, ANN query count, and serving cost.

I also want to test whether improvements are concentrated among users with more diverse future interests.

**This idea is not validated yet.** Soft matching could make representations less specialized, and listwise optimization could improve ranking without improving interest coverage. The ablations need to distinguish these effects.

## 5. Next Section: Generative Semantic IDs

To be expanded in a later reading entry.

Key questions:

- How should we train a generative model when the same user history has multiple valid future SID targets?
- Does autoregressive multi-target sequence training align with top-K SID decoding?
- When do beam-search pruning and interest duplication reduce coverage?
- Can beam-aware, set-aware, or listwise training improve generative retrieval?
- How should dense and discrete representations be compared under the same inference budget?

This section will build on TIGER, TokenMinds, BEAR, and other SID-based generative recommendation work.

## 6. Paper Reading Tracker

| Paper | Year | Status | Focus for Next Reading |
|---|---|---|---|
| [PinnerFormer](https://arxiv.org/abs/2205.04507) | 2022 | Discussed; revisit | Dense All Action and gradient averaging |
| [TIGER](https://arxiv.org/abs/2305.05065) | 2023 | Studied | SID semantics and dense baselines |
| [MIP](https://proceedings.mlr.press/v202/shi23b.html) | 2023 | To read | Learned interest weights |
| [Pinterest Multi-Embedding / DCM](https://arxiv.org/abs/2506.23060) | 2025 | Discussed; revisit | Assignment and collapse ablations |
| [PinFM](https://arxiv.org/abs/2507.12704) | 2025 | To read deeply | Multi-Token and Future-Token objectives |
| [PLUM](https://arxiv.org/abs/2510.07784) | 2025 | Reading queue | SID pretraining and dense-input baselines |
| [BEAR](https://arxiv.org/abs/2601.22925) | 2026 | To read deeply | Beam-search-aware objective |
| [TokenMinds](https://arxiv.org/abs/2606.25147) | 2026 | Studied | Multi-target loss and ranking integration |
| [Token Factory](https://arxiv.org/abs/2606.19635) | 2026 | Studied | Soft-token compression and ranking |
| [SetMIR](https://arxiv.org/abs/2608.30251) | 2026 | To read deeply | Set matching, presence and diversity |

## 7. Open Research Questions

**Q1.** Does soft matching improve both training stability and retrieval quality?

**Q2.** How can we distinguish multiple future positive items from multiple future interest modes?

**Q3.** How should we learn interest importance so that it represents retrieval utility rather than historical popularity?

**Q4.** Can a slowly refreshed heavy user model retain enough long-term interests while lightweight online features capture short-term intent?

**Q5.** Under a fixed serving budget, when should we prefer dense multi-interest retrieval versus generative SID prediction?

---

**Revision history**

- **2026-10-10 — v0.1:** Initial hypothesis on multi-target dense user representation, hard assignment instability, soft matching, listwise objectives, dynamic interest slots, and adaptive retrieval weights. Generative SID training and decoding reserved for a future entry.
