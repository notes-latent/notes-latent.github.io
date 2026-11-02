---
layout: post
math: true
title: "SID Inference and Serving"
description: "Autoregressive Transformer inference for two-level SIDs: full recomputation, KV caching, decoding, candidate diversity and serving."
---

I have been spending time on SID-based generative retrieval, first thinking about how to construct the IDs and then about what a retrieved set should cover. More recently, I started wondering what happens when this model has to serve requests. Can the GPU cluster that serves an ordinary prediction model also serve a generative retriever? What changes when a request produces several candidate paths rather than one prediction?

That question connects to another one: how do I make the generated SIDs more diverse, and which decoding choices might broaden relevant intent coverage? Before choosing a serving framework or a diversity objective, I wanted to understand exactly what runs between the input and the final candidate set.

This note follows one model architecture through **autoregressive Transformer → beam search → SID resolution → ranking**. I use familiar LLM decoding methods to understand how candidate selection can change while the model stays the same. Pseudocode makes the forwards, beam expansion and KV reuse explicit; a small executable SID example lets me check what each selector really does.

**Status:** Study note · Executable selector examples; measurements comparing full recomputation and KV-cached execution of the same Transformer remain to be completed.

Related notes: [Constructing good SIDs](/notes/constructing-good-semantic-ids-for-llm-prediction/) · [Intent-aware retrieval sets](/notes/from-semantic-ids-to-intent-aware-retrieval-sets/)

## 1. The Actual SID Inference Path

This note assumes an autoregressive Transformer that generates a fixed two-level Semantic ID, `(S1, S2)`, conditioned on a prompt x. Its probability factorization is:

```math
P(S_1,S_2\mid x)=P(S_1\mid x)P(S_2\mid x,S_1).
```

```text
Incoming requests
    ↓
Request queue and batching
    ↓
Transformer forward on prompt
    ↓
L1 logits: P(S1 | prompt)
    ↓
Select top-W S1 tokens per request
    ↓
Transformer forward conditioned on each selected S1
    ↓
L2 logits: P(S2 | prompt, S1)
    ↓
Accumulate log probabilities
    ↓
Select top-K complete SID sequences per request
    ↓
Validate and resolve SIDs to candidates
    ↓
Optional downstream ranking
    ↓
Return results
```

Both SID levels are predicted by the **same autoregressive Transformer**. The second forward incorporates the selected S1 token into the context. It uses the same model parameters as the first forward.

The GPU executes the model's tensor operations. The serving system decides which requests run together and when. Beam search decides which hypotheses survive within each request. Two SID levels mean two generation steps; they do not specify the number of Transformer layers or the beam width.

## 2. One Autoregressive Model, Two Execution Paths

The comparison is **full recomputation without KV cache versus incremental decoding with KV cache**. Both paths use the same Transformer, weights and probability factorization. Caching changes how historical computation is reused.

The pseudocode uses B requests, prompt length T, first-level beam width W, final candidate count K, and second-level vocabulary size V2. It shows equal-length prompts for clarity. Attention masks and position IDs are implicit in the model calls and must remain equivalent across both paths, including when prompts are padded.

### Baseline: Full Recomputation Without KV Cache

The baseline performs a full Transformer forward at each generation step:

```python
# Conceptual inference code; model is in evaluation mode.
def full_recompute_sid(prompt, model, W, K):
    B, T = prompt.shape

    # First full forward: [B, T]
    logits1 = last_logits(model(prompt, use_cache=False))
    logp1 = level_log_probs(logits1, level=1)
    score1, sid1 = topk(logp1, W)                  # [B, W]

    # Copy each prompt into its W beam branches.
    expanded = repeat_interleave(prompt, W, dim=0)
    input2 = append_token(expanded, sid1.reshape(B * W))

    # Second full forward: [B*W, T+1]
    logits2 = last_logits(model(input2, use_cache=False))
    logp2 = level_log_probs(logits2, level=2)
    V2 = logp2.shape[-1]
    logp2 = logp2.reshape(B, W, V2)

    # Keep each complete SID together with its cumulative score.
    joint_scores = score1[:, :, None] + logp2
    return select_top_sid_pairs(sid1, joint_scores, K)
```

`last_logits` extracts next-token logits at the last valid position. `level_log_probs` applies the model's level-specific token validity and normalization convention; that convention must be identical in both execution paths. `topk` returns scores and their actual SID token IDs.

`select_top_sid_pairs` associates each score with its complete `(S1, S2)` token pair and selects the best K pairs independently for each request. For example, branches A and B produce A1, A2, A3 and B1, B2, B3. Several winning pairs may come from A. K counts final SIDs; W counts the first-level paths retained for expansion.

The second forward recomputes the prompt for every selected beam. This repeats work, but keeps execution simple. For two SID levels there are **two batched Transformer forwards**: one over B prompts and one over B×W extended prompts. The W branches do not require W sequential model calls.

### Optimization: Incremental Decoding With KV Cache

The cached path retains the prompt's attention keys and values, then processes only the selected S1 token in each branch:

```python
# prefill and decode_one are conceptual wrappers around the same model.
# Each returns next-token logits and the updated per-layer KV cache.
def cached_sid(prompt, model, W, K):
    B, T = prompt.shape

    # Prefill: process [B, T] once and retain the prompt's KV.
    logits1, prompt_kv = model.prefill(prompt)
    logp1 = level_log_probs(logits1, level=1)
    score1, sid1 = topk(logp1, W)                  # [B, W]

    # Match the same request/beam order as full recomputation.
    beam_kv = fork_prompt_cache(prompt_kv, W)

    # Second forward: one new S1 token per branch, with prompt history.
    logits2, beam_kv = model.decode_one(
        sid1.reshape(B * W), beam_kv
    )
    logp2 = level_log_probs(logits2, level=2)
    V2 = logp2.shape[-1]
    logp2 = logp2.reshape(B, W, V2)

    joint_scores = score1[:, :, None] + logp2
    return select_top_sid_pairs(sid1, joint_scores, K)
```

`fork_prompt_cache` describes logical branching. An implementation may physically copy the cache or share its common prefix. In either case, each branch must attend to the correct prompt with the correct mask and token positions.

**Why is the second forward still necessary?** Prefill predicts a distribution over S1 before the decoder selects a token. To predict S2, the model must process that selected S1 under the original prompt context. KV caching reuses historical computation; the new S1 still traverses every Transformer layer.

| Execution path | Second Transformer input | Prompt computation |
| --- | --- | --- |
| Full recomputation | Complete `[prompt, S1]` for each branch | Repeated in every branch |
| KV-cached decoding | New S1 token with cached prompt K/V | Reused from prefill |

For a fixed two-token SID, selecting S2 completes generation. No third forward is required merely to return the SID.

### Training Versus Inference

During teacher-forced training, the ground-truth S1 is already available. A causal forward over `[prompt, true_S1]` can predict S1 at the last prompt position and S2 at the S1 position, using the appropriately shifted targets. Both predictions are computed in one parallel forward without exposing future target tokens to earlier positions.

During inference, S1 must first be predicted and selected. Generating S2 therefore requires the second forward shown above, with either recomputed or cached prompt history.

### What is actually inside the KV cache?

At each causal attention layer, a token's hidden state is projected into a query Q, a key K and a value V. The new query scores the keys of visible tokens; the resulting attention weights combine their values:

```math
\operatorname{Attention}(q_t,K_{\leq t},V_{\leq t})
=\operatorname{softmax}\left(\frac{q_tK_{\leq t}^{\mathsf T}}{\sqrt{d_k}}\right)V_{\leq t}.
```

The cache stores **past tokens' K and V tensors at every attention layer**. It does not store final SID scores or model weights. A typical per-layer layout is `[batch, KV_heads, cached_length, head_dim]`, with separate K and V tensors.

Why can they be reused? In causal attention, an old token cannot attend to future tokens. Appending A therefore does not change the old prompt positions' hidden states or their K/V, assuming the same model, positions and masks. Old queries are not needed to compute the new token's attention: the next operation uses the **new Q against the stored K**, then reads stored V.

```python
# Conceptual single-layer incremental attention; omit heads and positions.
q_new = project_q(h_new)
k_new = project_k(h_new)
v_new = project_v(h_new)
K = concatenate(past_K, k_new, axis="sequence")
V = concatenate(past_V, v_new, axis="sequence")
h_attn = softmax(q_new @ K.T / sqrt(head_dim)) @ V
# Apply the layer's remaining operations, then repeat at the next layer.
return h_attn, (K, V)
```

Caching avoids recomputing the old positions through the network. The new position still traverses every layer and attends over history. KV is therefore a trade of memory and cache management for less repeated computation—not an elimination of attention.

### Follow one SID through the cache

Let the prompt contain three tokens `[x0,x1,x2]` and the output be `(A,1)`.

| Operation | Tokens processed now | Cache after the operation | Distribution produced |
| --- | --- | --- | --- |
| Prefill | x0, x1, x2 | K/V for x0, x1, x2 | L1: A, B, C |
| Select A | None; selector only | Still the three prompt tokens | No new model distribution |
| Decode A | A | Prompt K/V plus A's K/V | L2: 1, 2, 3 conditioned on A |
| Select 1 | None; selector only | Still prompt + A | Complete SID A1 is ready |

For a fixed two-token SID, there is no reason to run another forward on 1 merely to return A1. If there were a third SID level, 1 would enter that next forward and be appended to the cache then.

### Follow several beams through the cache

After selecting A and B, both branches start from the prompt cache. Processing A produces `KV(prompt,A)`; processing B produces `KV(prompt,B)`. Suppose the next selection keeps A1, A2 and B1:

For the two-level model, these are final results. The following pseudocode shows how their caches would branch only if generation were extended to a third SID level:

```python
if another_sid_level_is_needed:
    # Each surviving path continues from its own parent's history.
    logits_A1, cache_A1 = model.decode_one(1, fork(cache_A))
    logits_A2, cache_A2 = model.decode_one(2, fork(cache_A))
    logits_B1, cache_B1 = model.decode_one(1, fork(cache_B))
```

A1 and A2 both start from the history of A, then process different new tokens. Here `fork` means giving each branch its own continuation; an implementation may copy the old cache or share its stored history. The three forwards are written separately to show which history each uses; they can run together as a batch.

Ordinary copied caches are enough to implement this branching. Paged KV storage becomes a consideration when duplicating and managing history consumes too much memory or time. It is an optional serving optimization, not a requirement for beam search.

### When Is KV Cache Worth It?

KV caching is a runtime optimization to measure on this same Transformer. With short prompts, a small model and only one continuation step, creating, expanding and accessing caches may offset the savings from avoiding prompt recomputation. Prompt length, beam width and cache storage strategy all affect that tradeoff.

First check that the two paths produce matching conditional distributions and cumulative SID scores within numerical tolerance, with equivalent masks, positions, precision and evaluation settings. Check selected SIDs as well, accounting for ties and small numerical differences near the selection boundary. Then compare execution time, memory use, throughput and tail latency. Section 9 separates this cache comparison from server and batching choices.

## 3. Forward produces logits; decoding chooses paths

The first Transformer forward produces logits for S1. In the cached path, that forward also retains the prompt's K/V. Selecting S1 does not yet append its K/V; feeding it through the second forward does.

For a two-level SID, the cached execution is:

```text
Forward on prompt → scores for A, B, C
Select A and B
Forward on A with prompt cache → scores for A1, A2, A3
Forward on B with prompt cache → scores for B1, B2, B3
Compare complete SID scores → select final candidates
```

The two branch forwards shown above can be executed together as one batch. Full recomputation supplies the same conditional distributions by forwarding the complete prompt plus each selected token, as in Section 2.

The model scores possible continuations. The decoding method decides which paths survive. Changing that selection rule is how beam search, sampling and diversity-aware decoding produce different candidate sets from the same model. The selector examples below work at this distribution boundary; they leave the choice of cached or full model execution to the runtime.

After selecting the final SID token, no additional forward is needed just to return that SID.

## 4. A small SID tree for comparing selectors

I use the same synthetic distribution throughout the runnable example:

```text
root
├─ A  0.70 → 1:0.40  2:0.35  3:0.25
├─ B  0.20 → 1:0.70  2:0.20  3:0.10
└─ C  0.10 → 1:0.80  2:0.15  3:0.05
```

For example, `P(A1)=0.28`, `P(A2)=0.245`, `P(A3)=0.175`, `P(B1)=0.14`, and `P(C1)=0.08`.

The model score is the sum of conditional log probabilities:

```math
S(y)=\sum_{\ell=1}^{L}\log P(s_\ell\mid x,s_{<\ell}).
```

A decoder may use a different score to choose candidates. I keep that **selection score** separate from the raw model score. Neither should automatically be interpreted as a calibrated probability of user intent.

| Method | Selection rule | What it does for diversity |
| --- | --- | --- |
| Greedy | Highest-probability token at each step | Produces one deterministic path |
| Beam | Highest cumulative scores among surviving expansions | Keeps multiple paths, possibly under one prefix |
| Ancestral sampling | Draw from each conditional distribution | Explores randomly; complete SIDs can repeat |
| Temperature | Reshape the token distribution before sampling | Changes exploration, without explicit coverage control |
| Top-k sampling | Sample among the k most probable tokens | Excludes the remaining tokens at that step |
| Top-p sampling | Sample from a probability-mass nucleus | Adapts the allowed token count to the distribution |
| Diverse beam | Penalize redundancy during selection | Depends on the chosen redundancy measure |
| Stochastic beam | Conditional Gumbel tree search | Samples complete sequences without replacement |
| Constrained decoding | Restrict valid continuations | Ensures catalog validity, not diversity |
| Pool + MMR | Select a less redundant final set | Can only use intents already present in the pool |

## 5. Greedy, beam and independent sampling

### Greedy and beam search

Greedy picks A and then 1. In this example A1 is also the most probable complete SID; that is not guaranteed in general.

Beam width three initially retains A, B and C. At L2, it compares all surviving expansions and returns A1, A2 and A3. The first level's spread has disappeared.

```python
beams = [((), 0.0)]
for level in range(depth):
    candidates = []
    for prefix, raw_score in beams:
        for token, logp in model.next_log_probs(prefix).items():
            candidates.append((prefix + (token,), raw_score + logp))
    beams = top_w_by_score(candidates)
```

Beam is approximate search: earlier pruning is irreversible. It does not guarantee the global top-W complete sequences, and a high-probability list does not guarantee broad intent coverage.

### Temperature, top-k and top-p

Ancestral sampling draws one token per step from the model distribution. Temperature changes that distribution:

```math
P_T(v)=\operatorname{softmax}(z_v/T),\qquad T>0.
```

T=1 preserves the distribution; larger T flattens it. Top-k retains a fixed number of tokens. [Nucleus / top-p sampling](https://arxiv.org/abs/1904.09751) retains the smallest probability-sorted set whose cumulative mass reaches p.

At the root, k=2 retains A and B. With p=0.8, the nucleus is also A and B, with total mass 0.9; their sampling probabilities become 7/9 and 2/9. C cannot be sampled at that step.

```python
prefix = ()
for level in range(depth):
    probs = temperature_scale(model.next_probs(prefix), T)
    probs = optional_top_k(probs, k)
    probs = optional_top_p(probs, p)
    token = categorical_sample(normalize(probs))
    prefix += (token,)
```

One run produces one complete SID. Multiple outputs require multiple rollouts, which may be batched. Repeated outputs are allowed. Temperature and truncation interact; when combining filters, their order and normalization convention must be explicit.

Independent sampling and beam can both batch model forwards. Sampling paths evolve independently; beam paths compete within a request and may require cache reordering. Neither algorithm name alone tells me which serving implementation is faster.

## 6. Make diversity part of candidate selection

### Sibling rank and grouped beam

[Li, Monroe and Jurafsky's diverse decoding](https://arxiv.org/abs/1611.08562) penalizes lower-ranked children of the same parent. The runnable teaching variant uses a current-step penalty:

```python
raw = parent_raw + child_log_probability
priority = raw - gamma * sibling_rank
```

With zero-based ranks and `gamma=1.5`, A2 is penalized enough for B1 and C1 to receive slots. The resulting set is A1, B1, C1. The example carries raw scores into the next level; it states this convention explicitly rather than treating every diverse decoder as having identical score propagation.

[Grouped Diverse Beam Search](https://arxiv.org/abs/1610.02424) splits beams into groups and diversifies later groups against earlier selections. The minimal code uses one beam per group and a same-position token penalty. For SID, this proxy deserves scrutiny: A1 and B1 share token 1 but may represent different semantic regions.

### Prefix-aware selection

A tree-aware alternative penalizes overlap between parent prefixes. At each level, select one candidate at a time:

```math
\begin{aligned}
\operatorname{priority}(c)
&= S_{\mathrm{raw}}(c) \\
&\quad - \lambda \sum_{b\in B_{\mathrm{selected}}}
\operatorname{LCP}\bigl(\operatorname{parent}(c),\operatorname{parent}(b)\bigr).
\end{aligned}
```

```python
selected = []
while candidates and len(selected) < width:
    best = argmax(
        candidates,
        key=lambda c: c.raw_score - lam * sum(
            lcp(c.parent, b.parent) for b in selected
        ),
    )
    selected.append(best)
    candidates.remove(best)
```

Selecting one child changes the remaining candidates' priorities. Raw probabilities remain intact. At the initial level, every parent prefix is empty, so this particular penalty is zero.

This gives me a concrete diversity mechanism, but only a proxy for intent coverage. Different prefixes can still mean similar things, and an irrelevant branch is not useful merely because it is different.

## 7. Stochastic beam: sample complete SIDs without replacement

[Stochastic Beams and Where To Find Them](https://proceedings.mlr.press/v97/kool19a.html) provides a different connection between search and sampling. Start with the complete-sequence view:

```math
G_y=-\log(-\log U_y),\quad U_y\sim\operatorname{Uniform}(0,1),\qquad
\operatorname{TopK}_y[\log P(y)+G_y].
```

Independent Gumbel perturbations over the complete SIDs yield probability-weighted sampling without replacement. This prevents exact SID duplicates; it does not force the selected SIDs into different L1 regions.

The toy tree has only nine leaves, so enumerating them is a useful reference implementation. Real sequence spaces are too large. Stochastic beam instead propagates **conditioned subtree maxima** through the tree:

```python
beams = initialize_root_with_gumbel_maximum()
for level in range(depth):
    children = expand(beams)
    draw_child_gumbels_from_subtree_log_masses(children)
    condition_child_maxima_on_their_parent_maxima(children)
    beams = top_w_by_conditioned_maximum(children)
```

The conditioning step is essential. Independently adding noise at every step and applying ordinary pruning does not implement the same complete-sequence sampling distribution.

The downloadable code includes both the exhaustive reference and the tree algorithm. Their outputs need not match under the same seed because they consume random numbers differently. In this normalized fixed-depth tree, a prefix's probability equals its subtree mass. Adding catalog constraints requires revisiting that assumption.

## 8. Validity and final-set selection

Suppose the catalog contains only A1, A2, B1 and C1. A3 has high model probability but is not a valid output. A Trie can remove invalid continuations before selection:

```python
allowed = trie.allowed_children(prefix)
log_probs = {t: original_log_probs[t] for t in allowed}
```

For beam, filtering can preserve the original scores. Sampling needs normalized weights over permitted tokens. Local renormalization generally differs from conditioning the original complete-sequence distribution on catalog membership: the latter depends on valid probability mass deeper in each subtree.

After generation, MMR can select a less redundant set from a larger pool:

```math
q^*=\arg\max_{q\notin S}\left[
\operatorname{Rel}(q,x)-\lambda\max_{s\in S}\operatorname{Sim}(q,s)
\right].
```

The code uses raw log probability and LCP as simple relevance and redundancy proxies. Actual query recommendation needs appropriate score scales and semantic validation. If the pool contains only A1, A2 and A3, no reranker can recover B1.

This is why I want separate measurements for candidate generation and final-set selection: valid unique SIDs, distinct prefixes, relevant intent coverage, and recall at a fixed final budget.

## 9. Do I need Triton Inference Server, or is the current Torch service enough?

**The model architecture is fixed: the autoregressive Transformer in Section 2.** The serving question is whether the current service can run its retrieval loop at the required throughput, p99 latency and cost. Full recomputation is the execution baseline; KV-cached decoding is the optimization to evaluate with the same model parameters.

The server handles requests, queues and workers. The Transformer runtime handles forwards, beam expansion and any KV cache. Both execution paths can run within an existing Torch-based service. A small HTTP wrapper built for an experiment cannot stand in for that service when evaluating migration.

### Baseline serving boundary: a complete SID retrieval call

For the fixed two-level output, a complete retrieval call can select either execution path from Section 2:

```python
# Pseudocode inside the current service, after assembling a request batch.
def retrieve_sid_batch(prompt, model, width, k, use_kv_cache=False):
    if use_kv_cache:
        return cached_sid(prompt, model, W=width, K=k)
    return full_recompute_sid(prompt, model, W=width, K=k)
```

Both functions receive the same model object and use the same probability and selection conventions. Each returns complete SIDs with cumulative scores. The selected SIDs are then validated and resolved to catalog candidates, with downstream ranking if needed.

In this baseline design, cached state lives inside the retrieval call and can be released when generation completes. L1 and L2 do not need separate client requests. Continuous batching changes the internal execution boundary, as described below; a client can still submit one retrieval request and receive one final result.

B counts requests and W counts hypotheses within each request. Both paths process B×W branches in the second forward. Full recomputation processes T+1 token positions per branch, while cached decoding processes one new position per branch and reads the prompt's cached K/V. Increasing W can change memory use and latency without changing the incoming request count.

### What would justify introducing Triton Server?

[NVIDIA Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html) offers request scheduling and dynamic batching. [TorchServe](https://docs.pytorch.org/serve/batch_inference_with_ts.html) also supports batching incoming requests, provided the handler performs batched computation. Request-level batching alone is therefore insufficient reason to choose one over the other.

For this note, the useful decision is whether those server capabilities solve a problem the current service actually has:

| What I observe | What I would do first | When Triton Server becomes worth evaluating |
| --- | --- | --- |
| Throughput and p99 meet the target, with acceptable cost | Keep the current Torch service and record the baseline | When requirements or operational needs change |
| Requests execute in small batches despite enough compatible traffic | Inspect the current batch assembly and scheduling policy | When a managed batching/scheduling layer would address a demonstrated gap |
| Queue time grows under load | Separate batching delay from insufficient execution capacity | When scheduling or admission behavior is the bottleneck; a saturated GPU still needs capacity or less work |
| The forward or SID selection itself dominates execution time | Profile and improve that Torch computation | A server change alone does not address this bottleneck |
| CPU launch gaps dominate a stable execution path | Evaluate compilation or CUDA Graph within the existing service | These execution changes can be tested before changing the server |
| Operating model versions and concurrent model instances becomes difficult | Identify the specific operational requirement | When a standardized serving layer offers enough operational value to justify migration |

These are proposed decision criteria, not benchmark results. Free GPU memory and utilization averages do not establish how much request capacity remains. I need throughput measured under the latency target, including queueing.

### If I do adopt it, where does the decoding loop go?

For a whole-request baseline, the backend runs both Transformer forwards and the candidate selections inside one invocation, then returns complete SIDs. The cache flag controls the execution strategy.

The Python backend's `execute(requests)` receives requests; the implementation assembles compatible input tensors, executes the batched model and returns the corresponding responses. Merely looping over requests and calling the model once per request does not make the GPU computation batched. [Triton Python backend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html).

```python
# Conceptual backend structure, not a complete Triton deployment.
def execute(requests):
    prompt, request_slices = assemble_compatible_inputs(requests)
    candidates = retrieve_sid_batch(
        prompt, model, width, k, use_kv_cache=use_kv_cache
    )
    return make_responses(candidates, request_slices)
```

This preserves the important ownership boundary: the backend/model code manages beam parents, diversity scores and local KV state. Triton's request batcher does not infer those operations from the model weights.

### Dynamic batching and continuous batching solve different scheduling problems

| Scheduling policy | When requests are grouped | What happens during generation |
| --- | --- | --- |
| Whole-request dynamic batching | Before a complete retrieval call | The selected batch runs through its SID generation together |
| Continuous / inflight batching | At generation-step boundaries | Finished requests leave; eligible new requests can join while others continue |

Dynamic batching collects compatible requests arriving at different times so the model can process them together. An optional wait can improve batch fill while adding latency. It does not automatically admit new requests into an ongoing decoding loop.

Continuous batching requires a runtime that preserves each request's decoding position, beam scores and, when used, KV state between executions. Prefill also consumes compute and must be scheduled. Triton's iterative-sequence mechanism allows a cooperating backend to reschedule unfinished requests. TorchServe also has continuous-batching support paths; the handler or integrated engine must implement the corresponding generation behavior. These capabilities require more than changing a batch-size setting. [Triton iterative sequences](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html#iterative-sequences), [TorchServe release notes](https://github.com/pytorch/serve/releases).

An autoregressive loop within one backend invocation does not itself require sequence batching. If execution is split across scheduling iterations, state ownership and lifetime must be explicit. Each request's beam candidates still compete only with one another, even when model forwards are batched across requests.

Fixed two-token outputs leave fewer opportunities for early completion than long, variable-length outputs. That can limit the additional benefit of continuous batching, but input length alone does not settle the question. Request arrivals, prefill cost, beam width and generation duration still need measurement.

### Eager, TorchScript and CUDA Graph describe how computation executes

These mechanisms are separate from the choice to recompute or cache prompt history:

| Mechanism | What changes |
| --- | --- |
| PyTorch eager | Python executes the model and dispatches its operations as it runs. |
| [TorchScript](https://docs.pytorch.org/docs/2.10/jit.html) | A compatible model is converted to a serializable executable program. |
| [CUDA Graph](https://docs.pytorch.org/docs/2.10/notes/cuda.html#cuda-graphs) | A compatible sequence of GPU operations is captured and replayed, reducing repeated CPU dispatch and launch work. |

CUDA Graph normally preserves the kernels rather than rewriting or fusing them. Each replay computes on new input data. For the same part of a model, a request can use ordinary execution or graph replay; a larger system can use both for different paths. A graph can also capture compatible work produced by a TorchScript model. The linked PyTorch documentation marks TorchScript as deprecated; it remains useful to distinguish it from server scheduling and graph replay when discussing an existing deployment.

```python
# Conceptual pseudocode: capture once, then run new requests.
# Setup includes warm-up and compatible fixed shapes / memory addresses.
graph, output_buffer = capture_gpu_work(model, input_buffer)

def run_request(new_input):
    input_buffer.copy_(new_input)
    graph.replay()
    return output_buffer  # Consume or copy before this buffer is reused.
```

The primary execution comparison remains full recomputation versus KV-cached decoding. Keep the execution mechanism fixed while measuring caching. If CUDA Graph or another execution optimization is evaluated later, measure that change separately on the same Transformer path.

### Evaluation plan: the same Transformer throughout

**Transformer measurements are pending.** The runnable probability-tree example in Section 10 checks selection algorithms; it does not measure Transformer execution or serving performance.

The first experiment compares the two paths from Section 2 with the same checkpoint, model configuration, precision, evaluation mode, token masks, positions, final candidate count and decoding rules. Vary prompt length T, request batch size B and beam width W, keeping each pair of measurements matched.

| Question | What to measure |
| --- | --- |
| Do both paths implement the same conditional predictions? | L1 and L2 log probabilities, cumulative SID scores, selected SIDs and catalog validity; document numerical tolerance and tie handling |
| How much repeated prompt computation does caching avoid? | First and second forward durations separately, plus total retrieval-call time |
| What does cache management cost? | Allocation, beam-cache copying or sharing, cache reads and peak GPU memory, including temporary buffers |
| Does faster model execution improve the service? | End-to-end p50/p95/p99, queue time and throughput under the latency target |
| Does the scheduling policy help this workload? | Actual batch sizes and request arrivals; for continuous batching, evidence that new requests join before other active requests finish |

Time complete retrieval calls as well as the individual forwards. State whether inputs start on the GPU and whether transfers, cache creation, selection and synchronization are included. Separate GPU execution time from wall-clock time, which also includes CPU dispatch. Include warm-up and repeated measurements; report compilation or graph-capture setup separately if those mechanisms are used.

After establishing this model baseline, compare the existing service with a Triton-hosted version of the **same Transformer runtime**. Match hardware, precision, cache mode, decoding rules, workload and measurement boundaries. Name the server integration and backend: a Python integration and a native model backend are different execution paths, so the backend must remain visible in the comparison.

End-to-end measurements should include request handling, queueing, transfers, generation, response processing, and any SID resolution or ranking inside the measured boundary. Report actual batch sizes, worker counts, batching waits and the arrival pattern. Equal maximum-batch settings do not guarantee equal executed batches.

Evaluate continuous batching separately from the whole-request baseline. The workload must include staggered arrivals and record admission and completion events, so the measurement demonstrates step-level scheduling rather than merely concurrent HTTP requests. The model's fixed SID depth and candidate-selection behavior should remain unchanged.

If the current service meets the target, a migration needs an additional operational or performance benefit. If it misses the target, identify whether queueing, repeated prompt computation, cache management, selection, transfers or CPU launches dominate before choosing an optimization. Deployment and maintenance cost belong in that decision alongside latency and throughput.

## 10. Executable check: one tree, several decoding methods

[Download the Python example](/assets/code/sid_decoding_minimal.py). It uses only the Python 3 standard library:

```sh
python3 sid_decoding_minimal.py
```

The probability table substitutes for model logits. The selectors themselves execute; this example makes no GPU-performance claim.

```text
greedy [('A1', 0.28)]
beam [('A1', 0.28), ('A2', 0.245), ('A3', 0.175)]
sibling [('A1', 0.28), ('B1', 0.14), ('C1', 0.08)]
grouped [('A1', 0.28), ('B2', 0.04), ('A3', 0.175)]
prefix [('A1', 0.28), ('B1', 0.14), ('C1', 0.08)]
catalog [('A1', 0.28), ('A2', 0.245), ('B1', 0.14)]
```

The script also prints independent sampling, Gumbel-Top-k, stochastic beam and pool-plus-MMR results. The displayed probabilities are raw path probabilities even when selection uses a different score.

I can change gamma or lambda to see which branch gets a slot, change top-k or top-p to see which branches become unreachable, and compare repeated sampling with sampling without replacement. These checks establish what the algorithms do on this tree. Whether they broaden useful intent coverage remains a downstream evaluation question.

## References

- Li, Monroe & Jurafsky (2016). [A Simple, Fast Diverse Decoding Algorithm for Neural Generation](https://arxiv.org/abs/1611.08562).
- Vijayakumar et al. (AAAI 2018). [Diverse Beam Search](https://arxiv.org/abs/1610.02424).
- Holtzman et al. (ICLR 2020). [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751).
- Kool, van Hoof & Welling (ICML 2019). [Stochastic Beams and Where To Find Them](https://proceedings.mlr.press/v97/kool19a.html).
