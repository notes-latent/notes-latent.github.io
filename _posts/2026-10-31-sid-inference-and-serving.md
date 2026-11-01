---
layout: post
math: true
title: "SID Inference and Serving"
description: "From two-level SID prediction to decoding, candidate diversity, and GPU serving: a research note with pseudocode and executable examples."
---

I have been spending time on SID-based generative retrieval, first thinking about how to construct the IDs and then about what a retrieved set should cover. More recently, I started wondering what happens when this model has to serve requests. Can the GPU cluster that serves an ordinary prediction model also serve a generative retriever? What changes when a request produces several candidate paths rather than one prediction?

That question connects to another one: how do I make the generated SIDs more diverse, and which decoding choices might broaden relevant intent coverage? Before choosing a serving framework or a diversity objective, I wanted to understand exactly what runs between the input and the final candidate set.

This note follows that execution path: **model forward → candidate selection → another forward, if needed → SID lookup and ranking**. I use familiar LLM decoding methods as a starting point, then check which assumptions actually apply to a short, structured SID output. Pseudocode makes the boundaries explicit; a small executable SID example lets me check what each selector really does.

**Status:** Study note · Executable decoding examples and a small GPU execution check; production serving remains to be validated.

Related notes: [Constructing good SIDs](/notes/constructing-good-semantic-ids-for-llm-prediction/) · [Intent-aware retrieval sets](/notes/from-semantic-ids-to-intent-aware-retrieval-sets/)

## 1. Start with the execution path

```text
Request arrives
    ↓
Queue and batch requests                 serving scheduler
    ↓
Encode input / prefill                   model computation
    ↓
Next-token logits
    ↓
Select parent paths and SID tokens       decoding algorithm
    ↓
Conditional head OR another forward      model computation
    ↓
Resolve SIDs → rank candidates → respond
```

The GPU executes the tensor operations in this path. The serving system decides which requests run together and when. Beam search decides which hypotheses survive. These decisions interact, but they belong to different layers.

The first question is therefore more concrete than “does this model generate SIDs?”:

> **After selecting the first SID token, does the second level run a lightweight conditional head, or another Transformer forward?**

Both architectures can represent the same factorization:

```math
P(s_1,s_2\mid x)=P(s_1\mid x)P(s_2\mid x,s_1).
```

They can have very different execution costs. Both can generate several candidate paths with beam search. Branching alone does not favor one architecture; retrieval quality and serving cost need to be evaluated. Two SID levels do not imply two Transformer layers, beam width two, or a particular KV-cache requirement.

## 2. Two ways to produce a two-level SID

### Encode once, then use conditional heads

The input is encoded once. A first head predicts L1; a second head takes the shared representation and each selected L1 token to predict L2.

```python
# Pseudocode for ONE request. Keep the actual token IDs with every score.
# top_tokens returns (token_id, log_probability) pairs.
h = encoder(x)
first_choices = top_tokens(log_softmax(head1(h)), width=W)
candidates = []

for first_token, first_logp in first_choices:
    second_logps = log_softmax(head2(h, embed(first_token)))
    for second_token, second_logp in enumerate(second_logps):
        candidates.append({
            "sid": (first_token, second_token),
            "score": first_logp + second_logp,
        })

results = sorted(candidates, key=lambda c: c["score"], reverse=True)[:W]
```

For example, keeping A and B at L1 gives two branches. The second head scores continuations under each branch: A → A1, A2, A3; B → B1, B2, B3. We compare the cumulative scores of those complete SIDs and keep the best W. Several winners may come from A.

There is no repeated autoregressive history attention to cache in this design. The encoder may itself use attention, but running it once is a different problem from incrementally extending a causal Transformer.

### Autoregressive Transformer

Here, the selected L1 token becomes input to another Transformer forward. The straightforward version reruns the prompt with that token appended. The cached version reuses the prompt's historical K/V.

```python
# Full recomputation
logits1 = model(prompt).last_logits
sid1 = select_l1(logits1, W)
logits2 = model(append_to_each_branch(prompt, sid1)).last_logits

# Incremental computation
logits1, prompt_kv = model.prefill(prompt)
sid1 = select_l1(logits1, W)
beam_kv = fork_prompt_cache(prompt_kv, W)
logits2, beam_kv = model.decode_one(sid1, beam_kv)
```

With equivalent positions, masks and evaluation settings, cached and full computation should give matching next-token distributions within numerical tolerance. That is a correctness check before it is a performance comparison.

A short output does not settle whether caching is worthwhile. Prompt length, beam width, memory traffic and runtime overhead all matter. Comparing an encode-once model with a different autoregressive model also does not isolate a cache optimization: their architectures and learned behavior differ.

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

```python
if another_sid_level_is_needed:
    # Each surviving path continues from its own parent's history.
    logits_A1, cache_A1 = model.decode_one(1, fork(cache_A))
    logits_A2, cache_A2 = model.decode_one(2, fork(cache_A))
    logits_B1, cache_B1 = model.decode_one(1, fork(cache_B))
```

A1 and A2 both start from the history of A, then process different new tokens. Here `fork` means giving each branch its own continuation; an implementation may copy the old cache or share its stored history. The three forwards are written separately to show which history each uses; they can run together as a batch.

Ordinary copied caches are enough to implement this branching. Paged KV storage becomes a consideration when duplicating and managing history consumes too much memory or time. It is an optional serving optimization, not a requirement for beam search.

This also explains when the cache discussion applies: an encode-once conditional head has no repeated causal history computation to eliminate. An autoregressive implementation does; whether caching wins at the actual prompt length and beam width should be measured.

## 3. Forward produces logits; decoding chooses paths

For an autoregressive decoder, prefill produces the distribution for the first generated token. Selecting that token does not yet append its K/V. Feeding it through the next forward does.

For a two-level SID, the sequence is:

```text
Forward on prompt → scores for A, B, C
Select A and B
Forward on A with prompt cache → scores for A1, A2, A3
Forward on B with prompt cache → scores for B1, B2, B3
Compare complete SID scores → select final candidates
```

The model scores possible continuations. The decoding method decides which paths survive. Changing that selection rule is how beam search, sampling and diversity-aware decoding produce different candidate sets from the same model.

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

**The starting point is the actual inference service: its request handling, batching, workers and model runtime.** Generating SIDs alone does not establish a need to replace it. The question is whether that service can execute the retrieval loop at the required throughput, p99 latency and cost. A small HTTP wrapper built for an experiment cannot stand in for the existing deployment when making this decision.

The earlier sections make this decision more concrete. An encode-once model runs its encoder and conditional heads. An autoregressive model runs prefill, candidate selection and the subsequent decode step, with a KV cache if useful. Both can be implemented in Torch and kept inside one request. Beam search, prefix penalties and cache reordering do not require a new inference server.

### Keep the whole SID computation inside one request

For a fixed two-level output, I would begin with this execution boundary:

```python
# Pseudocode inside the current service, after assembling a request batch.
def retrieve_sid_batch(x, model, width, architecture):
    if architecture == "encode_once":
        h = model.encode(x)
        first_logits = model.head1(h)
        sid1, score1 = select_first_level(first_logits, width)
        second_logits = model.head2(h, sid1)
    else:
        first_logits, prompt_kv = model.prefill(x)
        sid1, score1 = select_first_level(first_logits, width)
        beam_kv = fork_prompt_cache(prompt_kv, width)
        second_logits, beam_kv = model.decode_one(sid1, beam_kv)

    # Compare candidates separately within each request, not across users.
    return select_sid_pairs(sid1, score1, second_logits, width)
```

In this design, the serving layer owns request handling and batching; the model code owns this computation. In the autoregressive branch, the KV cache is temporary state inside this execution. It can be released when the SID results are complete. There is no need to turn L1 and L2 into separate client requests merely because the model is autoregressive.

B counts requests and W counts hypotheses within each request. The second autoregressive forward may process B×W rows, while the service still handles B requests. Increasing W can therefore change memory and latency substantially without changing the number of incoming requests.

### What would justify introducing Triton Server?

[NVIDIA Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html) offers request scheduling and dynamic batching. Its batcher can group compatible requests up to the configured maximum and optionally wait for more requests. That wait trades latency for batch fill; it is not a guaranteed throughput improvement at the same p99.

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

For this short, fixed-depth SID model, keep the entire decoding loop inside one backend invocation. Triton schedules requests; the backend runs encoding or prefill, candidate selection and subsequent model forwards, then returns complete SIDs.

The Python backend's `execute(requests)` receives requests; the implementation assembles compatible input tensors, executes the batched model and returns the corresponding responses. Merely looping over requests and calling the model once per request does not make the GPU computation batched. [Triton Python backend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html).

```python
# Conceptual backend structure, not a complete Triton deployment.
def execute(requests):
    x, request_slices = assemble_compatible_inputs(requests)
    candidates = retrieve_sid_batch(x, model, width, architecture)
    return make_responses(candidates, request_slices)
```

This preserves the important ownership boundary: the backend/model code manages beam parents, diversity scores and local KV state. Triton's request batcher does not infer those operations from the model weights.

An autoregressive loop contained within one request does not by itself require the sequence batcher. Sequence batching addresses stateful sequences of requests that need routing to the same model instance. Keeping internal decode state across separate executions requires explicit lifecycle and scheduling design. [Triton batching documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html).

Ordinary request-level dynamic batching also does not automatically insert new requests into an already-running decoding loop. If longer, variable-length generation eventually makes iteration-level scheduling important, that becomes a separate backend/engine evaluation. For the short fixed-depth path here, I would first measure the simpler whole-request execution.

### The experiment that would change my decision

I would compare the existing service with a Triton-hosted version of the **same SID computation**, using the same hardware, precision, candidate width and request trace. The comparison should include:

- End-to-end p50/p95/p99 and queue time at representative and peak arrivals.
- Actual batch sizes, throughput at the latency target, and memory use.
- SID scores and selected candidates, so the performance comparison preserves behavior.
- Deployment and maintenance cost, including backend integration and error handling.

If the current service already meets the target, introducing another serving layer needs a concrete benefit. If it misses the target, I first want to name the dominant term—queueing, model execution, selection, transfer or launch overhead—and explain why the proposed change reduces it. **The next step is a measurement of the current Torch path, not a migration triggered by the word “generative.”**

### Model execution check: eager, TorchScript and CUDA Graph

These names describe model execution, while Triton Server describes a serving system:

| Mechanism | What changes |
| --- | --- |
| PyTorch eager | Python executes the model and dispatches its operations as it runs. |
| [TorchScript](https://docs.pytorch.org/docs/2.10/jit.html) | The model is converted to a serializable executable program; this test uses `torch.jit.script` followed by `freeze`. |
| [CUDA Graph](https://docs.pytorch.org/docs/2.10/notes/cuda.html#cuda-graphs) | A compatible sequence of GPU operations is captured and replayed, reducing repeated CPU dispatch and launch work. |

CUDA Graph normally preserves the kernels rather than rewriting or fusing them. Each replay computes on new input data. For the same part of a model, a request can use ordinary execution or graph replay; a larger system can use both for different paths. A graph can also capture compatible work produced by a TorchScript model. TorchScript is deprecated in the PyTorch version used here; it is included to examine an existing deployment path.

```python
# Conceptual pseudocode: capture once, then run new requests.
# Setup includes warm-up and compatible fixed shapes / memory addresses.
graph, output_buffer = capture_gpu_work(model, input_buffer)

def run_request(new_input):
    input_buffer.copy_(new_input)
    graph.replay()
    return output_buffer  # Consume or copy before this buffer is reused.
```

A synthetic encode-once SID model was measured on the same single CUDA GPU with fixed weights, FP32, 45-token inputs, two SID levels and beam width 16. It uses pooled embeddings, a shared representation and two conditional heads, totaling about 10,000 parameters. The execution comparison used PyTorch 2.10.0 and separate CUDA Graph captures for batch sizes one and eight.

The table reports the median of three rounds' p50 model-call times. Each round used 50 warm-up calls and 200 timed calls; execution order rotated between rounds. Inputs were already on the GPU. Wall-clock timing included synchronization after each call, and the graph path included copying input into its persistent GPU buffer. Compilation and capture setup were outside the timed calls.

| Execution path | Batch 1 (ms/call) | Batch 8 (ms/call) |
| --- | --- | --- |
| PyTorch eager | 0.242 | 0.235 |
| TorchScript: script + freeze | 0.204 | 0.198 |
| CUDA Graph captured from eager | 0.066 | 0.069 |

For each batch size, 16 input examples produced identical selected SIDs across the three paths; scores passed checks with absolute and relative tolerances of `2e-5`. These are whole-call timings, including CPU submission overhead, rather than isolated GPU kernel durations. They exclude HTTP/RPC, host-to-device input transfer, response serialization and request queueing.

CUDA Graph reduced model-call time in this small fixed-shape example. The measurement does not establish end-to-end service capacity or favor one inference server over another. That comparison still requires the actual serving stack, equivalent model execution and a representative request load.

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
