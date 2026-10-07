---
layout: post
title: "LLM-enhanced Recommendation: From Sequence Encoding to Grounded Generation"
description: "A working mental model for using LLMs as context encoders and grounded generators, with notes on retrieval, serving cost, and behavioral alignment."
---

## 1. High-level observation

Reading public work on sequential recommendation, generative recommendation, and language-model-based retrieval and ranking, I find two useful directions:

1. **Use an LLM as a rich context / sequence encoder**, while keeping expensive candidate-by-candidate scoring outside the LLM.
2. **Use an LLM for grounded generation or target-conditioned understanding**, then connect its output back to retrieval, ranking, or direct recommendation.

My working interpretation is that language models are useful where **semantic understanding and flexible context modeling matter**, while efficient recommendation components can handle **large-scale scoring and serving**. HLLM illustrates language-model-based item and user modeling; TIGER illustrates recommendation through generated item identifiers. These are different architectures, rather than implementations of one universal recipe. [HLLM](https://arxiv.org/abs/2409.12740), [TIGER](https://arxiv.org/abs/2305.05065).

This article is a synthesis of public research and ideas I want to explore. The two directions are a mental model, not an exhaustive taxonomy or a claim that the entire industry is converging on one design. A generative recommender also need not be a pretrained natural-language LLM: TIGER, for example, trains a sequence-to-sequence model over semantic item IDs.

## 2. Direction I: LLM as a context and sequence encoder

### 2.1 Motivation

A straightforward idea is to feed user history, the current context, and every candidate item into an LLM and ask it to score the candidates. For a ranking stage with hundreds or thousands of candidates, repeated candidate-conditioned inference can become expensive.

A practical decomposition is:

**User history / context → LLM → user or context representation → lightweight downstream scorer → candidate score**

The expensive computation happens mainly on the context side and is amortized across candidates. This is an architectural option, not a guarantee of cheap serving: context length, model size, update frequency, caching, and latency still matter.

HLLM is a concrete related example: an Item LLM extracts content representations, and a User LLM models the sequence of those representations to predict future interests. Its inputs are not simply a free-form textual transcript of user behavior. [HLLM](https://arxiv.org/abs/2409.12740).

### 2.2 From feature engineering to context engineering

Classic sequential recommenders such as GRU4Rec, SASRec, and BERT4Rec learn from interaction sequences, commonly represented as item IDs:

**Item ID → item ID → item ID → …**

They use different sequence-learning mechanisms: recurrent modeling, causal self-attention, and bidirectional masked-item prediction, respectively. [GRU4Rec](https://arxiv.org/abs/1511.06939), [SASRec](https://arxiv.org/abs/1808.09781), [BERT4Rec](https://arxiv.org/abs/1904.06690).

In a broader recommendation system, additional features might include engagement type, dwell time, click / save / purchase, category, timestamp, and session information. Language models provide another way to combine heterogeneous information. P5, for example, expresses recommendation tasks and their inputs in a shared text-to-text format. [P5](https://arxiv.org/abs/2203.13366).

A hypothetical textual history could look like this:

> User searched for X, viewed Y for 40 seconds, saved Z, skipped A, and recently engaged with several items related to B.

Here, the duration is an illustrative value, not a reported production metric. The LLM could encode such a sequence into a latent representation.

The shift I find useful is:

- **Traditional sequential recommendation:** substantial effort goes into sequence architecture and input features.
- **LLM-based sequence modeling:** more of the iteration may move toward context construction, event selection, compression, and representation format.

Once a useful base encoder exists, some changes to the history representation may not require redesigning the architecture. This is my interpretation of a design opportunity; it does not mean feature schemas, recommendation-specific training, or model architecture cease to matter.

### 2.3 Why this is useful

Textual or structured semantic representations can make heterogeneous signals easier to combine. Search queries, item metadata, actions, durations, and behavioral summaries can share an input format.

They also create a flexible experimental surface. We can change:

- which events are included;
- how engagement strength and negative feedback are represented;
- how recent versus long-term behavior is summarized;
- whether events are chronological or grouped by intent;
- whether the current task or long-term preference receives more emphasis.

Some experiments can therefore become changes to **context representation** rather than new feature towers. They still need evaluation: natural language can lose numerical precision, consume many tokens, or obscure useful structure. A richer input is not automatically a better input.

### 2.4 LLM compression before downstream ranking

Long histories create context-length and inference-cost problems. One possible pipeline is:

**Raw history → context selection / compression → LLM encoding → compact representation → downstream ranking**

Depending on the architecture and training objective, the compact representation might use a last-token or special-token hidden state, pooling, a learned projection, or a task-specific embedding head. These are design options, not interchangeable choices proven equally effective.

The architectural idea is to separate **expensive semantic understanding** from **cheap candidate scoring**. After the representation is produced, the scorer can use an MLP, a two-tower retrieval model, a ranking network, or candidate interaction features. The representation and downstream scoring objective must be aligned; extracting an arbitrary hidden state is not enough.

Another branch is **context compression into soft tokens**: instead of summarizing a history as readable text, encode it into a small set of continuous vectors that a downstream language model can condition on. AutoCompressors use summary vectors as soft prompts, while ICAE produces compact memory slots. Gist tokens are related, but use special tokens and attention constraints to learn reusable compressed prompt states. These approaches differ in how they train and pass the compressed information to the model. [AutoCompressors](https://arxiv.org/abs/2305.14788), [ICAE](https://arxiv.org/abs/2307.06945), [Gist Tokens](https://arxiv.org/abs/2304.08467).

This differs from producing one user embedding for a conventional ranker: the compressed representation remains part of the language model's conditioning context. Here, “soft” refers to continuous representations, rather than the discrete codewords used in semantic IDs. Not every learned soft prompt compresses an input history, and these representations generally need compatible model interfaces and training; they are not arbitrary vectors that can be inserted into any text-only API.

**In the next note, I plan to take a closer look at soft-token context compression:** what methods are available, how their objectives and interfaces differ, and what information survives compression. Applications I want to explore include reusable user-history memory, query-conditioned interest compression, compressed item or retrieved-document context, and sharing compressed history across query generation and reranking. These are experiment ideas, not results established by the papers above. I would compare them with text summaries and ordinary embeddings on recommendation quality, latency, compression cost, and reuse across tasks.

### 2.5 Another option: generative scoring

A different approach expresses relevance scoring as language modeling:

**Query + context + candidate → language model → relevance-label token probabilities → score**

For example, the output might be “Yes” / “No.” The monoT5 work demonstrates this idea for document ranking, using the probabilities of relevance-label tokens. Its evidence is from information retrieval, rather than personalized recommendation. [Document Ranking with a Pretrained Sequence-to-Sequence Model](https://arxiv.org/abs/2003.06713).

The distinction is:

| Architecture | Computation | Main tradeoff |
| --- | --- | --- |
| Embedding style | Context → representation → downstream scorer | Context encoding can be shared across candidates |
| Candidate-conditioned generative scoring | Context + candidate → token probabilities | Richer interactions, with repeated candidate-conditioned computation |

The second approach is often easier to justify for reranking a small candidate set. Prefix caching and batching can help, but do not remove all candidate-specific cost. A useful score might normalize probabilities over the chosen relevance labels; a raw logit is not automatically a calibrated relevance probability.

## 3. Direction II: LLM as grounded generation

Generation is particularly natural when the output itself is semantic: a recommended query, autocomplete suggestion, rewrite, tag, category, or explanation. GQR directly frames query recommendation as generation; P5 covers multiple recommendation tasks through text-to-text learning. [Generating Query Recommendations via LLMs](https://arxiv.org/abs/2405.19749), [P5](https://arxiv.org/abs/2203.13366).

Discrete item representations provide another action space. TIGER predicts semantic IDs, which resolve to catalog items rather than arbitrary text. [TIGER](https://arxiv.org/abs/2305.05065).

### 3.1 General versus personalized understanding

I find it useful to distinguish two settings.

**General / aggregate-context understanding.** We want to interpret a target query or item without strong user personalization. Possible evidence includes related queries, related items, categories, aggregate activity, or co-engagement patterns.

For example, **“jaguar”** admits several intents. Related searches and clicked-item categories could help narrow the likely interpretation. This is an illustrative design, not an experiment reported here. Aggregate context is also broader than popularity: a frequent intent need not be the right intent for a particular request.

**Personalized understanding.** We additionally condition on user context:

**Target + global context + user history → target-conditioned understanding**

The result can guide downstream recommendation. It also changes serving economics: a representation conditioned on every candidate cannot be computed just once and reused unchanged across the entire candidate set. Conditioning on the current query or task can provide a more manageable middle ground.

### 3.2 Generation does not necessarily mean displaying generated text

Generation can play at least two roles.

**Direct generation:** produce a suggestion, query, tag, or description that can be surfaced after validation.

**Generate-to-retrieve:** produce an intermediate representation and map it back into a controlled space:

**Context → generated intent / concept / representation → retrieve candidates → rank candidates**

The intermediate output might be keywords, categories, structured attributes, graph nodes, or semantic IDs. Those options require different mapping and validation mechanisms; invented categories or invalid identifiers do not become grounded simply because they are structured.

HyDE is a related example from information retrieval: it generates a hypothetical document, encodes it, and retrieves real documents from a corpus. TIGER takes a different route by generating catalog-linked semantic IDs. These illustrate two distinct ways generation can connect to retrieval. [HyDE](https://arxiv.org/abs/2212.10496), [TIGER](https://arxiv.org/abs/2305.05065).

I think of this role as **semantic planning**: flexible generation proposes a direction, while retrieval and validation anchor the result to available content. This description does not imply that the model performs reliable multi-step reasoning.

## 4. Grounded generation may matter more than prompt polishing

My hypothesis is that improving **the evidence a model sees** can matter more than repeatedly polishing descriptive instructions.

A prompt can explain the platform, the recommendation surface, and what a good result should look like. But relevant behavioral evidence, related queries, item metadata, or graph neighborhoods may provide information that instructions alone cannot supply.

RA-GQR is a concrete example: it retrieves similar queries from logs to construct the prompt, and its paper reports improvements over the unaugmented GQR approach on its evaluation collections. That supports the usefulness of retrieved evidence in this setting; it does not establish a universal rule that grounding always beats prompt optimization. [Generating Query Recommendations via LLMs](https://arxiv.org/abs/2405.19749).

The research question I take from this is:

> What information should we retrieve, compress, and present to the model before generation?

This is a recommendation-specific context-engineering problem. Evidence quality, recency, coverage, and availability at decision time all matter. Grounding in biased or irrelevant logs can make the model confidently wrong.

## 5. A possible training pipeline: teacher SFT → exploration → preference optimization

The following is a design sketch assembled from related ideas. I am not claiming that every cited system uses this exact three-stage pipeline.

### Stage 1: Teacher-generated SFT

A powerful teacher can generate candidate outputs or supporting rationales. After filtering and validation, these can supervise a smaller production model:

**Teacher → validated training outputs → SFT dataset → smaller model**

Distilling Step-by-Step shows that LLM-produced rationales can help train smaller models on NLP tasks. It motivates the teacher–student idea, but is not evidence for this entire recommendation pipeline. Teacher outputs are synthetic supervision, not guaranteed high-quality labels. [Distilling Step-by-Step](https://arxiv.org/abs/2305.02301).

### Stage 2: Controlled exploration

The SFT model could generate multiple recommendations. Sampling diversity can broaden the candidate pool, while a separate exposure policy decides what eligible outputs users actually see:

**SFT model → diverse candidates → validation / exposure policy → logged interactions**

Higher temperature alone is not a well-defined exploration policy. The system needs controlled exposure, known or estimable selection probabilities where applicable, and feedback tied to the decision context. We observe feedback only for exposed outputs; unshown candidates are not observed negatives. Logged-bandit learning formalizes this partial-feedback problem. [Counterfactual Risk Minimization](https://arxiv.org/abs/1502.02362).

### Stage 3: Preference alignment

The feedback can inform different optimization methods:

- **DPO-style learning:** build justified, context-matched preferred / less-preferred output pairs and optimize their relative likelihood.
- **RL-style optimization:** define a reward, account for the behavior policy and data collection process, and optimize the recommendation policy.

DPO provides a preference-learning objective, not a method for automatically turning arbitrary click logs into valid preference pairs. [DPO](https://arxiv.org/abs/2305.18290).

A concrete recommendation example of alignment from user interactions is OneRec-V2, which uses behavioral feedback with duration-aware reward shaping and policy-optimization adjustments. It supports the behavioral-alignment part of this discussion, not the claim that teacher-generated SFT is always its starting point. [OneRec-V2](https://arxiv.org/abs/2508.20900).

## 6. Why direct behavioral feedback is attractive—and still biased

A learned engagement model can be inaccurate or inherit biases from its training data. Optimizing a generator against that model can exploit its errors. OneRec-V2 discusses this motivation for using real user feedback. [OneRec-V2](https://arxiv.org/abs/2508.20900).

However, **real behavior is not unbiased ground truth**. Exposure, position, popularity, and selection affect what is observed. Clicks and dwell time also need not equal satisfaction. Unbiased Learning-to-Rank demonstrates why directly treating biased clicks as relevance labels can be problematic. [Unbiased Learning-to-Rank with Biased Feedback](https://arxiv.org/abs/1608.04468).

A possible bootstrap loop is:

**Teacher → SFT → controlled exploration → behavioral feedback → preference data / reward → improved policy**

The appeal is progressively adding **task-specific behavioral supervision** to synthetic supervision. The qualification is that exposure design, reward definition, and bias correction remain necessary. Replacing a reward model with logged clicks does not make those problems disappear. [Counterfactual Risk Minimization](https://arxiv.org/abs/1502.02362).

## 7. Important distinction: DPO versus GRPO

**DPO** is naturally suited to preference pairs for the same conditioning context. It optimizes a policy relative to a reference model without explicitly fitting a separate reward model or sampling during the standard training objective. [DPO](https://arxiv.org/abs/2305.18290).

**GRPO**, introduced in DeepSeekMath, is a policy-optimization method that samples a group of outputs from a sampling policy and uses their relative rewards to form advantages. It avoids a separately trained critic. The original paper concerns mathematical reasoning, not a deployed recommendation system. [DeepSeekMath](https://arxiv.org/abs/2402.03300).

So I would avoid describing standard GRPO simply as an “off-policy update.” A more useful abstraction is:

| Method | Typical training signal | Important qualification |
| --- | --- | --- |
| DPO | Context-matched preference pairs | Logged engagement needs processing before it constitutes valid pairs |
| GRPO / related RL methods | Sampled outputs and their rewards | Sampling policy, likelihood ratios, and update constraints matter |

The original GRPO formulation can update using samples from an older policy with importance-ratio terms. It is therefore more precise to describe its sampling-and-update procedure than to impose an absolute on-policy / off-policy label. Replay and other variants introduce further design choices. [DeepSeekMath](https://arxiv.org/abs/2402.03300).

## 8. A unified view

### Pattern A: Representation-first

**Behavioral sequence → context engineering → LLM encoder → representation → retrieval / ranking**

The question is:

> What does this user/context mean?

The main bottlenecks include sequence construction, compression, representation learning, embedding quality, and serving cost.

### Pattern B: Generation-first

**Behavioral / target context → grounding → model generation → intent / query / category / semantic ID → retrieval or direct recommendation**

The question is:

> Given this context, what should we recommend or search for next?

The main bottlenecks include grounding, action-space control, invalid or unsupported outputs, exploration, reward design, and preference alignment.

The patterns can be combined. A system might encode history, generate an intent, retrieve catalog items, and rerank them. A generation-first system need not preserve a traditional scorer, and an encoder-first system need not serialize everything as text.

## 9. Broader trend

One way I interpret these research directions is:

**ID sequence modeling → semantic sequence modeling → context-conditioned representation learning → grounded generation**

This is a conceptual progression, not a chronological replacement of older methods. Efficient ID-based models remain useful, and not all generative recommenders inherit natural-language understanding from an LLM.

The design question becomes:

> What information should the model see, how should it be represented, and what decision should the model itself be responsible for?

That is why I expect context engineering to become more important in recommendation. Some experiments can change event selection, target-conditioned grounding, representation format, intermediate concepts, or downstream retrieval without redesigning the entire sequence model. Whether that tradeoff works should be measured against strong, cheaper baselines.

## 10. Ideas worth exploring

These are research questions, rather than established results of the cited papers.

### 1. Target-conditioned sequence compression

**History + current query → query-specific user representation**

Could this preserve useful information better than one universal user embedding? The serving tradeoff depends on whether the target is one query or thousands of candidate items.

### 2. Generate-to-retrieve

**Model → intents / concepts / semantic IDs → retrieval**

How can intermediate generation stay flexible while reliably resolving to valid catalog content?

### 3. Context engineering as an optimization layer

**Raw events → selection → compression → ordering → representation → model**

Can we learn or systematically evaluate the context pipeline itself, including what information compression discards?

### 4. Exploration as data generation

Can controlled generation and exposure deliberately collect useful preference data? Diversity at generation time should be evaluated separately from the policy that exposes recommendations.

### 5. Separate semantic intelligence from large-scale scoring

Where does an expensive foundation model provide enough marginal value to justify its cost? A practical experiment would compare hybrid designs with strong conventional retrieval and ranking baselines under the same latency and resource budget.

## References

The papers below support specific mechanisms discussed above. Cross-domain papers are identified in the text; they are not presented as evidence of production recommendation deployment.

1. Hidasi et al. (2015 preprint; ICLR 2016). [Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939). GRU4Rec.
2. Kang and McAuley (2018). [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781). SASRec.
3. Sun et al. (2019). [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/abs/1904.06690).
4. Chen et al. (2024). [HLLM: Enhancing Sequential Recommendations via Hierarchical Large Language Models for Item and User Modeling](https://arxiv.org/abs/2409.12740).
5. Nogueira, Jiang, and Lin (2020 arXiv version). [Document Ranking with a Pretrained Sequence-to-Sequence Model](https://arxiv.org/abs/2003.06713). Relevance-label generation for document ranking.
6. Geng et al. (2022). [Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5)](https://arxiv.org/abs/2203.13366).
7. Rajput et al. (2023). [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065). TIGER and semantic IDs.
8. Gao et al. (2022 preprint; ACL 2023). [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496). HyDE.
9. Bacciu et al. (2024). [Generating Query Recommendations via LLMs](https://arxiv.org/abs/2405.19749). GQR and retrieval-augmented GQR.
10. Hsieh et al. (2023). [Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes](https://arxiv.org/abs/2305.02301).
11. Zhou et al. / OneRec Team (2025). [OneRec-V2 Technical Report](https://arxiv.org/abs/2508.20900).
12. Rafailov et al. (2023). [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290).
13. Shao et al. (2024). [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300). Introduces GRPO.
14. Joachims, Swaminathan, and Schnabel (2016 preprint; WSDM 2017). [Unbiased Learning-to-Rank with Biased Feedback](https://arxiv.org/abs/1608.04468).
15. Swaminathan and Joachims (2015). [Counterfactual Risk Minimization: Learning from Logged Bandit Feedback](https://arxiv.org/abs/1502.02362).
16. Chevalier et al. (2023). [Adapting Language Models to Compress Contexts](https://arxiv.org/abs/2305.14788). AutoCompressors.
17. Ge et al. (2023). [In-context Autoencoder for Context Compression in a Large Language Model](https://arxiv.org/abs/2307.06945). ICAE.
18. Mu, Li, and Goodman (2023). [Learning to Compress Prompts with Gist Tokens](https://arxiv.org/abs/2304.08467).

## Keywords

`Sequential Recommendation` · `Generative Recommendation` · `LLM Recommendation` · `Context Engineering` · `Sequence Engineering` · `User Representation Learning` · `Target-aware Representation` · `Grounded Generation` · `Generate-to-Retrieve` · `Semantic ID` · `Query Recommendation` · `Autocomplete` · `LLM Ranker` · `Teacher-Student Distillation` · `SFT` · `DPO` · `GRPO` · `Online Exploration` · `Preference Alignment` · `Engagement Modeling`
