"""Python 3, standard library only. Synthetic fixed-length SID decoder.

Sibling-rank penalty: Li et al. (2016), https://arxiv.org/abs/1611.08562
This demo uses a CURRENT-STEP penalty, retaining raw scores across steps.
It is a teaching variant, not a full paper reproduction or neural model.
"""
import math
import random


# A normalized conditional distribution at every node, standing in for a model.
TABLE = {
    (): {"A": .7, "B": .2, "C": .1},
    ("A",): {"1": .4, "2": .35, "3": .25},
    ("B",): {"1": .7, "2": .2, "3": .1},
    ("C",): {"1": .8, "2": .15, "3": .05},
}
CATALOG = {("A", "1"), ("A", "2"), ("B", "1"), ("C", "1")}


def next_probs(prefix):
    return TABLE.get(prefix, {})


def beam_search(width=3, depth=2, gamma=0.0, catalog=None):
    """Each beam is (SID prefix, RAW cumulative log probability)."""
    beams = [((), 0.0)]
    for _ in range(depth):
        candidates = []
        for prefix, raw_score in beams:
            children = sorted(next_probs(prefix).items(),
                              key=lambda pair: pair[1], reverse=True)
            for rank, (token, prob) in enumerate(children):
                sid = prefix + (token,)
                if catalog is not None and not any(
                    item[:len(sid)] == sid for item in catalog
                ):
                    continue  # miniature trie: reject invalid prefixes
                raw = raw_score + math.log(prob)
                adjusted = raw - gamma * rank  # zero-based sibling rank
                candidates.append((sid, raw, adjusted))
        candidates.sort(key=lambda row: row[2], reverse=True)
        beams = [(sid, raw) for sid, raw, _ in candidates[:width]]
    return beams


def top_p_one(rng, p=1.0, temperature=1.0, depth=2, top_k=None):
    if not 0 < p <= 1 or temperature <= 0:
        raise ValueError("Require 0 < p <= 1 and temperature > 0")
    prefix = ()
    for _ in range(depth):
        probs = next_probs(prefix)
        logits = {t: math.log(v) / temperature for t, v in probs.items()}
        peak = max(logits.values())
        weights = {t: math.exp(v - peak) for t, v in logits.items()}
        if top_k is not None:
            if top_k < 1:
                raise ValueError("Require top_k >= 1")
            weights = dict(sorted(weights.items(), key=lambda x: x[1],
                                  reverse=True)[:top_k])
        total = sum(weights.values())
        nucleus, cumulative = [], 0.0
        for token, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            nucleus.append((token, weight))
            cumulative += weight / total
            if cumulative >= p:
                break
        # choices normalizes retained weights internally; draws ONE token.
        tokens, retained_weights = zip(*nucleus)
        prefix += (rng.choices(tokens, weights=retained_weights, k=1)[0],)
    return prefix


def gumbel(rng):
    return -math.log(-math.log(max(rng.random(), 1e-15)))


def all_sequences(depth=2):
    # Exhaustive reference for this tiny tree only.
    return beam_search(width=100, depth=depth)


def gumbel_top_k(rng, k=3):
    leaves = all_sequences()
    return sorted(leaves, key=lambda row: row[1] + gumbel(rng), reverse=True)[:k]


def stochastic_beam(rng, width=3, depth=2):
    # prefix, raw log mass, subtree's conditioned Gumbel maximum
    beams = [((), 0.0, gumbel(rng))]
    for _ in range(depth):
        candidates = []
        for prefix, raw, parent_max in beams:
            children = [(prefix + (t,), raw + math.log(p))
                        for t, p in next_probs(prefix).items()]
            noisy = [score + gumbel(rng) for sid, score in children]
            z = max(noisy)
            for (sid, score), g in zip(children, noisy):
                # Condition child maxima so their maximum equals parent_max.
                # Scale by exp(-parent_max) to avoid unnecessary overflow.
                delta = math.exp(parent_max - g) * (-math.expm1(g - z))
                conditioned = parent_max - math.log1p(delta)
                candidates.append((sid, score, conditioned))
        beams = sorted(candidates, key=lambda row: row[2], reverse=True)[:width]
    return [(sid, raw) for sid, raw, noisy in beams]


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def prefix_beam(width=3, depth=2, lam=1.5):
    beams = [((), 0.0)]
    for _ in range(depth):
        pool = [(pre + (t,), raw + math.log(p), pre)
                for pre, raw in beams for t, p in next_probs(pre).items()]
        selected = []
        while pool and len(selected) < width:
            best = max(pool, key=lambda c: c[1] - lam * sum(
                lcp(c[2], old[2]) for old in selected))
            selected.append(best)
            pool.remove(best)
        beams = [(sid, raw) for sid, raw, parent in selected]
    return beams


def grouped_beam(groups=3, depth=2, lam=1.5):
    # Minimal grouped DBS: ONE beam per group, same-position token overlap.
    beams = [((), 0.0, 0.0) for _ in range(groups)]
    for _ in range(depth):
        updated, used_tokens = [], []
        for prefix, raw, search in beams:
            options = [(prefix + (t,), raw + math.log(p),
                        search + math.log(p) - lam * used_tokens.count(t))
                       for t, p in next_probs(prefix).items()]
            best = max(options, key=lambda c: c[2])
            updated.append(best)
            used_tokens.append(best[0][-1])
        beams = updated
    return [(sid, raw) for sid, raw, search in beams]


def mmr(pool, k=3, lam=1.5):
    remaining, selected = list(pool), []
    while remaining and len(selected) < k:
        best = max(remaining, key=lambda c: c[1] - lam * max(
            (lcp(c[0], old[0]) for old in selected), default=0))
        selected.append(best)
        remaining.remove(best)
    return selected


def show(label, beams):
    print(label, [("".join(sid), round(math.exp(raw), 3)) for sid, raw in beams])


def main():
    ordinary = beam_search()
    diverse = beam_search(gamma=1.5)
    constrained = beam_search(catalog=CATALOG)
    show("greedy", beam_search(width=1))
    show("beam", ordinary)
    show("sibling", diverse)
    show("grouped", grouped_beam())
    show("prefix", prefix_beam())
    show("catalog", constrained)
    for name, kwargs in [("ancestral", {}), ("temperature", {"temperature": 1.5}),
                         ("top-k", {"top_k": 2}), ("top-p", {"p": .8})]:
        rng = random.Random(7)
        print(name, ["".join(top_p_one(rng, **kwargs)) for _ in range(6)])
    show("Gumbel leaves", gumbel_top_k(random.Random(7)))
    show("stochastic beam", stochastic_beam(random.Random(7)))
    show("pool + MMR", mmr(all_sequences()))

    assert [sid for sid, _ in ordinary] == [("A", "1"), ("A", "2"), ("A", "3")]
    assert [sid for sid, _ in diverse] == [("A", "1"), ("B", "1"), ("C", "1")]
    assert all(sid in CATALOG for sid, _ in constrained)
    assert top_p_one(random.Random(0), p=.01) == ("A", "1")
    # Retaining all leaves must yield nine different SIDs, with original scores.
    samples = stochastic_beam(random.Random(7), width=9)
    assert len({sid for sid, _ in samples}) == 9
    assert dict(samples) == dict(all_sequences())
    assert prefix_beam(lam=0) == ordinary
    print("checks passed")


if __name__ == "__main__":
    main()
