"""
Skip-gram with negative sampling (Mikolov et al., NIPS 2013), minimal PyTorch version.

Implements the three ideas from the paper on a small synthetic corpus:
  1. Negative sampling from the unigram distribution raised to the 3/4 power.
  2. Subsampling of frequent words: drop word w with probability 1 - sqrt(t / f(w)).
  3. Vector arithmetic on the learned embeddings (country + capital relation).
"""
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

random.seed(0); np.random.seed(0); torch.manual_seed(0)

# ----------------------------------------------------------------------
# Synthetic corpus: countries, capitals, languages and filler words
# ----------------------------------------------------------------------
PAIRS = [("korea", "seoul", "korean"), ("japan", "tokyo", "japanese"),
         ("france", "paris", "french"), ("germany", "berlin", "german"),
         ("italy", "rome", "italian"), ("spain", "madrid", "spanish")]
TEMPLATES = [
    "the capital of {c} is {cap}", "{cap} is the capital city of {c}",
    "people in {c} speak {lang}", "{lang} is spoken in {c}",
    "i flew to {cap} in {c}", "{cap} has many museums and the {lang} language",
]

def build_corpus(n_sentences=20000):
    sents = []
    for _ in range(n_sentences):
        c, cap, lang = random.choice(PAIRS)
        sents.append(random.choice(TEMPLATES).format(c=c, cap=cap, lang=lang).split())
    return sents

corpus = build_corpus()
counts = Counter(w for s in corpus for w in s)
vocab = sorted(counts)
w2i = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
total = sum(counts.values())
freq = np.array([counts[w] / total for w in vocab])

# ----------------------------------------------------------------------
# 1) Subsampling of frequent words
# ----------------------------------------------------------------------
T_SUB = 1e-2
p_keep = np.minimum(1.0, np.sqrt(T_SUB / freq))

# ----------------------------------------------------------------------
# 2) Noise distribution: unigram ^ 3/4
# ----------------------------------------------------------------------
noise = freq ** 0.75
noise = torch.tensor(noise / noise.sum(), dtype=torch.float)

def skipgram_pairs(window=2):
    pairs = []
    for s in corpus:
        ids = [w2i[w] for w in s if random.random() < p_keep[w2i[w]]]
        for i, center in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if j != i:
                    pairs.append((center, ids[j]))
    return torch.tensor(pairs)


class SGNS(nn.Module):
    def __init__(self, V, dim=32):
        super().__init__()
        self.inp = nn.Embedding(V, dim)    # v_w  (center word)
        self.out = nn.Embedding(V, dim)    # v'_w (context word)
        nn.init.uniform_(self.inp.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.out.weight)

    def loss(self, center, context, k=5):
        v = self.inp(center)                                          # (B, D)
        pos = (v * self.out(context)).sum(-1)                         # (B,)
        neg_ids = torch.multinomial(noise, center.numel() * k, replacement=True).view(-1, k)
        neg = torch.bmm(self.out(neg_ids), v.unsqueeze(-1)).squeeze(-1)  # (B, k)
        # log sigma(v'_o . v_c) + sum_k log sigma(-v'_k . v_c)
        return -(nn.functional.logsigmoid(pos) + nn.functional.logsigmoid(-neg).sum(-1)).mean()


model = SGNS(V, dim=24)
opt = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(15):
    pairs = skipgram_pairs()                 # re-subsample every epoch
    perm = torch.randperm(len(pairs))
    tot = 0.0
    for b in range(0, len(pairs), 512):
        batch = pairs[perm[b:b + 512]]
        loss = model.loss(batch[:, 0], batch[:, 1])
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(batch)
    print(f"epoch {epoch}  pairs={len(pairs):6d}  loss={tot / len(pairs):.3f}")

# ----------------------------------------------------------------------
# 3) Nearest neighbours and analogies
# ----------------------------------------------------------------------
E = nn.functional.normalize(model.inp.weight.detach(), dim=1)

def nearest(vec, exclude=(), k=3):
    sims = E @ nn.functional.normalize(vec, dim=0)
    order = sims.argsort(descending=True).tolist()
    return [vocab[i] for i in order if vocab[i] not in exclude][:k]

print("\nNearest to 'seoul':", nearest(E[w2i["seoul"]], exclude={"seoul"}))
correct = 0
tests = [(a, b, c, d) for (a, b, _) in PAIRS for (c, d, _) in PAIRS if a != c]
for a, b, c, d in tests:                      # b - a + c should be close to d
    guess = nearest(E[w2i[b]] - E[w2i[a]] + E[w2i[c]], exclude={a, b, c}, k=1)[0]
    correct += guess == d
print(f"Analogy accuracy (capital - country + country'): {correct}/{len(tests)}")
print("Example: paris - france + korea ->",
      nearest(E[w2i["paris"]] - E[w2i["france"]] + E[w2i["korea"]], exclude={"paris", "france", "korea"}, k=1)[0])
