"""
BALD active learning (Houlsby et al., 2011) with MC dropout, plus a simple
BatchBALD-style greedy batch selection (Kirsch et al., NeurIPS 2019).

BALD picks the unlabeled points where the model's possible hypotheses
disagree most, i.e. the mutual information between the prediction y and
the model parameters w:

    I[y; w | x] = H[ E_w p(y|x,w) ] - E_w H[ p(y|x,w) ]

With MC dropout, each forward pass with dropout on is one sample of w.

The script starts from 20 labeled digits (scikit-learn), acquires 10 points
per round for 10 rounds, and compares random acquisition, top-k BALD and a
greedy joint (BatchBALD-style) batch.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float); y = torch.tensor(y)
Xpool, Xte, ypool, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
K_MC = 20


def make_model():
    return nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.5),
                         nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 10))

def fit(idx, seed):
    torch.manual_seed(seed)
    model = make_model(); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(300):
        model.train()
        loss = F.cross_entropy(model(Xpool[idx]), ypool[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return model

def accuracy(model):
    model.eval()
    with torch.no_grad():
        return (model(Xte).argmax(1) == yte).float().mean().item()

def mc_probs(model, x):
    model.train()                                    # keep dropout on: each pass samples w
    with torch.no_grad():
        return torch.stack([F.softmax(model(x), 1) for _ in range(K_MC)])   # (K, N, C)

def entropy(p, dim=-1):
    return -(p * torch.log(p + 1e-12)).sum(dim)

def bald_scores(P):
    return entropy(P.mean(0)) - entropy(P).mean(0)

def batchbald_greedy(P, b, M=1000, seed=0):
    """Greedy batch that maximizes the joint mutual information
    I[y_1..y_b; w] = H[y_1..y_b] - sum_i E_w H[y_i | w].

    The joint entropy is exact while the number of label configurations is
    small. After that, M configurations are sampled from p(y_1..y_t) and the
    entropy is estimated by importance sampling, as in the BatchBALD paper.
    Columns of `joint` hold p(config | w_k) for each MC sample k."""
    g = torch.Generator().manual_seed(seed)
    K, N, C = P.shape
    cond_ent = entropy(P).mean(0)                    # E_w H[y_i | w]
    joint, sampled = torch.ones(K, 1), False         # exact configurations until sampled=True
    chosen = []
    cand = torch.topk(bald_scores(P), min(N, 100)).indices.tolist()
    for _ in range(b):
        best, best_score = None, -math.inf
        for i in cand:
            if i in chosen:
                continue
            new = joint.unsqueeze(-1) * P[:, i, :].unsqueeze(1)          # (K, S, C)
            mix = new.mean(0)                                            # (S, C)
            if sampled:
                q = joint.mean(0).unsqueeze(-1)                          # proposal prob of each sample
                h = -((mix / q) * torch.log(mix + 1e-12)).sum() / joint.shape[1]
            else:
                h = -(mix * torch.log(mix + 1e-12)).sum()
            score = h - cond_ent[chosen + [i]].sum()
            if score > best_score:
                best, best_score = i, score
        chosen.append(best)
        new = joint.unsqueeze(-1) * P[:, best, :].unsqueeze(1)
        if not sampled and new.shape[1] * C <= M:
            joint = new.reshape(K, -1)                                   # stay exact
        elif not sampled:                                                # switch to M samples
            flat = new.reshape(K, -1)
            idx = torch.multinomial(flat.mean(0), M, replacement=True, generator=g)
            joint, sampled = flat[:, idx], True
        else:                                                            # extend each sample by one label
            mix = new.mean(0)                                            # (M, C)
            c = torch.multinomial(mix / mix.sum(1, keepdim=True), 1, generator=g).squeeze(1)
            joint = new[:, torch.arange(new.shape[1]), c]
    return chosen


def active_learning(strategy, rounds=10, b=10, seed=0):
    rng = np.random.default_rng(seed)
    labeled = list(rng.choice(len(Xpool), 20, replace=False))
    accs = []
    for r in range(rounds + 1):
        model = fit(torch.tensor(labeled), seed + r)
        accs.append(accuracy(model))
        if r == rounds:
            break
        pool = np.setdiff1d(np.arange(len(Xpool)), labeled)
        if strategy == "random":
            pick = list(rng.choice(pool, b, replace=False))
        else:
            P = mc_probs(model, Xpool[pool])
            if strategy == "bald":
                pick = pool[torch.topk(bald_scores(P), b).indices.numpy()].tolist()
            else:
                pick = pool[batchbald_greedy(P, b, seed=seed + r)].tolist()
        labeled += pick
    return accs


results = {s: np.mean([active_learning(s, seed=sd) for sd in range(3)], axis=0)
           for s in ("random", "bald", "batchbald")}
print("Test accuracy vs number of labeled examples (mean of 3 seeds)")
print("labels   random    BALD   BatchBALD")
for r in range(0, 11, 2):
    print(f"{20 + 10 * r:6d}   {results['random'][r]:.3f}   {results['bald'][r]:.3f}   {results['batchbald'][r]:.3f}")
