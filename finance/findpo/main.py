"""
FinDPO in miniature (ICAIF 2025): align a sentiment model with Direct
Preference Optimization, turn its logits into a continuous sentiment score,
and trade a long-short portfolio on that score.

DPO loss (Rafailov et al., 2023) for a preferred answer y_w and a rejected
answer y_l to the same prompt x:
    L = -log sigmoid( beta * [ (log pi(y_w|x) - log pi_ref(y_w|x))
                             - (log pi(y_l|x) - log pi_ref(y_l|x)) ] )

A real LLM is replaced by a small bag-of-words network whose "answer" is one
of three tokens: positive / negative / neutral. Steps:
  1. SFT: train the reference model on a small set of noisily labeled headlines.
  2. DPO: start from the reference, train on preference pairs
     (correct label preferred over a wrong one), keeping pi_ref frozen.
  3. Score = p(positive) - p(negative); each day go long the top third of
     stocks by score and short the bottom third.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

rng = np.random.default_rng(0); torch.manual_seed(0)
LABELS = ["positive", "negative", "neutral"]
POS = ["beats", "surges", "upgrade", "record", "growth", "raises"]
NEG = ["misses", "plunges", "downgrade", "lawsuit", "loss", "cuts"]
NEU = ["announces", "schedules", "meeting", "reports", "update", "comments"]
FILL = ["company", "shares", "quarter", "guidance", "analysts", "market", "outlook", "ceo"]
vocab = POS + NEG + NEU + FILL; w2i = {w: i for i, w in enumerate(vocab)}

def headline(sent):
    words = list(rng.choice([POS, NEG, NEU][sent], 1)) + list(rng.choice(FILL, 3))
    if rng.random() < 0.3:                           # distractor word from another class
        words += list(rng.choice([POS, NEG, NEU][rng.integers(3)], 1))
    x = np.zeros(len(vocab), np.float32)
    for w in words: x[w2i[w]] += 1
    return x

def data(n, label_noise=0.0):
    y = rng.integers(0, 3, n)
    X = np.stack([headline(s) for s in y])
    y_obs = np.where(rng.random(n) < label_noise, rng.integers(0, 3, n), y)
    return torch.tensor(X), torch.tensor(y), torch.tensor(y_obs)

policy = nn.Sequential(nn.Linear(len(vocab), 32), nn.ReLU(), nn.Linear(32, 3))
logp = lambda m, x: F.log_softmax(m(x), -1)

# 1) SFT on 150 headlines with 35% label noise -> reference model
X_sft, _, y_sft = data(150, label_noise=0.35)
opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
for _ in range(200):
    loss = F.cross_entropy(policy(X_sft), y_sft); opt.zero_grad(); loss.backward(); opt.step()
ref = copy.deepcopy(policy).eval()
for p in ref.parameters(): p.requires_grad_(False)

# 2) DPO on preference pairs: correct label (chosen) vs a random wrong label (rejected)
X_pref, y_true, _ = data(600)
y_rej = (y_true + torch.tensor(rng.integers(1, 3, 600))) % 3
dpo = copy.deepcopy(policy); opt = torch.optim.Adam(dpo.parameters(), lr=5e-3); beta = 0.1
for _ in range(300):
    lp, lr_ = logp(dpo, X_pref), logp(ref, X_pref)
    idx = torch.arange(len(X_pref))
    margin = (lp[idx, y_true] - lr_[idx, y_true]) - (lp[idx, y_rej] - lr_[idx, y_rej])
    loss = -F.logsigmoid(beta * margin).mean()
    opt.zero_grad(); loss.backward(); opt.step()

X_te, y_te, _ = data(2000)
for name, m in [("SFT (reference)", ref), ("SFT + DPO", dpo)]:
    with torch.no_grad():
        print(f"{name:16s} sentiment accuracy: {(m(X_te).argmax(1) == y_te).float().mean():.3f}")

# 3) Continuous score -> long-short portfolio on synthetic returns
#    Each day 60 stocks get one headline; next-day return = 0.1% * true sentiment + 2% noise.
def backtest(m, days=250, n=60):
    daily = []
    for _ in range(days):
        X, y, _ = data(n)
        true = torch.tensor([1.0, -1.0, 0.0])[y].numpy()
        ret = 0.001 * true + 0.02 * rng.standard_normal(n)
        with torch.no_grad():
            p = F.softmax(m(X), -1).numpy()
        score = p[:, 0] - p[:, 1]
        order = np.argsort(score); k = n // 3
        daily.append(ret[order[-k:]].mean() - ret[order[:k]].mean())
    daily = np.array(daily)
    return daily.mean() * 252, daily.mean() / daily.std() * np.sqrt(252)

print("\nLong-short backtest on synthetic returns (annualized)")
for name, m in [("SFT (reference)", ref), ("SFT + DPO", dpo)]:
    r, sr = backtest(m)
    print(f"{name:16s} return {r:6.1%}   Sharpe {sr:5.2f}")
