"""
Anomaly Transformer (Xu et al., ICLR 2022), minimal single-layer version,
plus a check of the evaluation protocol.

Anomaly-Attention keeps two association maps per window:
  prior-association  P: Gaussian kernel over |i - j| with a learnable width sigma_i
  series-association S: ordinary softmax self-attention
Association discrepancy AssDis = KL(P||S) + KL(S||P), averaged over layers.
Anomalous points attend mostly to their neighbours, so their discrepancy is small.

Training uses the paper's minimax strategy:
  minimize phase: L_recon + lambda * AssDis(P, stopgrad S)   (P moves toward S)
  maximize phase: L_recon - lambda * AssDis(stopgrad P, S)   (S moves away from P)
Anomaly score = softmax(-AssDis) * per-point reconstruction error.

The script also scores the model with raw PR-AUC and with point-adjusted F1,
and shows that random scores reach a high point-adjusted F1.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score

torch.manual_seed(0); rng = np.random.default_rng(0)
WIN, D_IN, D, LAMBDA = 50, 4, 32, 3.0

# ----------------------------------------------------------------------
# Synthetic multivariate data with injected anomaly segments in the test set
# ----------------------------------------------------------------------
def series(T):
    t = np.arange(T)
    base = np.stack([np.sin(2 * np.pi * t / p) for p in (25, 40, 60, 90)], 1)
    return (base + 0.1 * rng.standard_normal((T, D_IN))).astype(np.float32)

train = series(4000)
test = series(2000); label = np.zeros(2000, int)
for s in rng.choice(np.arange(100, 1900), 8, replace=False):
    L = rng.integers(10, 40); ch = rng.integers(0, D_IN)
    kind = rng.integers(0, 2)
    test[s:s + L, ch] = test[s, ch] if kind == 0 else test[s:s + L, ch] + rng.normal(0, 0.8, L)
    label[s:s + L] = 1

def windows(x):
    return torch.tensor(np.stack([x[i:i + WIN] for i in range(0, len(x) - WIN + 1, WIN)]))


class AnomalyAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q, self.k, self.v, self.sigma = (nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, 1))
        idx = torch.arange(WIN).float()
        self.register_buffer("dist", (idx[:, None] - idx[None, :]).abs())

    def forward(self, x):
        S = F.softmax(self.q(x) @ self.k(x).transpose(1, 2) / math.sqrt(x.size(-1)), -1)
        sigma = torch.sigmoid(self.sigma(x) * 5) + 1e-5                     # (B, L, 1)
        sigma = torch.pow(3, sigma) - 1
        P = torch.exp(-self.dist ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
        P = P / P.sum(-1, keepdim=True)                                      # row-normalize
        return S @ self.v(x), P, S


class AnomalyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Linear(D_IN, D)
        self.pos = nn.Parameter(torch.randn(1, WIN, D) * 0.02)
        self.attn = AnomalyAttention(D)
        self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.out = nn.Linear(D, D_IN)

    def forward(self, x):
        h = self.emb(x) + self.pos
        a, P, S = self.attn(h)
        h = self.n1(h + a)
        h = self.n2(h + self.ff(h))
        return self.out(h), P, S


def kl(p, q):                                  # per-point KL over the attention row
    return (p * (torch.log(p + 1e-4) - torch.log(q + 1e-4))).sum(-1)

def ass_dis(P, S):
    return kl(P, S) + kl(S, P)                  # (B, L)

model = AnomalyTransformer()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
Xtr = windows(train)
for ep in range(30):
    for b in range(0, len(Xtr), 16):
        x = Xtr[b:b + 16]
        rec, P, S = model(x)
        r = F.mse_loss(rec, x)
        loss_min = r + LAMBDA * ass_dis(P, S.detach()).mean()    # prior follows series
        loss_max = r - LAMBDA * ass_dis(P.detach(), S).mean()    # series moves away from prior
        opt.zero_grad()
        loss_min.backward(retain_graph=True)
        loss_max.backward()
        opt.step()

model.eval()
with torch.no_grad():
    Xte = windows(test)
    rec, P, S = model(Xte)
    recon_err = ((rec - Xte) ** 2).mean(-1)                      # (B, L)
    score = (F.softmax(-ass_dis(P, S), dim=-1) * recon_err).reshape(-1).numpy()
    recon_only = recon_err.reshape(-1).numpy()
lab = label[:len(score)]


# ----------------------------------------------------------------------
# Evaluation: raw PR-AUC vs point-adjusted F1
# ----------------------------------------------------------------------
def point_adjust(pred, lab):
    """If any point inside an anomaly segment is flagged, flag the whole segment."""
    pred = pred.copy()
    d = np.diff(np.concatenate([[0], lab, [0]]))
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if pred[s:e].any():
            pred[s:e] = 1
    return pred

def best_f1(score, lab, adjust):
    best = 0
    for q in np.linspace(0.5, 0.995, 100):
        pred = (score > np.quantile(score, q)).astype(int)
        best = max(best, f1_score(lab, point_adjust(pred, lab) if adjust else pred))
    return best

random_score = rng.random(len(score))
print(f"{'scores':28s} {'raw PR-AUC':>10s} {'best F1':>8s} {'PA best F1':>10s}")
for name, s in [("Anomaly Transformer", score), ("reconstruction error only", recon_only),
                ("random", random_score)]:
    print(f"{name:28s} {average_precision_score(lab, s):10.3f} {best_f1(s, lab, False):8.3f} {best_f1(s, lab, True):10.3f}")
print(f"\nanomaly ratio in test: {lab.mean():.3f}")
