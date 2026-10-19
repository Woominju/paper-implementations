"""
Selective Learning for deep time-series forecasting (NeurIPS 2025), simplified.

Idea: plain MSE trains on every future time point, including ones that are
anomalous or unpredictable, which pushes the model to overfit them. Selective
learning computes the loss only on "generalizable" time points, chosen with
two masks:
  anomaly mask      drop targets whose residual under a reference estimate is
                    unusually large (likely corrupted)
  uncertainty mask  drop targets whose residual varies most across training
                    (inherently unpredictable)

This script trains the same MLP forecaster with plain MSE and with the masked
loss on a training series containing injected spikes, then evaluates both on a
clean test series. The masks here use simple residual quantiles from a
reference model trained for a few epochs; the paper's own estimators are more
elaborate.
"""
import numpy as np
import torch
import torch.nn as nn

rng = np.random.default_rng(0); torch.manual_seed(0)
LOOKBACK, HORIZON = 48, 12

def clean_series(T, phase=0.0):
    t = np.arange(T)
    return (np.sin(2 * np.pi * t / 24 + phase) + 0.5 * np.sin(2 * np.pi * t / 7)
            + 0.1 * rng.standard_normal(T)).astype(np.float32)

train = clean_series(6000)
spikes = rng.choice(len(train), 300, replace=False)
train[spikes] += rng.choice([-1, 1], 300) * rng.uniform(3, 6, 300)      # corrupted training targets
test = clean_series(2000, phase=1.0)

def make_xy(s):
    idx = range(0, len(s) - LOOKBACK - HORIZON)
    X = np.stack([s[i:i + LOOKBACK] for i in idx]); Y = np.stack([s[i + LOOKBACK:i + LOOKBACK + HORIZON] for i in idx])
    return torch.tensor(X), torch.tensor(Y)

Xtr, Ytr = make_xy(train); Xte, Yte = make_xy(test)

def model_fn():
    return nn.Sequential(nn.Linear(LOOKBACK, 128), nn.ReLU(), nn.Linear(128, HORIZON))

def train_model(mask=None, epochs=40, seed=0, track=False):
    torch.manual_seed(seed)
    m = model_fn(); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    history = []
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr))
        for b in range(0, len(Xtr), 128):
            i = perm[b:b + 128]
            err = (m(Xtr[i]) - Ytr[i]) ** 2
            w = mask[i] if mask is not None else torch.ones_like(err)
            loss = (err * w).sum() / w.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
        if track:
            with torch.no_grad():
                history.append((m(Xtr) - Ytr).abs())
    return m, history

# Reference model (a few epochs) gives residuals for the two masks
_, hist = train_model(epochs=6, track=True)
res = torch.stack(hist)                                  # (epochs, N, H)
mean_res, var_res = res.mean(0), res.var(0)
anomaly_mask = (mean_res < torch.quantile(mean_res, 0.95)).float()
uncert_mask = (var_res < torch.quantile(var_res, 0.95)).float()
mask = anomaly_mask * uncert_mask
spike_targets = torch.zeros_like(Ytr, dtype=torch.bool)
for s in spikes:                                         # which targets are corrupted
    for h in range(HORIZON):
        i = s - LOOKBACK - h
        if 0 <= i < len(Ytr): spike_targets[i, h] = True
print(f"masked out {1 - mask.mean():.1%} of training targets; "
      f"{(mask[spike_targets] == 0).float().mean():.1%} of corrupted targets are masked")

for name, m in [("plain MSE", None), ("selective (masked) MSE", mask)]:
    model, _ = train_model(mask=m)
    with torch.no_grad():
        mse = ((model(Xte) - Yte) ** 2).mean().item()
    print(f"{name:24s} clean test MSE = {mse:.4f}")
