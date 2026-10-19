"""
MSCRED (Zhang et al., AAAI 2019), simplified.

1. Signature matrices: for each time step t and window size w,
       M_ij(t) = sum_{d=0}^{w-1} x_i(t-d) * x_j(t-d) / w
   computed for several window sizes, so a sensor-by-sensor correlation
   picture at multiple scales becomes one multi-channel "image".
2. A convolutional encoder-decoder reconstructs the signature matrices of
   normal data. (The paper adds an attention-based ConvLSTM over a short
   history of matrices between encoder and decoder; that part is left out.)
3. Anomaly score = number of poorly reconstructed matrix entries
   (residual above a threshold set on validation data).
4. Root cause: the sensors whose rows have the largest residuals.

Synthetic data: 8 sensors driven by 2 latent signals. In the test period one
sensor decouples from the others, breaking the correlation pattern.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

rng = np.random.default_rng(0); torch.manual_seed(0)
N_SENS, SCALES, STEP = 8, (10, 30, 60), 5

def sensors(T):
    t = np.arange(T)
    z = np.stack([np.sin(2 * np.pi * t / 50), np.cos(2 * np.pi * t / 80)], 1)
    W = rng.normal(size=(2, N_SENS)) if not hasattr(sensors, "W") else sensors.W
    sensors.W = W
    return (z @ W + 0.1 * rng.standard_normal((T, N_SENS))).astype(np.float32)

train, test = sensors(5000), sensors(3000)
label = np.zeros(3000, int)
broken = []
for s, sensor in [(800, 2), (1700, 5), (2400, 0)]:
    test[s:s + 120, sensor] = rng.normal(0, 1.0, 120)       # decoupled sensor
    label[s:s + 120] = 1; broken.append(sensor)

def signature(x):
    """(T, n) -> (num_steps, len(SCALES), n, n), one stack every STEP points."""
    ts = range(max(SCALES), len(x), STEP)
    out = np.zeros((len(ts), len(SCALES), N_SENS, N_SENS), np.float32)
    for a, t in enumerate(ts):
        for c, w in enumerate(SCALES):
            seg = x[t - w:t]
            out[a, c] = seg.T @ seg / w
    return torch.tensor(out), np.array(list(ts))

Mtr, _ = signature(train); Mte, t_idx = signature(test)


class ConvED(nn.Module):
    def __init__(self, c=len(SCALES)):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(c, 32, 3, padding=1), nn.SELU(),
                                 nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SELU())
        self.dec = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.SELU(),
                                 nn.Conv2d(32, c, 3, padding=1))
    def forward(self, x):
        return self.dec(self.enc(x))

n_val = len(Mtr) // 5
Mfit, Mval = Mtr[:-n_val], Mtr[-n_val:]
model = ConvED(); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for ep in range(60):
    perm = torch.randperm(len(Mfit))
    for b in range(0, len(Mfit), 64):
        x = Mfit[perm[b:b + 64]]
        loss = F.mse_loss(model(x), x)
        opt.zero_grad(); loss.backward(); opt.step()

model.eval()
with torch.no_grad():
    r_val = (model(Mval) - Mval) ** 2
    r_te = (model(Mte) - Mte) ** 2
thr = torch.quantile(r_val.flatten(), 0.99)              # "poorly reconstructed" threshold
score = (r_te > thr).float().sum((1, 2, 3)).numpy()      # count of bad entries per step
lab = label[t_idx]
print(f"raw PR-AUC (per signature step): {average_precision_score(lab, score):.3f}   "
      f"anomaly ratio: {lab.mean():.3f}")

print("\nRoot-cause diagnosis (sensor with the largest residual row sum):")
for s, sensor in zip((800, 1700, 2400), broken):
    steps = np.where((t_idx >= s + 60) & (t_idx < s + 120))[0]
    row = r_te[steps].sum((0, 1, 3)).numpy()
    print(f"  anomaly at t={s}: true sensor {sensor}, ranked: {np.argsort(-row)[:3].tolist()}")
