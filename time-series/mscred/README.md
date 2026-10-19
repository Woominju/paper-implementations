# MSCRED: Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series

AAAI 2019 | [Notion summary](https://road-lamp-b0c.notion.site/3f664a44030c81098b2bcfcecb681858) | [Paper](https://arxiv.org/abs/1811.08055)

## What is in the code
- Multi-scale signature matrices (sensor-by-sensor inner products over 10, 30 and 60 steps)
- Convolutional encoder-decoder trained to reconstruct normal signature matrices
- Anomaly score = number of badly reconstructed entries; root-cause ranking by residual row sums

## Differences from the paper
The attention-based ConvLSTM between encoder and decoder is omitted. Root cause is ranked correctly first in 2 of 3 injected anomalies.

## Run
```bash
python main.py
```

## Output
```text
raw PR-AUC (per signature step): 0.714   anomaly ratio: 0.122

Root-cause diagnosis (sensor with the largest residual row sum):
  anomaly at t=800: true sensor 2, ranked: [6, 2, 7]
  anomaly at t=1700: true sensor 5, ranked: [5, 4, 6]
  anomaly at t=2400: true sensor 0, ranked: [0, 4, 1]
```
