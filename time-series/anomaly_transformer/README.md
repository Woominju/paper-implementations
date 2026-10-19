# Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy

*ICLR 2022* · [Notion summary](https://road-lamp-b0c.notion.site/3fa64a44030c81b3b639cc27a0b51420) · [Paper](https://arxiv.org/abs/2110.02642) · [Official code](https://github.com/thuml/Anomaly-Transformer)

## What this implements
- Anomaly-Attention with a learnable Gaussian prior-association and a softmax series-association
- Symmetric-KL association discrepancy and the minimax training strategy
- Anomaly score softmax(-AssDis) x reconstruction error, evaluated with raw PR-AUC and with point-adjusted F1

## Simplifications
One layer on synthetic data. On raw PR-AUC this minimal model scores close to random and below plain reconstruction error, while random scores get the highest point-adjusted F1. This matches what I found in my thesis: point adjustment hides the difference between a model and random guessing.

## Run
```bash
python main.py
```

## Output
```text
scores                       raw PR-AUC  best F1 PA best F1
Anomaly Transformer               0.099    0.157      0.732
reconstruction error only         0.221    0.233      0.725
random                            0.095    0.137      0.815

anomaly ratio in test: 0.073
```
