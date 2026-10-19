# Selective Learning for Deep Time Series Forecasting

NeurIPS 2025 | [Notion summary](https://road-lamp-b0c.notion.site/3f664a44030c817fb3d3f276a45d62f8) | [Paper](https://arxiv.org/abs/2510.25207) | [Official code](https://github.com/GestaltCogTeam/selective-learning)

## What is in the code
- Loss computed only on generalizable time points through a dual mask
- Anomaly mask from large residuals and uncertainty mask from residual variance during early training
- Plain MSE vs. masked MSE on a training series with injected spikes, tested on clean data

## Differences from the paper
The masks use simple residual quantiles from a briefly trained reference model; the paper's estimators are more elaborate.

## Run
```bash
python main.py
```

## Output
```text
masked out 9.5% of training targets; 99.4% of corrupted targets are masked
plain MSE                clean test MSE = 0.0260
selective (masked) MSE   clean test MSE = 0.0141
```
