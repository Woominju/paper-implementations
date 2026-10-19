# FinDPO: Financial Sentiment Analysis for Algorithmic Trading through Preference Optimization of LLMs

*ICAIF 2025* · [Notion summary](https://road-lamp-b0c.notion.site/3fa64a44030c81b49fa9d6b663d258a1) · [Paper](https://arxiv.org/abs/2507.18417)

## What this implements
- DPO loss with a frozen reference policy obtained by supervised fine-tuning on noisy labels
- Logits turned into a continuous score p(positive) - p(negative)
- Daily long-short portfolio (top third vs. bottom third by score)

## Simplifications
A small bag-of-words network stands in for the LLM, and headlines and returns are synthetic, so the backtest numbers only illustrate the pipeline.

## Run
```bash
python main.py
```

## Output
```text
SFT (reference)  sentiment accuracy: 0.669
SFT + DPO        sentiment accuracy: 0.904

Long-short backtest on synthetic returns (annualized)
SFT (reference)  return  12.3%   Sharpe  1.16
SFT + DPO        return  36.8%   Sharpe  3.75
```
