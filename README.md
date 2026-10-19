# Paper Implementations

Small, runnable implementations of the core idea of papers I have read and summarized in my [Notion paper review](https://road-lamp-b0c.notion.site/6a0851c1fc00409191d803d84c9481c4).
Each folder links back to its Notion summary, the paper and the official code, says what was simplified, and shows the output of `python main.py`.

Everything runs on a CPU in about a minute or less with synthetic or built-in data (no downloads).
These are study implementations, not reproductions of the papers' reported numbers.

| Field | Paper | Venue | What the code shows | Notes |
|---|---|---|---|---|
| Time-series anomaly detection | [Anomaly Transformer](time-series/anomaly_transformer) | ICLR 2022 | Association discrepancy; raw PR-AUC vs. point-adjusted F1 | [Notion](https://road-lamp-b0c.notion.site/3fa64a44030c81b3b639cc27a0b51420) |
| Time-series anomaly detection | [MSCRED](time-series/mscred) | AAAI 2019 | Signature matrices, conv encoder-decoder, root-cause ranking | [Notion](https://road-lamp-b0c.notion.site/3f664a44030c81098b2bcfcecb681858) |
| Time-series forecasting | [Selective Learning](time-series/selective_learning) | NeurIPS 2025 | Training only on generalizable time points | [Notion](https://road-lamp-b0c.notion.site/3f664a44030c817fb3d3f276a45d62f8) |
| NLP | [Word2Vec (skip-gram, negative sampling)](nlp/word2vec_sgns) | NIPS 2013 | Negative sampling, subsampling, analogies | [Notion](https://road-lamp-b0c.notion.site/3fa64a44030c810894b0f06a25d430ab) |
| NLP | [GPT-1](nlp/gpt1_mini) | OpenAI 2018 | Generative pre-training, then fine-tuning with an auxiliary LM loss | [Notion](https://road-lamp-b0c.notion.site/3fa64a44030c8153a3adec88b54de3e3) |
| NLP | [End-To-End Memory Networks](nlp/memn2n) | NIPS 2015 | Multi-hop soft attention over memories | [Notion](https://road-lamp-b0c.notion.site/3fa64a44030c812c9c1cf7eb6318f37d) |
| Deep-learning methods | [Batch Normalization](deep-learning/batchnorm) | ICML 2015 | BN from scratch, checked against PyTorch | [Notion](https://road-lamp-b0c.notion.site/3fa64a44030c81ca93fdd9c4a54b53df) |
| Deep-learning methods | [BALD & BatchBALD](deep-learning/bald_active_learning) | 2011 / NeurIPS 2019 | Choosing which data to label with MC-dropout mutual information | [Notion](https://road-lamp-b0c.notion.site/3fa64a44030c8127886ef440cbf2dd98) |
| Finance | [FinDPO](finance/findpo) | ICAIF 2025 | DPO for sentiment, logits to scores, long-short portfolio | [Notion](https://road-lamp-b0c.notion.site/3fa64a44030c81b49fa9d6b663d258a1) |
| Finance | [Robust portfolios (worst-case)](finance/robust_portfolio) | JOTA 2014 | Ellipsoidal uncertainty set as a cone program in cvxpy | [Notion](https://road-lamp-b0c.notion.site/3fa64a44030c810b9bcbcd48ff6aa45e) |
| Reinforcement learning | [AlphaGo](reinforcement-learning/alphago_lite) | Nature 2016 | Policy/value network + PUCT search + self-play on tic-tac-toe | [Notion](https://road-lamp-b0c.notion.site/3fa64a44030c8115a514d652c7af8e08) |

## Read and summarized, not yet implemented

- TadGAN (IEEE BigData 2020) · [Notion](https://road-lamp-b0c.notion.site/3f664a44030c8167913fd8cf9e199926)
- AER: Auto-Encoder with Regression (IEEE BigData 2022) · [Notion](https://road-lamp-b0c.notion.site/3f664a44030c81e3a6fddb0aa6230512)
- CATCH: Channel-Aware MTS Anomaly Detection via Frequency Patching (ICLR 2025) · [Notion](https://road-lamp-b0c.notion.site/3f664a44030c81129592f43e649bddb2)
- Time-LLM (ICLR 2024) · [Notion](https://road-lamp-b0c.notion.site/3f664a44030c8172baa8e90161849fdd)
- FOIL: OOD Time-Series Forecasting with Invariant Learning (ICML 2024) · [Notion](https://road-lamp-b0c.notion.site/3f664a44030c81fc9d3cd85f19dcc970)
- Semi-supervised Learning with Deep Generative Models (NeurIPS 2014) · [Notion](https://road-lamp-b0c.notion.site/3fa64a44030c81df9ab0fec30afb8971)
- Decoding the Beige Book: LLM-Powered Sentiment Analysis (ICAIF 2025) · [Notion](https://road-lamp-b0c.notion.site/3fa64a44030c81c0a1b1d3d5239dc22b)
- Reasoning or Overthinking: LLMs on Financial Sentiment Analysis (ICAIF 2025) · [Notion](https://road-lamp-b0c.notion.site/3fa64a44030c81eb9ab9f48c137f8e09)

## Setup
```bash
pip install -r requirements.txt
cd time-series/anomaly_transformer && python main.py
```
