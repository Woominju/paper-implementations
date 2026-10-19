# BALD & BatchBALD

*arXiv 2011 / NeurIPS 2019* · [Notion summary](https://road-lamp-b0c.notion.site/3fa64a44030c8127886ef440cbf2dd98) · [Paper](https://arxiv.org/abs/1112.5745) · [Official code](https://github.com/BlackHC/BatchBALD)

## What this implements
- BALD score I[y; w | x] = H[E_w p(y|x,w)] - E_w H[p(y|x,w)] estimated with MC dropout
- BatchBALD-style greedy batch selection: exact joint entropy while small, then importance sampling over label configurations
- Random vs. BALD vs. BatchBALD acquisition on scikit-learn digits (3 seeds)

## Simplifications
A small MLP with dropout instead of a Bayesian CNN; the greedy batch only considers the 100 points with the highest BALD score. In this small setting BatchBALD does not beat top-k BALD.

## Run
```bash
python main.py
```

## Output
```text
Test accuracy vs number of labeled examples (mean of 3 seeds)
labels   random    BALD   BatchBALD
    20   0.571   0.571   0.571
    40   0.767   0.838   0.754
    60   0.848   0.874   0.843
    80   0.870   0.888   0.894
   100   0.888   0.918   0.917
   120   0.907   0.930   0.930
```
