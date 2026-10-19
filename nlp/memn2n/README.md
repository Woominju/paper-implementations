# End-To-End Memory Networks

NIPS 2015 | [Notion summary](https://road-lamp-b0c.notion.site/3fa64a44030c812c9c1cf7eb6318f37d) | [Paper](https://arxiv.org/abs/1503.08895) | [Official code](https://github.com/facebook/MemNN)

## What is in the code
- Soft attention over sentence memories with multiple hops (u <- u + o)
- Adjacent weight tying, position encoding and temporal encoding
- Attention weights printed per hop for one story

## Differences from the paper
A synthetic version of bAbI task 1, which needs only one supporting fact, so one hop already solves it; multi-hop tasks (e.g. task 2) are not included.

## Run
```bash
python main.py
```

## Output
```text
1 hop(s): test accuracy 0.999
3 hop(s): test accuracy 0.993

Story:
  0: daniel travelled to the garden      hop1=1.00 hop3=1.00
  1: sandra moved to the garden          hop1=0.00 hop3=0.00
Q: where is daniel?  prediction=garden  answer=garden
```
