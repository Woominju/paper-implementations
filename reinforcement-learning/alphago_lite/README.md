# Mastering the game of Go with deep neural networks and tree search

*Nature, 2016* · [Notion summary](https://road-lamp-b0c.notion.site/3fa64a44030c8115a514d652c7af8e08) · [Paper](https://www.nature.com/articles/nature16961)

## What this implements
- Policy and value heads on one network
- PUCT tree search with leaves evaluated by the value network
- Self-play training with visit counts as policy targets and game results as value targets

## Simplifications
Tic-tac-toe instead of Go, and no supervised stage on human games (closer to AlphaGo Zero). Against a perfect minimax player the agent goes from losing every game to drawing half of them.

## Run
```bash
python main.py
```

## Output
```text
before training: vs random {'win': 33, 'draw': 1, 'loss': 6} | vs perfect player {'win': 0, 'draw': 0, 'loss': 40}
iteration 5: loss 1.289 | vs random {'win': 28, 'draw': 8, 'loss': 4} | vs perfect {'win': 0, 'draw': 0, 'loss': 40}
iteration 10: loss 1.100 | vs random {'win': 36, 'draw': 4, 'loss': 0} | vs perfect {'win': 0, 'draw': 20, 'loss': 20}
iteration 15: loss 0.980 | vs random {'win': 35, 'draw': 3, 'loss': 2} | vs perfect {'win': 0, 'draw': 20, 'loss': 20}
```
