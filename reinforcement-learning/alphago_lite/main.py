"""
AlphaGo's core loop in miniature (Silver et al., Nature 2016) on tic-tac-toe.

AlphaGo combines a policy network (move priors), a value network (position
evaluation) and Monte Carlo tree search. This script keeps that structure:
  * one network with a policy head p(a|s) and a value head v(s)
  * PUCT search: a* = argmax Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
    leaves are evaluated by v(s) instead of random rollouts
  * self-play: the search's visit counts become policy targets and game
    outcomes become value targets
Unlike AlphaGo, there is no supervised stage on human games (as in AlphaGo
Zero), and the game is tiny so everything runs on a CPU in about a minute.
Evaluation uses only 15 simulations per move so the learned networks matter;
against a perfect (minimax) player the best possible result is a draw.
"""
import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(0); np.random.seed(0); torch.manual_seed(0)
LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def winner(b):
    for i, j, k in LINES:
        if b[i] != 0 and b[i] == b[j] == b[k]:
            return b[i]
    return 0 if 0 in b else 2          # 2 = draw

def legal(b): return [i for i in range(9) if b[i] == 0]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(18, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.pi, self.v = nn.Linear(64, 9), nn.Linear(64, 1)
    def forward(self, x):
        h = self.body(x)
        return self.pi(h), torch.tanh(self.v(h)).squeeze(-1)

def encode(b, player):                   # board from the perspective of the player to move
    b = np.array(b) * player
    return torch.tensor(np.concatenate([b == 1, b == -1]).astype(np.float32))


def mcts(net, board, player, sims=60, c=1.5):
    N, W, P = {}, {}, {}
    def search(b, p):
        w = winner(b)
        if w: return 0.0 if w == 2 else (1.0 if w == p else -1.0)
        key = (tuple(b), p)
        if key not in P:                                   # expand leaf with the networks
            with torch.no_grad():
                logits, v = net(encode(b, p))
            mask = torch.full((9,), -1e9); mask[legal(b)] = 0
            P[key] = F.softmax(logits + mask, -1).numpy()
            N[key], W[key] = np.zeros(9), np.zeros(9)
            return v.item()
        n_total = N[key].sum()
        ucb = [W[key][a] / (N[key][a] + 1e-9) * (N[key][a] > 0)
               + c * P[key][a] * math.sqrt(n_total + 1) / (1 + N[key][a]) for a in legal(b)]
        a = legal(b)[int(np.argmax(ucb))]
        b2 = list(b); b2[a] = p
        val = -search(b2, -p)                              # value for the player to move here
        N[key][a] += 1; W[key][a] += val
        return val
    for _ in range(sims):
        search(list(board), player)
    return N[(tuple(board), player)]


def self_play(net, temp_moves=3):
    b, p, hist = [0] * 9, 1, []
    while not winner(b):
        visits = mcts(net, b, p)
        pi = visits / visits.sum()
        a = np.random.choice(9, p=pi) if len(hist) < temp_moves else int(visits.argmax())
        hist.append((encode(b, p), torch.tensor(pi, dtype=torch.float), p))
        b[a] = p; p = -p
    w = winner(b)
    return [(x, pi, 0.0 if w == 2 else (1.0 if w == pl else -1.0)) for x, pi, pl in hist]


from functools import lru_cache

@lru_cache(maxsize=None)
def minimax(b, p):
    """Perfect play: returns (value for player p, best move)."""
    w = winner(list(b))
    if w: return (0 if w == 2 else (1 if w == p else -1)), None
    best = (-2, None)
    for a in legal(b):
        b2 = list(b); b2[a] = p
        v = -minimax(tuple(b2), -p)[0]
        if v > best[0]: best = (v, a)
    return best

def evaluate(net, opponent, games=40, sims=15):
    """Play with a small search budget so the networks matter."""
    results = {"win": 0, "draw": 0, "loss": 0}
    for g in range(games):
        b, p, me = [0] * 9, 1, 1 if g % 2 == 0 else -1
        while not winner(b):
            if p == me:
                a = int(mcts(net, b, p, sims=sims).argmax())
            else:
                a = random.choice(legal(b)) if opponent == "random" else minimax(tuple(b), p)[1]
            b[a] = p; p = -p
        w = winner(b)
        results["draw" if w == 2 else ("win" if w == me else "loss")] += 1
    return results


net = Net(); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
print("before training: vs random", evaluate(net, "random"), "| vs perfect player", evaluate(net, "perfect"))
buffer = []
for it in range(15):
    for _ in range(20):
        buffer += self_play(net)
    buffer = buffer[-3000:]
    for _ in range(200):
        batch = random.sample(buffer, min(64, len(buffer)))
        x = torch.stack([b[0] for b in batch]); pi = torch.stack([b[1] for b in batch])
        z = torch.tensor([b[2] for b in batch])
        logits, v = net(x)
        loss = F.mse_loss(v, z) - (pi * F.log_softmax(logits, -1)).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    if it % 5 == 4:
        print(f"iteration {it + 1}: loss {loss.item():.3f} | vs random {evaluate(net, 'random')} | vs perfect {evaluate(net, 'perfect')}")
