"""
End-To-End Memory Networks (Sukhbaatar et al., NIPS 2015), minimal PyTorch version.

Task: a synthetic version of bAbI task 1 ("single supporting fact").
    Mary went to the kitchen. John moved to the garden. ... Where is Mary?  -> kitchen

Model (adjacent weight tying, position encoding, temporal encoding):
    m_i = sum_j l_j * A x_ij + T_A(i)        memory vectors
    c_i = sum_j l_j * C x_ij + T_C(i)        output vectors
    p   = softmax(u^T m_i)                   soft attention over memories
    o   = sum_i p_i c_i ;  u <- u + o        one hop; repeated K times
    a   = softmax(W u)
The script compares 1 hop with 3 hops and prints the attention of a trained
3-hop model on one story.
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(0); torch.manual_seed(0)

PEOPLE = ["mary", "john", "sandra", "daniel"]
PLACES = ["kitchen", "garden", "office", "hallway", "bedroom", "bathroom"]
VERBS = ["went to", "moved to", "travelled to", "journeyed to"]
MEM, SENT = 10, 6                        # max memories, max words per sentence

def make_story():
    n = random.randint(2, MEM)
    story, loc = [], {}
    for _ in range(n):
        p, pl = random.choice(PEOPLE), random.choice(PLACES)
        story.append(f"{p} {random.choice(VERBS)} the {pl}".split()); loc[p] = pl
    who = random.choice(list(loc))
    return story, ["where", "is", who], loc[who]

vocab = ["<pad>"] + sorted({w for _ in range(2000) for s in [make_story()]
                            for w in sum(s[0], []) + s[1] + [s[2]]})
w2i = {w: i for i, w in enumerate(vocab)}

def vec(sent):
    return [w2i[w] for w in sent] + [0] * (SENT - len(sent))

def dataset(n):
    S, Q, A = [], [], []
    for _ in range(n):
        story, q, a = make_story()
        mem = [vec(s) for s in story[-MEM:]]
        S.append(mem + [[0] * SENT] * (MEM - len(mem))); Q.append(vec(q)); A.append(w2i[a])
    return torch.tensor(S), torch.tensor(Q), torch.tensor(A)

def position_encoding(J, d):
    l = np.zeros((J, d))
    for j in range(1, J + 1):
        for k in range(1, d + 1):
            l[j - 1, k - 1] = (1 - j / J) - (k / d) * (1 - 2 * j / J)
    return torch.tensor(l, dtype=torch.float)


class MemN2N(nn.Module):
    def __init__(self, V, d=32, hops=3):
        super().__init__()
        self.hops = hops
        # adjacent tying: A^{k+1} = C^k, B = A^1, W^T = C^K
        self.emb = nn.ModuleList([nn.Embedding(V, d, padding_idx=0) for _ in range(hops + 1)])
        self.T = nn.ParameterList([nn.Parameter(torch.randn(MEM, d) * 0.1) for _ in range(hops + 1)])
        self.register_buffer("pe", position_encoding(SENT, d))

    def encode(self, x, k):
        return (self.emb[k](x) * self.pe).sum(-2)           # bag of words with position weights

    def forward(self, story, q, return_attn=False):
        u = self.encode(q, 0)                               # B = A^1
        attns = []
        for k in range(self.hops):
            m = self.encode(story, k) + self.T[k]           # (B, MEM, d)
            c = self.encode(story, k + 1) + self.T[k + 1]
            empty = (story.sum(-1) == 0)
            p = F.softmax((m @ u.unsqueeze(-1)).squeeze(-1).masked_fill(empty, -1e9), dim=-1)
            u = u + (p.unsqueeze(-1) * c).sum(1)
            attns.append(p)
        logits = u @ self.emb[self.hops].weight.T           # W^T = C^K
        return (logits, attns) if return_attn else logits


Str, Qtr, Atr = dataset(5000)
Ste, Qte, Ate = dataset(1000)

def train(hops, epochs=40):
    torch.manual_seed(0)
    model = MemN2N(len(vocab), hops=hops)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for ep in range(epochs):
        perm = torch.randperm(len(Str))
        for b in range(0, len(Str), 64):
            i = perm[b:b + 64]
            loss = F.cross_entropy(model(Str[i], Qtr[i]), Atr[i])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 40.0); opt.step()
    with torch.no_grad():
        acc = (model(Ste, Qte).argmax(1) == Ate).float().mean().item()
    return model, acc

for hops in (1, 3):
    model, acc = train(hops)
    print(f"{hops} hop(s): test accuracy {acc:.3f}")

# Inspect attention of the 3-hop model on one story
story, q, a = make_story()
S = torch.tensor([[vec(s) for s in story] + [[0] * SENT] * (MEM - len(story))])
with torch.no_grad():
    logits, attns = model(S, torch.tensor([vec(q)]), return_attn=True)
print("\nStory:")
for i, s in enumerate(story):
    print(f"  {i}: {' '.join(s):35s} hop1={attns[0][0, i]:.2f} hop3={attns[-1][0, i]:.2f}")
print(f"Q: {' '.join(q)}?  prediction={vocab[logits.argmax()]}  answer={a}")
