"""
GPT-1 in miniature (Radford et al., 2018): generative pre-training, then
discriminative fine-tuning with an auxiliary language-modeling loss.

  Stage 1  Pre-train a small Transformer decoder as a language model on
           unlabeled sentences.
  Stage 2  Fine-tune on a tiny labeled set with a linear head on the last
           token's hidden state:  L = L_cls + lambda * L_lm  (lambda = 0.5).

The script compares fine-tuning from the pre-trained weights against training
the same network from scratch on the same 40 labeled examples.
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(0); torch.manual_seed(0)

# ----------------------------------------------------------------------
# Synthetic data: product reviews; label = positive (1) / negative (0)
# ----------------------------------------------------------------------
ITEMS = ["phone", "laptop", "camera", "headset", "watch", "tablet", "speaker", "keyboard"]
POS = ["great", "excellent", "amazing", "reliable", "fantastic", "solid"]
NEG = ["terrible", "awful", "broken", "useless", "disappointing", "flimsy"]
TEMPLATES = ["the {i} is {a}", "this {i} feels {a} to use", "my new {i} was {a}",
             "i think the {i} is really {a}", "honestly the {i} looks {a}"]

def sentence(label):
    adj = random.choice(POS if label else NEG)
    return random.choice(TEMPLATES).format(i=random.choice(ITEMS), a=adj).split()

words = sorted({w for t in TEMPLATES for w in t.replace("{i}", "").replace("{a}", "").split()}
               | set(ITEMS) | set(POS) | set(NEG))
vocab = ["<pad>", "<s>", "<e>"] + words
w2i = {w: i for i, w in enumerate(vocab)}
MAXLEN = 10

def encode(toks):
    ids = [w2i["<s>"]] + [w2i[t] for t in toks] + [w2i["<e>"]]
    return ids + [0] * (MAXLEN - len(ids))

unlabeled = torch.tensor([encode(sentence(random.random() < 0.5)) for _ in range(4000)])
def labeled(n):
    ys = [i % 2 for i in range(n)]
    return torch.tensor([encode(sentence(y)) for y in ys]), torch.tensor(ys)
Xtr, ytr = labeled(40)          # tiny labeled set
Xte, yte = labeled(1000)


# ----------------------------------------------------------------------
# Transformer decoder (masked self-attention only)
# ----------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, V, d=64, layers=2, heads=4):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(MAXLEN, d)
        block = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, V, bias=False)
        self.lm_head.weight = self.tok.weight            # tied input/output embeddings
        self.cls_head = nn.Linear(d, 2)                  # added only for fine-tuning
        self.register_buffer("mask", nn.Transformer.generate_square_subsequent_mask(MAXLEN))

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.size(1)))
        return self.ln(self.blocks(h, mask=self.mask, is_causal=True))

    def lm_loss(self, x):
        logits = self.lm_head(self.forward(x)[:, :-1])
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), ignore_index=0)

    def classify(self, x):
        h = self.forward(x)
        last = (x != 0).sum(1) - 1                       # position of the <e> token
        return self.cls_head(h[torch.arange(len(x)), last])


def pretrain(model, epochs=6):
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for ep in range(epochs):
        perm = torch.randperm(len(unlabeled)); tot = 0
        for b in range(0, len(unlabeled), 64):
            loss = model.lm_loss(unlabeled[perm[b:b + 64]])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        print(f"  pre-train epoch {ep}  LM loss {tot / (len(unlabeled) / 64):.3f}")


def finetune(model, lam=0.5, steps=150):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(steps):
        loss = F.cross_entropy(model.classify(Xtr), ytr) + lam * model.lm_loss(Xtr)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return (model.classify(Xte).argmax(1) == yte).float().mean().item()


print("Stage 1: generative pre-training on 4,000 unlabeled sentences")
pre = GPT(len(vocab)); pre.train(); pretrain(pre)
acc_pre = finetune(pre)

torch.manual_seed(0)
acc_scratch = finetune(GPT(len(vocab)))

print("\nStage 2: fine-tuning on 40 labeled sentences, test on 1,000")
print(f"  from scratch          : {acc_scratch:.3f}")
print(f"  pre-trained + finetune: {acc_pre:.3f}")
