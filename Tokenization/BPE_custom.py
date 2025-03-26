import re
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import defaultdict

# ---------------- BPE Tokenizer ----------------
def basic_tokenize(text):
    return list(text)

def get_stats(tokens):
    pairs = defaultdict(int)
    for token in tokens:
        for i in range(len(token) - 1):
            pairs[(token[i], token[i + 1])] += 1
    return pairs

def merge_vocab(tokens, pair):
    new_tokens = []
    bigram = re.escape(' '.join(pair))
    pattern = re.compile(r'(?<!\\S)' + bigram + r'(?!\\S)')
    for token in tokens:
        token_str = ' '.join(token)
        token_str = pattern.sub(''.join(pair), token_str)
        new_tokens.append(token_str.split())
    return new_tokens

def train_bpe(text, vocab_limit):
    tokens = basic_tokenize(text)
    vocab = [' '.join(tokens)]
    vocab = [vocab[0].split()]
    merges = []

    while True:
        pairs = get_stats(vocab)
        if not pairs:
            break
        most_frequent = max(pairs, key=pairs.get)
        merges.append(most_frequent)
        vocab = merge_vocab(vocab, most_frequent)
        if len(set(sum(vocab, []))) >= vocab_limit:
            break
    return merges

def bpe_encode(text, merges):
    # Replace spaces and newlines with special characters
    text = text.replace(" ", "▁").replace("\n", "↩")
    tokens = list(text)
    token_str = ' '.join(tokens)
    for pair in merges:
        pattern = re.escape(' '.join(pair))
        token_str = re.sub(r'(?<!\\S)' + pattern + r'(?!\\S)', ''.join(pair), token_str)
    return token_str.split()

def bpe_decode(tokens):
    return ''.join(tokens).replace("▁", " ").replace("↩", "\n")

# ---------------- Hyperparameters ----------------
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 500
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2


#vocav size 2000
# batch_size = 16
# block_size = 32
# max_iters = 5000
# eval_interval = 100
# learning_rate = 1e-3
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 200
# n_embd = 64
# n_head = 4
# n_layer = 4
# dropout = 0.0

torch.manual_seed(1337)

# ---------------- Load Data ----------------
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace space and newline before BPE training
text = text.replace(" ", "▁").replace("\n", "↩")

# ---------------- BPE Tokenization ----------------
merges = train_bpe(text, vocab_limit=500)
encoded_text = bpe_encode(text, merges)
vocab = sorted(set(encoded_text))
stoi = {tok: i for i, tok in enumerate(vocab)}
itos = {i: tok for tok, i in stoi.items()}

vocab_size = len(vocab)
encode = lambda s: [stoi[tok] for tok in bpe_encode(s, merges)]
decode = lambda l: bpe_decode([itos[i] for i in l if i in itos])

data = torch.tensor([stoi[t] for t in encoded_text], dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_ = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_) - block_size, (batch_size,))
    x = torch.stack([data_[i:i+block_size] for i in ix])
    y = torch.stack([data_[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ---------------- GPT Model ----------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits = logits.view(B * T, vocab_size)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ---------------- Train ----------------
model = BigramLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Starting training...")
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ---------------- Generate ----------------
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
output = model.generate(context, max_new_tokens=100)
print("\n📜 Generated Text:")
print(decode(output[0].tolist()))

# ---------------- Evaluate Perplexity ----------------
loss = estimate_loss()['val']
perplexity = torch.exp(loss)
print(f"\n Validation Perplexity: {perplexity.item():.2f}")
