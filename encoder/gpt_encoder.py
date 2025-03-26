import torch
import torch.nn as nn
from torch.nn import functional as F
from itertools import product
from matplotlib import pyplot as plt
import torch.optim as optim

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2


# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

###Understood until here

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

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
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
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

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C) #YOOOOO WHY IS IT -
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
'''Prove the basis pursuit problem

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
'''
import numpy as np

# Grid search space
batch_sizes = [32, 64, 128,256]
block_sizes = [64, 128, 256, 512]
n_layers = [4, 6, 8,10]
eval_iters_values = [100, 200, 300, 400]

# Store results for plotting
results = []
best_val_loss = np.inf
# Grid search loop
for batch_size in batch_sizes:
    for block_size in block_sizes:
        for n_layer in n_layers:
            for eval_iters in eval_iters_values:
                # Set hyperparameters
                print(f"Running with batch_size={batch_size}, block_size={block_size}, n_layer={n_layer}, eval_iters={eval_iters}")
                
                # Define the model with the current hyperparameters
                model = GPTLanguageModel()
                model.to(device)
                
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

                # Reinitialize training data and the batch generator for each combination
                train_data = torch.tensor(encode(text), dtype=torch.long)
                n = int(0.9 * len(train_data))  # 90% for training, 10% for validation
                val_data = train_data[n:]

                # Start training loop
                for iter in range(max_iters):
                    if iter % eval_interval == 0 or iter == max_iters - 1:
                        losses = estimate_loss()
                        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

                    xb, yb = get_batch('train')
                    logits, loss = model(xb, yb)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                # After training, evaluate the final loss
                losses = estimate_loss()
                train_loss = losses['train']
                val_loss = losses['val']
                
                if val_loss < best_val_loss:
                    best_model = model
                    best_val_loss = val_loss
                # Store results
                results.append((batch_size, block_size, n_layer, eval_iters, train_loss, val_loss))

# Convert results to a numpy array for easier processing
results_array = np.array(results)

# Find the best parameters (minimize validation loss)
best_result = results_array[np.argmin(results_array[:, 5])]  # column 5 is val_loss

best_batch_size, best_block_size, best_n_layer, best_eval_iters, best_train_loss, best_val_loss = best_result

# Output best hyperparameters
print(f"Best parameters: batch_size={best_batch_size}, block_size={best_block_size}, n_layer={best_n_layer}, eval_iters={best_eval_iters}")
print(f"Best validation loss: {best_val_loss:.4f}")

# Plot how the validation loss changes based on the hyperparameters

# Plot validation loss vs batch size
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for block_size in block_sizes:
    block_size_results = results_array[results_array[:, 1] == block_size]
    plt.plot(block_size_results[:, 0], block_size_results[:, 5], label=f'block_size={block_size}')
plt.xlabel('Batch Size')
plt.ylabel('Validation Loss')
plt.title('Validation Loss vs Batch Size')
plt.legend()

# Plot validation loss vs block size
plt.subplot(1, 2, 2)
for batch_size in batch_sizes:
    batch_size_results = results_array[results_array[:, 0] == batch_size]
    plt.plot(batch_size_results[:, 1], batch_size_results[:, 5], label=f'batch_size={batch_size}')
plt.xlabel('Block Size')
plt.ylabel('Validation Loss')
plt.title('Validation Loss vs Block Size')
plt.legend()

plt.tight_layout()
plt.show()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
#print(decode(best_model.generate(context, max_new_tokens=500)[0].tolist()))
open('generated.txt', 'w').write(decode(best_model.generate(context, max_new_tokens=1000)[0].tolist()))
