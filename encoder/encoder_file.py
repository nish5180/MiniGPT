import torch
import torch.nn as nn
from torch.nn import functional as F


from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import torch
# hyperparameters
batch_size = 64 #64 # how many independent sequences will we process in parallel?
block_size = 256#256 # what is the maximum context length for predictions?
max_iters = 5000#5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
n_embd = 384#384
n_head = 6#6
n_layer = 6#6
dropout = 0.2#0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

with open('banana.txt', 'r', encoding='utf-8') as f:
    text_enc = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

chars_enc = sorted(list(set(text_enc)))
vocab_size_enc = len(chars_enc)
stoi_enc = { ch:i for i,ch in enumerate(chars_enc) }
itos_enc = { i:ch for i,ch in enumerate(chars_enc) }
encode_enc = lambda s_enc: [stoi_enc[c] for c in s_enc] # encoder: take a string, output a list of integers
decode_enc = lambda l_enc: ''.join([itos_enc[i] for i in l_enc]) # decoder: take a list of integers, output a string
print(vocab_size, vocab_size_enc)


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

enc_data = torch.tensor(encode_enc(text_enc), dtype=torch.long)
n_enc = int(0.9*len(enc_data))
enc_data = enc_data[:n_enc]

ic = torch.randint(len(enc_data) - block_size, (batch_size,))
enc_data = torch.stack([enc_data[i:i+block_size] for i in ic])
enc_data = enc_data.to(device)

#val_enc_data = enc_data[n_enc:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    #datab = data if block == 'decoder' else enc_data
    #datab = datab[:n] if split == 'train' else datab[n:]
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
            logits, loss = model(X, targets = Y)
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

class Encoder(nn.Module):
    """ Encoder with stacked transformer blocks """
    
    def __init__(self, n_embd, n_head, n_layer, vocab_size_enc, block_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size_enc, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[EncoderBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        
        pos_emb = self.position_embedding_table(torch.arange(T, device= device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        return x
    
#THIS IS NOT MULTILAYER
class CrossAttention(nn.Module):
    """ Cross-attention layer for attending to encoder output """

    def __init__(self, n_embd, n_head):
        super().__init__()

        head_size = n_embd // n_head       
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        #self.proj = nn.Linear(head_size * n_head, n_embd)
        self.proj = nn.Linear(head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out):
        B, T, C = x.shape
        B_enc, T_enc, C_enc = enc_out.shape
        
        # Compute queries from decoder input and keys/values from encoder output
        k = self.key(enc_out)  # (B_enc, T_enc, hs)
        q = self.query(x)      # (B, T, hs)
        v = self.value(enc_out) # (B_enc, T_enc, hs)

    
        # Attention weights (B, T, T_enc)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # Weighted sum of values (B, T, hs)
        out = wei @ v
                
        #return out
        return self.proj(out)
'''
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
        return out'
'''
class EncoderBlock(nn.Module):
    """ Transformer encoder block """
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)  # self-attention
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
class DecoderBlockWithCrossAttention(nn.Module):
    """ Decoder block with self-attention, cross-attention, and feed-forward """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.self_attn = MultiHeadAttention(n_head, head_size)
        self.cross_attn = CrossAttention(n_embd, n_head)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

    def forward(self, x_comb):
        x = x_comb[:, :block_size, :]
        enc_out = x_comb[:, block_size:, :]
        x = x + self.self_attn(self.ln1(x))        # self-attention
        x = x + self.cross_attn(self.ln2(x), enc_out)  # cross-attention with encoder output
        x = x + self.ffwd(self.ln3(x))             # feed-forward
        return x


class GPTLanguageModelWithEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder(n_embd, n_head, n_layer, vocab_size_enc, block_size)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[DecoderBlockWithCrossAttention(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        #self.apply(self._init_weights)

    def forward(self, idx, encoder_input = None, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.ln_f(x)
        # Get encoder output from encoder_input (xb_enc)
        if encoder_input is not None:
            enc_out = self.encoder(encoder_input)  # shape: (B_enc, T_enc, n_embd)
            x_comb = torch.cat((x, enc_out), dim=1)
        else: 
            x_comb = x    
        # Pass the decoder embeddings along with the encoder output into the decoder blocks.
        x = self.blocks(x_comb)  
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
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
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx





model = GPTLanguageModelWithEncoder()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):


    xb, yb = get_batch('train')

    logits, loss = model(xb, enc_data, targets=yb)  
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data for the decoder
    # Sample a batch of data for the encoder
    
    
    
    
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
#print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=5000)[0].tolist()))




