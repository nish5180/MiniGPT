import torch
import torch.nn as nn
from torch.nn import functional as F
import csv
import unicodedata

# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 1
eval_interval = 5000
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.4

BOS_token = "<BOS>"
EOS_token = "<EOS>"

encoder_texts = []
decoder_texts = []

csv_file = "pair_dataset.csv"  
with open(csv_file, newline='', encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        encoder_line = row["encoder_input"].strip()
        decoder_line = row["decoder_target"].strip()
        decoder_line_with_tokens = f"{BOS_token} {decoder_line} {EOS_token}"
        encoder_texts.append(encoder_line)
        decoder_texts.append(decoder_line_with_tokens)

all_texts = encoder_texts + decoder_texts
chars = sorted(list(set("".join(all_texts)) | set([BOS_token, EOS_token])))  # Convert string to set before union
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
def pad_sequence(seq, max_length, pad_token=0):
    if len(seq) > max_length:
        return seq[:max_length]  # Truncate if sequence is too long
    return seq + [pad_token] * (max_length - len(seq))  # Pad if sequence is too short

max_length_enc = block_size
max_length_dec = block_size

enc_data_full = torch.tensor(
    [pad_sequence(encode(text), max_length_enc) for text in encoder_texts],
    dtype=torch.long
)
dec_data_full = torch.tensor(
    [pad_sequence(encode(text), max_length_dec) for text in decoder_texts],
    dtype=torch.long
)

#print(enc_data_full)
#print(dec_data_full)


n = int(0.9 * len(enc_data_full))
train_enc_data = enc_data_full[:n]
val_enc_data = enc_data_full[n:]

train_dec_data = dec_data_full[:n]
val_dec_data = dec_data_full[n:]

# Batch generation function
def get_paired_batch(split="train"):
    if split == "train":
        enc_data = train_enc_data
        dec_data = train_dec_data
    else:
        enc_data = val_enc_data
        dec_data = val_dec_data

    enc_indices = torch.randint(0, len(enc_data), (batch_size,))
    x_enc = torch.stack([enc_data[i][:block_size] for i in enc_indices])

    dec_indices = torch.randint(0, len(dec_data), (batch_size,))
    y_dec = torch.stack([dec_data[i][:block_size] for i in dec_indices])  # Decoder target

    # Create decoder input by shifting the decoder target right and adding <BOS>
    bos_token_index = stoi[BOS_token]
    x_dec = torch.cat(
        [torch.full((batch_size, 1), bos_token_index, dtype=torch.long), y_dec[:, :-1]],
        dim=1
    )

    # Create padding masks
    enc_padding_mask = (x_enc != 0).to(device)  # Mask for encoder input
    dec_padding_mask = (x_dec != 0).to(device)  # Mask for decoder input

    return x_enc.to(device), x_dec.to(device), y_dec.to(device), enc_padding_mask, dec_padding_mask
class Head(nn.Module):
    """ One head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, hs)
        q = self.query(x) # (B, T, hs)
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # Apply causal mask (B, T, T)

        if mask is not None:
            mask = mask.unsqueeze(1)  # Reshape mask to (B, 1, T) for broadcasting
            wei = wei.masked_fill(mask == 0, float('-inf'))  # Mask padding tokens (B, T, T)

        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # Perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)

        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        out = torch.cat([h(x, mask=mask) for h in self.heads], dim=-1)
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
        print(idx.shape)

        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        
        pos_emb = self.position_embedding_table(torch.arange(T, device= device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        return x
    
class CrossAttention(nn.Module):
    """ Cross-attention layer for attending to encoder output """

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        #self.proj = nn.Linear(head_size * n_head, n_embd)
        #self.proj = nn.Linear(head_size, n_embd)
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
        return out

class MultiHeadCrossAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([CrossAttention(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out):
        out = torch.cat([h(x,enc_out) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    

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
        self.cross_attn = MultiHeadCrossAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

    def forward(self, x, enc_out, enc_padding_mask=None, dec_padding_mask=None):
        x = x + self.self_attn(self.ln1(x), mask=dec_padding_mask)
        x = x + self.cross_attn(self.ln2(x), enc_out)
        x = x + self.ffwd(self.ln3(x))
        return x
   

class GPTLanguageModelWithEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder(n_embd, n_head, n_layer, vocab_size, block_size)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[DecoderBlockWithCrossAttention(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, encoder_input=None, targets=None, enc_padding_mask=None, dec_padding_mask=None):
        if encoder_input is not None:
            enc_out = self.encoder(encoder_input)  # Shape: (B_enc, T_enc, n_embd)
        else:
            enc_out = None
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb

        # Pass the decoder embeddings along with the encoder output and masks into the decoder blocks
        for block in self.blocks:
            x = block(x, enc_out, enc_padding_mask, dec_padding_mask)
        #x = self.blocks(x, enc_out, enc_padding_mask, dec_padding_mask) # (B,T,C)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets, ignore_index=0)  # Ignore padding token in loss

        return logits, loss

    
    def generate(self, idx, encoder_input=None, max_new_tokens=500):
    # Process encoder input if provided
        if encoder_input is not None:
            # Ensure encoder_input is raw token indices (shape: [B, T_enc])
            assert len(encoder_input.shape) == 2, f"Expected encoder_input to have shape (B, T_enc), got {encoder_input.shape}"
            enc_out = self.encoder(encoder_input)  # Shape: (B, T_enc, C)
        else:
            enc_out = None

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # Truncate to block_size
            # Ensure idx_cond is raw token indices (shape: [B, T])
            assert len(idx_cond.shape) == 2, f"Expected idx_cond to have shape (B, T), got {idx_cond.shape}"

            # Forward pass with encoder output
            logits, _ = self(idx_cond, encoder_input=enc_out)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# Training loop
model = GPTLanguageModelWithEncoder().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    x_enc, x_dec, y_dec, enc_padding_mask, dec_padding_mask = get_paired_batch("train")
    logits, loss = model(x_dec, encoder_input=x_enc, targets=y_dec, enc_padding_mask=enc_padding_mask, dec_padding_mask=dec_padding_mask)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % eval_interval == 0 or iter == max_iters - 1:
        model.eval()
        losses = []
        for _ in range(eval_iters):
            x_enc_val, x_dec_val, y_dec_val, enc_padding_mask_val, dec_padding_mask_val = get_paired_batch("val")
            with torch.no_grad():
                _, l = model(x_dec_val, encoder_input=x_enc_val, targets=y_dec_val, enc_padding_mask=enc_padding_mask_val, dec_padding_mask=dec_padding_mask_val)
            losses.append(l.item())
        avg_loss = sum(losses) / len(losses)
        print(f"Step {iter}: loss {avg_loss:.4f}")
        model.train()

# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Starting token
encoder_input = train_enc_data[0:1].to(device)  # Ensure batch dimension is retained
print(f"context shape: {context.shape}")
print(f"encoder_input shape: {encoder_input.shape}")
generated = model.generate(context, encoder_input=encoder_input, max_new_tokens=500)
with open('more.txt', 'w', encoding='utf-8') as f:
    f.write(decode(generated[0].tolist()))