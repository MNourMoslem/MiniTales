import torch
from torch import nn
import torch.nn.functional as F

class Head(nn.Module):
  def __init__(self, embed_size, head_size, max_len, dropout):
    super().__init__()

    self.head_size = head_size

    self.Q = nn.Linear(embed_size, head_size)
    self.K = nn.Linear(embed_size, head_size)
    self.V = nn.Linear(embed_size, head_size)

    self.register_buffer('tril', torch.tril(torch.ones(max_len, max_len)))

    self.dropout = nn.Dropout(dropout)

  def forward(self, x, masked = False):
    B, T, E = x.shape # B, T, E

    q = self.Q(x) # B, T, H
    k = self.K(x) # B, T, H

    w = q @ k.transpose(-2, -1) * self.head_size ** -0.5 # B, T, H @ B, H, T -> B, T, T
    if masked and self.training:
      w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # B, T, T
    w = F.softmax(w, -1) # B, T, T
    w = self.dropout(w) # B, T, T

    v = self.V(x) # B, T, H
    out = w @ v # B, T, T @ # B, T, H -> # B, T, H

    return out # B, T, H

class MultiHeadAttention(nn.Module):
  def __init__(self, embed_size, n_heads, max_len, masked, dropout = 0.2):
    super().__init__()
    
    head_size = embed_size // n_heads

    assert head_size * n_heads == embed_size , "n_heads must be truly divisible on embed_size"

    self.heads = nn.ModuleList([Head(embed_size, head_size, max_len, dropout) for _ in range(n_heads)])
    self.proj = nn.Linear(head_size * n_heads, embed_size)
    self.dropout = nn.Dropout(dropout)

    self.masked = masked

  def forward(self, x):
    out = torch.cat([h(x, self.masked) for h in self.heads], dim = -1)
    out = self.dropout(self.proj(x))
    return out


class FeedFoward(nn.Module):
  def __init__(self, embed_size, dropout = 0.2):
    super().__init__()
    
    self.net = nn.Sequential(
        nn.Linear(embed_size, 4 * embed_size),
        nn.ReLU(),
        nn.Linear(4 * embed_size, 6 * embed_size),
        nn.ReLU(),
        nn.Linear(6 * embed_size, 4 * embed_size),
        nn.ReLU(),
        nn.Linear(4 * embed_size, embed_size),
    )

    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    x = self.net(x)
    x = self.dropout(x)
    return x

class Block(nn.Module):
  def __init__(self, embed_size, n_heads, max_len, dropout):
    super().__init__()
    self.sa = MultiHeadAttention(embed_size, n_heads, max_len, True, 0.2)
    self.ffwd = FeedFoward(embed_size, dropout)
    self.ln1 = nn.LayerNorm(embed_size)
    self.ln2 = nn.LayerNorm(embed_size)

  def forward(self, x):
      x = x + self.sa(self.ln1(x))
      x = x + self.ffwd(self.ln2(x))
      return x


class MT(nn.Module):
  def __init__(self, embed_size, vocab_size, n_layers, n_heads, max_len, dropout):
    super().__init__()

    self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
    self.position_embedding_table = nn.Embedding(max_len, embed_size)

    self.blocks = nn.Sequential(*[Block(embed_size, n_heads, max_len, dropout) for _ in range(n_layers)])
    self.ln = nn.LayerNorm(embed_size)
    self.lm_head = nn.Linear(embed_size, vocab_size)

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

      tok_emb = self.token_embedding_table(idx) # (B,T,C)
      pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
      x = tok_emb + pos_emb # (B,T,C)
      x = self.blocks(x) # (B,T,C)
      x = self.ln(x) # (B,T,C)
      logits = self.lm_head(x) # (B,T,vocab_size)

      return logits