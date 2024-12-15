import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
  def __init__(self, n_embedding):
    super().__init__()

    self.net = nn.Sequential(
        nn.Linear(n_embedding, 512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, n_embedding)
    )

  def forward(self, x):
    return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, block_size, n_embedding, n_heads, masked=False):
        super().__init__()

        self.n_embedding = n_embedding
        self.n_heads = n_heads
        self.head_dim = n_embedding // n_heads
        assert self.head_dim * n_heads == n_embedding, "Embedding dimension must be divisible by the number of heads"

        self.masked = masked

        self.Q = nn.Linear(n_embedding, n_embedding, bias=False)
        self.K = nn.Linear(n_embedding, n_embedding, bias=False)
        self.V = nn.Linear(n_embedding, n_embedding, bias=False)

        self.fc_out = nn.Linear(n_embedding, n_embedding)

        if masked:
            self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))

    def forward(self, x):
        batch_size, seq_length, _ = x.shape

        q = self.Q(x)  # (B, L, E)
        k = self.K(x)  # (B, L, E)
        v = self.V(x)  # (B, L, E)

        # Reshape into multiple heads
        q = q.view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2) # (B, N, L, H)
        k = k.view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2) # (B, N, L, H)
        v = v.view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2) # (B, N, L, H)

        # Attention calculation
        wei = q @ k.transpose(-2, -1) / self.head_dim ** 0.5 # (B, N, L, L)
        if self.masked and self.training:
          wei = wei.masked_fill(self.tril[:seq_length, :seq_length] == 0, float('-inf'))

        attention = F.softmax(wei, dim=-1) #(B, N, L)

        out = attention @ v  # (batch_size, n_heads, seq_length, head_dim)

        # Concatenate heads and pass through final linear layer
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.n_embedding)
        out = self.fc_out(out)

        return out

class Decoder(nn.Module):
    def __init__(self, vocab_size, n_embedding, block_size, n_heads):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, n_embedding)
        self.pos_embedding = nn.Embedding(block_size, n_embedding)

        self.masked_head = MultiHeadAttention(block_size, n_embedding, n_heads, masked=True)
        self.head = MultiHeadAttention(block_size, n_embedding, n_heads, masked=False)
        self.feedforward = FeedForward(n_embedding)

        self.layernorm = nn.LayerNorm(n_embedding)

    def forward(self, x):
        token_embeds = self.token_embedding(x)
        pos_embeds = self.pos_embedding(torch.arange(x.size(1), device=x.device))

        embeds = token_embeds + pos_embeds

        masked_res = self.masked_head(embeds)
        add_norm_res = self.layernorm(embeds + masked_res)

        head_res = self.head(add_norm_res)
        add_norm_res = self.layernorm(add_norm_res + head_res)

        feed_res = self.feedforward(add_norm_res)
        add_norm_res = self.layernorm(add_norm_res + feed_res)

        return add_norm_res

class Transformer(nn.Module):

  def __init__(self, vocab_size, n_embedding, block_size, n_heads):
    super().__init__()

    self.decoder = Decoder(vocab_size, n_embedding, block_size, n_heads)
    self.output = nn.Linear(n_embedding, vocab_size)

  def forward(self, x):
    x = self.decoder(x)
    return self.output(x)
