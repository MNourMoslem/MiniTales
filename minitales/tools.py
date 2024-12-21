import torch
from typing import Any
import torch.nn.functional as F

def generate(model : torch.nn.Module,
            tokenizer : Any,
            prompt: str,
            device : str = None,
            max_length : int = 50,
            temperature : float = 1.0,
            top_k : int = 10,
            eos_token='<|eos|>'):
            
    model.eval()
    enc = tokenizer.encode(prompt)

    for _ in range(max_length):
        input_tensor = torch.tensor(enc).unsqueeze(0).to(device)
        output = model(input_tensor)  # Forward pass through the model

        logits = output[:, -1, :].squeeze(0)

        probabilities = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(probabilities, 1).item()

        if tokenizer.vocab[next_token] == eos_token:
          break

        enc.append(next_token)

    dec = tokenizer.decode(enc)

    return dec

def get_batch(tokens : torch.tensor, from_ : int, to_ : int, batch_size : int, block_size : int, device):
  indices = torch.randint(0, to_, (batch_size,)).to(device)  # Generate all random indices at once
  input_ = tokens[indices[:, None] + torch.arange(block_size, device=device)]  # Use broadcasting to create input
  target = tokens[indices[:, None] + torch.arange(1, block_size + 1, device=device)]
  return input_, target