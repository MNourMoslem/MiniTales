import torch
import torch.nn.functional as F

def generate(model, tokenizer, device, prompt: str, max_length=50, temperature=1.0, top_k=10, eos_token='<|eos|>'):
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