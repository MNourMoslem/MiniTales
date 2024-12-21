import torch
from .tools import generate, get_batch
from typing import Any

def train(model : torch.nn.Module,
          optimizer : torch.optim,
          criterion,
          tokens : torch.tensor,
          block_size : int,
          batch_size : int,
          device : str = None,
          epochs : int = 20,
          sub_epochs : int = 1000,
          n_samples : int = 100,
          scheduler : torch.optim.lr_scheduler = None,
          prompt_to_test : str = None,
          tokenizer : Any = None,
          generate_kwargs : dict = {}):
    """
    Train the model using the given optimizer and criterion.
    """

    upper = len(tokens) - block_size - 1

    losses = []

    print(f'Training for {epochs} epochs with {sub_epochs} sub-epochs each')
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        total_loss = 0

        for _ in range(sub_epochs):  # Manually generating batches
            input_, target = get_batch(tokens, 0, upper, batch_size, block_size, device) # Use broadcasting to create target

            optimizer.zero_grad()  # Zero the parameter gradients

            output = model(input_, device = device).transpose(1, 2)  # Transpose for CrossEntropyLoss

            loss = criterion(output, target)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize the model parameters

            total_loss += loss.item()

        avg_loss = total_loss / sub_epochs  # Corrected averaging
        losses.append(avg_loss)

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

        if scheduler:
            scheduler.step()

        if prompt_to_test != None and tokenizer != None:
            print("Prompt: {0}".format(prompt_to_test))
            answer = generate(model, tokenizer, prompt_to_test, device=device, *generate_kwargs)
            print("Answer: {0}\n".format(answer))

    return losses
