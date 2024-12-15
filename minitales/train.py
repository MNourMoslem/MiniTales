import torch

def evaluate(model, criterion, tokens, block_size, batch_size, n_samples, device):
    """
    Evaluate the model on the test dataset and calculate the loss.
    
    Parameters:
    - model: The trained model to evaluate.
    - criterion: The loss function used for evaluation.
    - test_data: The dataset to evaluate the model on.
    - block_size: The size of the input blocks.
    - batch_size: The number of samples per batch.
    - device: The device to run the evaluation on (CPU or GPU).
    
    Returns:
    - avg_loss: The average loss over the test dataset.
    """
    model.eval()  # Set the model to evaluation mode

    upper = len(tokens) - block_size - 1

    total_loss = 0
    with torch.no_grad():  # Disable gradient calculation
        for _ in range(n_samples):
            indices = torch.randint(0, upper, (batch_size,)).to(device)  # Generate all random indices at once
            input_ = tokens[indices[:, None] + torch.arange(block_size, device=device)]  # Use broadcasting to create input
            target = tokens[indices[:, None] + torch.arange(1, block_size + 1, device=device)]  # Use broadcasting to create target

            output = model(input_).transpose(1, 2)  # Transpose for CrossEntropyLoss

            loss = criterion(output, target)  # Compute loss
            loss = loss.item()
            total_loss += loss

    return total_loss / n_samples

def train(model, optimizer, criterion, tokens, block_size, batch_size, device, epochs=20, sub_epochs=1000):
    """
    Train the model using the given optimizer and criterion.
    """

    upper = len(tokens) - block_size - 1

    losses = {"train": [], "test": []}

    print(f'Training for {epochs} epochs with {sub_epochs} sub-epochs each')
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        total_loss = 0

        for _ in range(sub_epochs):  # Manually generating batches
            indices = torch.randint(0, upper, (batch_size,)).to(device)  # Generate all random indices at once
            input_ = tokens[indices[:, None] + torch.arange(block_size, device=device)]  # Use broadcasting to create input
            target = tokens[indices[:, None] + torch.arange(1, block_size + 1, device=device)]  # Use broadcasting to create target

            optimizer.zero_grad()  # Zero the parameter gradients

            output = model(input_).transpose(1, 2)  # Transpose for CrossEntropyLoss

            loss = criterion(output, target)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize the model parameters

            total_loss += loss.item()

        avg_loss = total_loss / sub_epochs  # Corrected averaging
        losses["train"].append(avg_loss)

        test_loss = evaluate(model, criterion, tokens, block_size, batch_size, device)
        losses["test"].append(test_loss)

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f}')

    return losses