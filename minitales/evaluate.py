import torch

def evaluate(model, criterion, test_data, block_size, batch_size, device):
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
    total_loss = 0
    upper = len(test_data) - block_size - 1

    with torch.no_grad():  # Disable gradient calculation
        for i in range(0, upper, batch_size):
            # Create input and target batches
            indices = torch.arange(i, min(i + batch_size, upper)).to(device)
            input_ = test_data[indices[:, None] + torch.arange(block_size, device=device)]
            target = test_data[indices[:, None] + torch.arange(1, block_size + 1, device=device)]

            output = model(input_).transpose(1, 2)  # Transpose for CrossEntropyLoss
            loss = criterion(output, target)  # Compute loss
            total_loss += loss.item()

    avg_loss = total_loss / (upper // batch_size)  # Calculate average loss
    return avg_loss