import torch

def train(model, optimizer, criterion, tokens, block_size, batch_size, device, epochs=20, sub_epochs=1000):
    """
    Train the model using the given optimizer and criterion.
    """

    upper = len(tokens) - block_size - 1
    
    print(f'Training for {epochs} epochs with {sub_epochs} sub-epochs each')
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        total_loss = 0

        for _ in range(sub_epochs):  # Manually generating batches
            indices = torch.randint(0, upper, (batch_size,))  # Generate all random indices at once
            input_ = tokens[indices[:, None] + torch.arange(block_size, device=device)]  # Use broadcasting to create input
            target = tokens[indices[:, None] + torch.arange(1, block_size + 1, device=device)]  # Use broadcasting to create target

            optimizer.zero_grad()  # Zero the parameter gradients

            output = model(input_).transpose(1, 2)  # Transpose for CrossEntropyLoss

            loss = criterion(output, target)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize the model parameters

            total_loss += loss.item()

        avg_loss = total_loss / sub_epochs  # Corrected averaging
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

def main():
    from dataset import load_dataset
    from tokenizer.tokenizer import SimpleTokenizer
    from model import Transformer
    import os

    ds_path = 'dataset/TinyStories10k_eos.txt'
    vocab_path = 'tokenizer/vocab10k.json'

    if not os.path.exists(vocab_path):
        from tokenizer.train import main as tokenizer_train
        tokenizer_train(vocab_path)

    from torch import optim, nn

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    print('Loading dataset...')
    ds = load_dataset(ds_path)
    print('Loading tokenizer...')
    tokenizer = SimpleTokenizer(vocab_path, special_tokens=['<|eos|>'])
    print('Encoding...')
    tokens = tokenizer.encode(ds)

    vocab_size = tokenizer.num_tokens
    n_embedding = 36
    block_size = 96
    n_heads = 4  # Number of attention heads
    batch_size = 32

    # Corrected class name (assuming typo)
    print('Loading model...')
    model = Transformer(vocab_size, n_embedding, block_size, n_heads).to(device)
    print('Loading tokens as tensor...')
    tensor_tokens = torch.tensor(tokens, dtype=torch.long, device=device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print('Training...')
    train(model, optimizer, criterion, tensor_tokens, block_size, batch_size, device)

if __name__ == '__main__':
    main()