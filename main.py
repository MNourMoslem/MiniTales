import torch
import torch.nn as nn
import torch.optim as optim

def main():
    from minitales.dataset import load_dataset
    from minitales.tokenizer.tokenizer import SimpleTokenizer
    from minitales.model import Transformer
    from minitales.train import train as train_model
    import os

    ds_path = 'minitales/dataset/TinyStories10k_eos.txt'
    vocab_path = 'minitales/tokenizer/vocab10k.json'

    print('Loading dataset...')
    ds = load_dataset(ds_path)

    if not os.path.exists(vocab_path):
        from minitales.tokenizer.train import train_tokenizer as tokenizer_train

        input_ = input("could not find vocab file for the tokenizer, would you like to train it? (y/n)")
        if input_ == 'y':
            print('Training tokenizer...')
            tokenizer_train(vocab_path, ds)
        else:
            print("Could not find vocab file, exiting...")
            return

    from torch import optim, nn

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

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
    train_model(model, optimizer, criterion, tensor_tokens, block_size, batch_size, device)

    return model

if __name__ == '__main__':
    main()