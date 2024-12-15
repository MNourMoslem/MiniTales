from dataset import load_dataset
from tokenizer.tokenizer import SimpleTokenizer

def main(vocab_path):
    ds = load_dataset('dataset/TinyStories10k_eos.txt')
    tokenizer = SimpleTokenizer(special_tokens=["<|eos|>"])
    tokenizer.train(ds, num_tokens=10000, k=100, add_remaining_tokens=True, detailed_print=True)

    tokenizer.save(vocab_path)

if __name__ == '__main__':
    main('tokenizer/vocab10k.json')