from minitales.tokenizer.tokenizer import SimpleTokenizer

def train_tokenizer(vocab_path, ds, num_tokens=10000, k=100, special_tokens=["<|eos|>"], **kwargs):
    tokenizer = SimpleTokenizer(special_tokens=special_tokens)
    tokenizer.train(ds, num_tokens=num_tokens, k=k, add_remaining_tokens=True, detailed_print=True, **kwargs)
    tokenizer.save(vocab_path)