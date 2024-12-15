"""
This file contains the SimpleTokenizer class, which implements the tokenizer with helper functions.
"""

import regex
import json

class SimpleTokenizer:
    """
    SimpleTokenizer class is responsible for tokenizing text data using regular expressions.
    It also allows saving and loading vocabularies.
    """

    def __init__(self, vocab_dir=None, special_tokens=None, unknown_token=None):
        """
        Initializes the SimpleTokenizer with default values.
        If a vocabulary directory is provided, it loads the vocabulary from that directory.
        """
        self.special_tokens_pattern = r"<\|[\w]+\|>"
        self.pattern = r"'s|'t|'ll|'ve|'r|\s*[^\d\W]+|[\d]+|[^\w]"  # Regex pattern for tokenizing text

        self.special_tokens = set(special_tokens or [])  # Use an empty set if no special tokens are provided
        self.unknown_token = unknown_token or "<|unknown|>"  # Used to handle unknown characters
        self.special_tokens.add(self.unknown_token)

        if vocab_dir:
            self.load(vocab_dir)  # Load vocabulary if directory is provided
        else:
            self.vocab = {i: chr(i) for i in range(256)}  # Initialize vocabulary with ASCII characters
            self.word2token = {v: k for k, v in self.vocab.items()}  # Map words to token IDs
            self.num_tokens = len(self.vocab)

    def set_special_tokens(self, tokens, unknown_token=None):
        """
        Sets special tokens that should be included in the vocabulary.
        """
        self.special_tokens.update(tokens)
        if unknown_token:
            self.unknown_token = unknown_token
        self.special_tokens.add(self.unknown_token)

    def train(self, data, num_tokens, k = 5, without_default_tokens = False,
              dont_print = False, dont_show_error = False, add_remaining_tokens = False,
              detailed_print = False):
        """
        Trains the tokenizer by analyzing the given data and creating
        a vocabulary with the specified number of tokens.
        """
        adjusted_data = self._apply_pattern(data)
        raws = [get_raw(item) for item in adjusted_data if item not in self.special_tokens and len(item) > 1]

        if without_default_tokens:
            target_tokens = num_tokens
        else:
            target_tokens = num_tokens - 256 - len(self.special_tokens)

        epochs = max(1, target_tokens // k)

        if epochs * k < target_tokens and add_remaining_tokens:
            epochs += 1

        not_complete = True
        for epoch in range(epochs):
            if epoch == epochs - 1 and add_remaining_tokens:
                k = target_tokens - epochs * k + k

            if not dont_print:
                if detailed_print:
                    raws_len = len(raws)
                    suffix = f", looping throw {raws_len} raw item"
                else:
                    suffix = ""
                print(f"Processing: {epoch + 1}/{epochs}{suffix}")
            pairs = {}

            for raw in raws:
                for pair in zip(raw, raw[1:]):
                    pairs[pair] = pairs.get(pair, 0) + 1

            if not pairs:
                not_complete = False
                break

            top_pairs = get_topk_pair(pairs, k)
            for pair in top_pairs:
                token_id = len(self.vocab)
                self.vocab[token_id] = self.decode(pair)

                new_raws = []
                for raw in raws:
                    merged = []
                    continue_merge = False
                    for item in zip(raw, raw[1:]):
                        if continue_merge:
                            continue_merge = False
                            continue
                        if item == pair:
                            merged.append(token_id)
                            continue_merge = True
                        else:
                            merged.append(item[0])
                    if not continue_merge:
                        merged.append(raw[-1])
                    if len(merged) > 1:
                        new_raws.append(merged)
                raws = new_raws
                
        for i, token in enumerate(self.special_tokens, start=1):
            self.vocab[len(self.vocab)] = token

        self.word2token = {v: k for k, v in self.vocab.items()}
        self.num_tokens = len(self.vocab)

        if not not_complete:
            suffix = " (not complete) can't train more tokens"
        else:
            suffix = ""

        if self.num_tokens < num_tokens and k != 1:
            default_tokens = 256 + len(self.special_tokens)
            suffix += f" num_tokens is less than target_tokens, try to train with k = {default_tokens // (num_tokens - self.num_tokens)}"

        if dont_show_error:
            suffix = ""

        if not dont_print:
            print(f"Training complete!, vocab size: {self.num_tokens} {suffix}")

    def _apply_pattern(self, data):
        """
        Applies the given regex pattern to the input data and returns the matches.
        """
        combined_pattern = f"{self.special_tokens_pattern}|{self.pattern}"
        return regex.findall(combined_pattern, data)

    def decode_no_join(self, token_ids):
        """
        Decodes a sequence of token IDs into their character equivalents.
        """
        return [self.vocab.get(token, self.unknown_token) for token in token_ids]

    def decode(self, token_ids):
        """
        Decodes a sequence of token IDs back into the original text.
        """
        return "".join(self.decode_no_join(token_ids))

    def encode(self, text):
        """
        Encodes a string into token IDs using the current vocabulary.
        """
        temp = ""
        raw = []
        unknown = self.word2token[self.unknown_token]

        for char in text:
            temp += char
            if temp in self.word2token:
                continue

            if not any(token.startswith(temp) for token in self.special_tokens):
                temp = temp[:-1]
                raw.append(self.word2token.get(temp, unknown))
                temp = char

        if temp:
            raw.append(self.word2token.get(temp, unknown))

        return raw

    def save(self, filepath):
        """
        Saves the vocabulary to a specified file in JSON format.
        """
        with open(filepath, "w") as f:
            json.dump(self.vocab, f)

    def load(self, filepath):
        """
        Loads a vocabulary from a specified JSON file.
        """
        with open(filepath, "r") as f:
            self.vocab = {int(k): v for k, v in json.load(f).items()}

        self.word2token = {v: k for k, v in self.vocab.items()}
        self.num_tokens = len(self.vocab)


def get_raw(text, encoding="utf-8"):
    """
    Converts a string into its raw byte representation.
    """
    return list(text.encode(encoding))

def get_topk_pair(pairs, k):
    """
    Returns the top-k most frequent token pairs.
    """
    return [pair for pair, _ in sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:k]]