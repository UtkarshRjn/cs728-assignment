import torch
from torch.utils.data import Dataset, DataLoader

class Tokenizer:
    def __init__(self, vocab, mask_token_id=None):
        self.vocab = vocab
        self.token2idx = {token: idx for idx, token in enumerate(vocab)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        self.mask_token_id = mask_token_id

    def tokenize(self, input):
        tokens = input.split()
        token_ids = [self.token2idx[token] for token in tokens if token in self.token2idx]
        return torch.tensor(token_ids)
