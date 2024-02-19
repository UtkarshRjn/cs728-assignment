from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data.dataloader import default_collate

class MaskedDataset(Dataset):
    def __init__(self, triples, tokenizer):
        self.triples = triples
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        subject, relation, obj = self.triples[idx]
        masked_input = f'[CLS] {subject} [SEP1] {relation} [SEP2] [MASK] [END]'
        input_ids = self.tokenizer.tokenize(masked_input)
        target_id = self.tokenizer.tokenize(obj)[0]  # Assuming obj is a single token
        return torch.tensor(input_ids), target_id
    

class TripletDataset(Dataset):
    def __init__(self, triples, tokenizer):
        self.triples = triples
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        subject, relation, object = self.triples[idx]
        target = 1.0
        input = f"[CLS] {subject} [SEP1] {relation} [SEP2] {object} [END]"
        tokenized_triple = self.tokenizer.tokenize(input)
        return tokenized_triple, target