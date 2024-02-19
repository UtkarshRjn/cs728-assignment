from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data.dataloader import default_collate

class TripleDataset(Dataset):
    def __init__(self, triplets, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.triplets = triplets
        self.max_length = max_length

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        subject_id, relation_id, object_id = self.triplets[idx]
        inputs = self.tokenizer.encode_plus(
            f"[CLS] {subject_id} [SEP] {relation_id} [SEP] {object_id} [END]",
            add_special_tokens=False,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {"input_ids": inputs.input_ids.squeeze(0), "attention_mask": inputs.attention_mask.squeeze(0), "labels": torch.tensor(1, dtype=torch.float)} # dummy score
    

class MaskedDataset(Dataset):
    def __init__(self, triplets, tokenizer, max_length=128):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        s, r, o = self.triplets[idx]
        # For masked generation approach
        inputs = self.tokenizer(f"[CLS] {s} [SEP] {r} [SEP] [MASK] [END]", return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
        labels = self.tokenizer(f"[CLS] {s} [SEP] {r} [SEP] {o} [END]", return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)["input_ids"]
        inputs["labels"] = labels
        return inputs
    
def custom_collate_fn(batch):
    batch = default_collate(batch)
    batch['input_ids'] = batch['input_ids'].squeeze(1)
    batch['token_type_ids'] = batch['token_type_ids'].squeeze(1)
    batch['attention_mask'] = batch['attention_mask'].squeeze(1)
    batch['labels'] = batch['labels'].squeeze(1)
    return batch