import os
import argparse
from transformers.utils import *
from torch.optim import AdamW
import torch.nn as nn
import torch

from dataset import *
from train import *
from evaluate import *
from utils import *
from model import *
from tokenizer import *

if __name__ =="__main__":
    
    cur_path = os.path.dirname(os.path.realpath( os.path.basename(__file__)))
    parser = argparse.ArgumentParser(
        description='Train a model'
    )

    parser.add_argument('--dataset', type=str, default='fb15k', help='Dataset to use')
    parser.add_argument('--mask_train', type=bool, default=False, help='Masked training or not')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train for')
    args = parser.parse_args()

    # Load data, ensure that data is at path: 'path'/'name'/[train|valid|test].txt
    train_triples, val_triples, test_triples = build_data(name = args.dataset,path = cur_path + '/datasets/')
    entity_list = list(set([x[0] for x in train_triples] + [x[2] for x in train_triples] + [x[0] for x in val_triples] + [x[2] for x in val_triples] + [x[0] for x in test_triples] + [x[2] for x in test_triples]))
    relation_list = list(set([x[1] for x in train_triples] + [x[1] for x in val_triples] + [x[1] for x in test_triples]))

    vocab = list(set(entity_list).union(relation_list).union(['[CLS]', '[SEP1]', '[SEP2]', '[END]']))
    mask_token_id = len(vocab)  # Add a new token for the mask
    vocab.append('[MASK]')

    ntoken = len(vocab)  # Number of tokens in your vocabulary
    d_model = 256  # Embedding dimension
    nhead = 4  # Number of attention heads
    d_hid = 512  # Hidden dimension
    nlayers = 4  # Number of transformer layers
    dropout = 0.1  # Dropout probability

    tokenizer = Tokenizer(vocab, mask_token_id=mask_token_id)

    if not args.mask_train:
        model = TransformerModel(ntoken=ntoken, d_model=d_model, nhead=nhead, d_hid=d_hid,
                             nlayers=nlayers, dropout=dropout)
        
        train_dataset = TripletDataset(train_triples, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    else:
        train_dataset = MaskedDataset(train_triples, tokenizer)
        model = TransformerModel(ntoken=ntoken, d_model=d_model, nhead=nhead, d_hid=d_hid,
                             nlayers=nlayers, dropout=dropout, is_classifier=False)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    epochs = args.num_epochs

    for epoch in range(epochs):
        
        if not args.mask_train:
            loss = train(model, train_loader, optimizer, criterion, device)
        else:
            loss = train2(model, train_loader, optimizer, criterion, device)
        
        print(f"Epoch {epoch+1}, Loss: {loss}")

    # Evaluate on test set
    if not args.mask_train:
        metrics = evaluate_model(model, tokenizer, test_triples, entity_list, device)
    else:
        metrics = evaluate_model2(model, tokenizer, test_triples, entity_list, device)

    print(metrics)