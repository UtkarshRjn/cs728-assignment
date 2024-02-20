import os
import argparse
from transformers.utils import *
from torch.optim import AdamW
import torch.nn as nn
import torch

from dataset import *
from train import *
from evaluation import *
from utils import *
from models import *
from tokenizer import *

if __name__ =="__main__":
    
    cur_path = os.path.dirname(os.path.realpath( os.path.basename(__file__)))
    parser = argparse.ArgumentParser(
        description='Train a model'
    )

    parser.add_argument('--dataset', type=str, default='fb15k', help='Dataset to use')
    parser.add_argument('--mask_train', action='store_true', help='Masked training or not')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train for')
    parser.add_argument('--down_fac', type=int, default=1, help='Reduce the size of dataset by this factor')
    parser.add_argument('--save_path', type=str, default='model_checkpoint.pt', help='Path to save the trained model checkpoint')
    parser.add_argument('--load_path', type=str, help='Path to load a pre-trained model checkpoint')
    parser.add_argument('--no_train', action='store_true', help='Do not train the model, only evaluate')
    args = parser.parse_args()

    # Load data, ensure that data is at path: 'path'/'name'/[train|valid|test].txt
    train_triples, val_triples, test_triples = build_data(name = args.dataset,path = cur_path + '/datasets/')
    train_triples = train_triples[:int(len(train_triples)/args.down_fac)]
    val_triples = val_triples[:int(len(val_triples)/args.down_fac)]
    test_triples = test_triples[:int(len(test_triples)/args.down_fac)]
    
    entity_list = list(set([x[0] for x in train_triples] + [x[2] for x in train_triples] + [x[0] for x in val_triples] + [x[2] for x in val_triples] + [x[0] for x in test_triples] + [x[2] for x in test_triples]))
    relation_list = list(set([x[1] for x in train_triples] + [x[1] for x in val_triples] + [x[1] for x in test_triples]))

    vocab = list(set(entity_list).union(relation_list).union(['[CLS]', '[SEP1]', '[SEP2]', '[END]']))
    mask_token_id = len(vocab)  # Add a new token for the mask
    vocab.append('[MASK]')

    ntoken = len(vocab)  # Number of tokens in your vocabulary
    print(f"Vocab size: {ntoken}")
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
    
    # Load a pre-trained model if specified
    if args.load_path:
        model.load_state_dict(torch.load(args.load_path))
        model.to(device)
        print(f"Model loaded from {args.load_path}")

    if not args.no_train:
        epochs = args.num_epochs

        for epoch in range(epochs):
            
            if not args.mask_train:
                loss = train(model, train_loader, optimizer, criterion, device)
            else:
                loss = train2(model, train_loader, optimizer, criterion, device)
            
            print(f"Epoch {epoch+1}, Loss: {loss}")

        torch.save(model.state_dict(), args.save_path)
        print(f"Model saved to {args.save_path}")

    # Evaluate on test set
    if not args.mask_train:
        metrics = evaluate_model(model, tokenizer, test_triples, entity_list, device)
    else:
        metrics = evaluate_model2(model, tokenizer, test_triples, entity_list, device)

    print(metrics)