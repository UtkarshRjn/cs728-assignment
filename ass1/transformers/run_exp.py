import os
import argparse
from transformers.utils import *
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import BertForSequenceClassification, AdamW, BertForMaskedLM
from torch.optim import AdamW
from transformers import BertConfig, BertModel
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import torch

from dataset import *
from train import *
from evaluate import *
from utils import *

class CustomBertForSequenceClassification(nn.Module):
    def __init__(self, config):
        super(CustomBertForSequenceClassification, self).__init__()
        self.num_labels = config.num_labels
        
        # Initialize the BERT model with the given config
        self.bert = BertModel(config)
        
        # Custom classifier for sequence classification on top of BERT
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        
        # You might want to add more layers here depending on your use case

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        # Get the outputs from the BERT model
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        
        pooled_output = outputs[1]

        # Apply the classification head
        logits = self.classifier(pooled_output)

        return logits

if __name__ =="__main__":
    
    cur_path = os.path.dirname(os.path.realpath( os.path.basename(__file__)))
    parser = argparse.ArgumentParser(
        description='Train a model'
    )

    parser.add_argument('--dataset', type=str, default='fb15k', help='Dataset to use')
    parser.add_argument('--mask_train', type=bool, default=False, help='Masked training or not')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs to train for')
    args = parser.parse_args()

    # Load data, ensure that data is at path: 'path'/'name'/[train|valid|test].txt
    train_triples, val_triples, test_triples = build_data(name = args.dataset,path = cur_path + '/datasets/')
    entity_list = list(set([x[0] for x in train_triples] + [x[2] for x in train_triples] + [x[0] for x in val_triples] + [x[2] for x in val_triples] + [x[0] for x in test_triples] + [x[2] for x in test_triples]))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    if not args.mask_train:
        train_dataset = TripleDataset(train_triples, tokenizer, max_length=128)
        # val_dataset = TripleDataset(val_triples, tokenizer, max_length=128)
        # test_dataset = TripleDataset(test_triples, tokenizer, max_length=128)
    
        config = BertConfig()
        config.num_labels = 1
        model = CustomBertForSequenceClassification(config)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    else:
        train_dataset = MaskedDataset(train_triples, tokenizer, max_length=128)
        # val_dataset = MaskedDataset(val_triples, tokenizer, max_length=128)
        # test_dataset = TripleDataset(test_triples, tokenizer, max_length=128)

        model = BertForMaskedLM(BertConfig())
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    criterion = BCEWithLogitsLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    epochs = args.num_epochs

    for epoch in range(epochs):
        
        if not args.mask_train:
            loss = train(model, train_loader, optimizer, criterion, device)
        else:
            loss = train2(model, train_loader, optimizer, device)
        
        print(f"Epoch {epoch+1}, Loss: {loss}")

    # Evaluate on test set
    if not args.mask_train:
        metrics = evaluate_model(model, tokenizer, test_triples, entity_list, device)
    else:
        metrics = evaluate_model2(model, tokenizer, test_triples, entity_list, device)

    print(metrics)