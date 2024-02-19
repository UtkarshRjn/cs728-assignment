from tqdm import tqdm

def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits.squeeze(-1), labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    return total_loss / len(data_loader)

def train2(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc=f"Training"):
        optimizer.zero_grad()
        batch_on_device = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch_on_device)
        loss = outputs.loss
        
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    return total_loss / len(data_loader)