from tqdm import tqdm

def train(model, data_loader, optimizer, criterion, device):
    
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        input_ids, target = batch
        input_ids = input_ids.transpose(0, 1)
        input_ids = input_ids.to(device)
        target = target.to(device)
        output = model(input_ids)
        # Calculate loss
        loss = criterion(output, target)
        total_loss += loss.item()
        # Backpropagation
        loss.backward()
        optimizer.step()
    
    return total_loss / len(data_loader)

def train2(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for input_ids, target_id in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        input_ids = input_ids.to(device)
        target_id = target_id.to(device)
        output = model(input_ids)
        output = output[:, -1, :]  # Take the prediction for the last token in the sequence
        loss = criterion(output, target_id)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(data_loader)