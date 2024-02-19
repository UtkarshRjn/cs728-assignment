import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from dataset import TripleDataset

def evaluate_model(model, tokenizer, test_data, entity_list, device, batch_size=32):
    """
    Fast evaluation of the model using batch processing.
    """
    model.to(device)
    model.eval()
    
    hits_at_1_count = 0
    hits_at_10_count = 0
    reciprocal_ranks = []
    average_precisions = []

    for triplet in tqdm(test_data, desc="Evaluating"):
        head, relation, tail = triplet
        
        # Prepare batches for head replacement
        replaced_heads_triplets = [(entity,relation,tail) for entity in entity_list]
        head_scores = batch_predict_triplet_scores(model, tokenizer, replaced_heads_triplets, device, batch_size)

        # Prepare batches for tail replacement
        replaced_tails_triplets = [(head,relation,entity) for entity in entity_list]
        tail_scores = batch_predict_triplet_scores(model, tokenizer, replaced_tails_triplets, device, batch_size)

        for scores, true_entity in [(head_scores, head), (tail_scores, tail)]:
            scores = scores.cpu().numpy()  # Assuming scores are logits or probabilities
            true_idx = entity_list.index(true_entity)
            
            # Calculate rankings
            sorted_indices = np.argsort(-scores)  # Assuming higher score indicates better match
            rank = np.where(sorted_indices == true_idx)[0] + 1
            
            if rank == 1:
                hits_at_1_count += 1
            if rank <= 10:
                hits_at_10_count += 1
            
            reciprocal_ranks.append(1.0 / rank)

            # Calculate MAP
            precisions = [1.0 / rank if sorted_indices[i] == true_idx else 0 for i in range(len(entity_list))]
            if np.sum(precisions) > 0:
                average_precisions.append(np.mean(precisions))

    # Calculate final metrics
    num_evaluations = 2 * len(test_data)  # Each test triplet is evaluated twice (head and tail replacement)
    hits_at_1 = hits_at_1_count / num_evaluations
    hits_at_10 = hits_at_10_count / num_evaluations
    mrr = np.mean(reciprocal_ranks)
    map_score = np.mean(average_precisions) if average_precisions else 0

    return {
        "HITS@1": hits_at_1,
        "HITS@10": hits_at_10,
        "MRR": mrr,
        "MAP": map_score,
    }

# Note: Ensure that batch_predict_triplet_scores returns the logits or probabilities for each entity prediction.
# Adjust the score extraction and ranking calculation as necessary based on your model's output format.

def batch_predict_triplet_scores(model, tokenizer, triplets, device, batch_size=32):
    """
    Predict scores for a batch of triplets.
    """
    
    # Create TensorDataset and DataLoader for efficient batch processing
    dataset = TripleDataset(triplets, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Store predictions
    all_logits = []

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            all_logits.append(logits)

    # Concatenate all logits from each batch
    all_logits = torch.cat(all_logits, dim=0)
    
    return all_logits
# Example usage (assuming you have `model`, `tokenizer`, `test_data`, and `entity_list` defined)
# metrics = evaluate_model(model, tokenizer, test_data, entity_list)
# print(metrics)

def evaluate_model2( model, tokenizer, test_data, entity_list, device):
    hits_at_1 = 0
    hits_at_10 = 0
    reciprocal_ranks = []
    average_precisions = []  # List to hold the average precision for each query

    # Iterate through each test triplet
    for s, r, o in test_data:
        # Prepare the masked queries
        masked_head_query = f"[CLS] [MASK] [SEP] {r} [SEP] {o} [END]"
        masked_tail_query = f"[CLS] {s} [SEP] {r} [SEP] [MASK] [END]"

        # Predict the ranking for the masked head entity
        head_predictions = model_predict(masked_head_query, entity_list, model, tokenizer, device)
        head_rank = 1 + [entity_id for entity_id, _ in head_predictions].index(s)  # Get rank of the correct entity

        # Predict the ranking for the masked tail entity
        tail_predictions = model_predict(masked_tail_query, entity_list, model, tokenizer, device)
        tail_rank = 1 + [entity_id for entity_id, _ in tail_predictions].index(o)

        # Update metrics
        hits_at_1 += (head_rank == 1) + (tail_rank == 1)
        hits_at_10 += (head_rank <= 10) + (tail_rank <= 10)
        reciprocal_ranks.append(1 / head_rank)
        reciprocal_ranks.append(1 / tail_rank)

        average_precisions.append(1 / head_rank)
        average_precisions.append(1 / tail_rank)

    # Calculate final metrics
    total_examples = 2 * len(test_data)  # Each test triplet results in two queries
    hits_at_1 /= total_examples
    hits_at_10 /= total_examples
    mrr = sum(reciprocal_ranks) / total_examples
    map_metric = sum(average_precisions) / total_examples  # Calculate MAP

    return {
        "HITS@1": hits_at_1,
        "HITS@10": hits_at_10,
        "MRR": mrr,
        "MAP": map_metric,
    }


def model_predict(masked_query, entity_list, model, tokenizer, device):
    """
    Predicts the ranking of entities for a masked query.

    Args:
    - masked_query (str): The input text with one entity masked.
    - entity_list (list): List of all possible entity IDs (strings).
    - model (transformers.PreTrainedModel): The trained model.
    - tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
    - device (torch.device): The device to run the prediction on.

    Returns:
    - List of tuples (entity_id, score) sorted by descending score.
    """

    # Tokenize the input
    inputs = tokenizer(masked_query, return_tensors='pt', padding=True, truncation=True, max_length=128)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Get model predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # Shape: (batch_size, seq_length, vocab_size)

    # Identify the masked position index
    mask_token_index = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    # Extract logits for the masked position; shape: (vocab_size,)
    mask_logits = logits[0, mask_token_index.item(), :]

    # Score each entity in the entity list
    entity_scores = []
    for entity_id in entity_list:
        entity_token_id = tokenizer.convert_tokens_to_ids(entity_id)
        entity_score = mask_logits[entity_token_id].item()  # Logit for the entity
        entity_scores.append((entity_id, entity_score))

    # Sort by descending score
    ranked_entities = sorted(entity_scores, key=lambda x: x[1], reverse=True)

    return ranked_entities