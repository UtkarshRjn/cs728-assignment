import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from dataset import *
import torch.nn.functional as F

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
            precisions = np.array(precisions)
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

def batch_predict_triplet_scores(model, tokenizer, triplets, device, batch_size=32):
    """
    Predict scores for a batch of triplets.
    """
    
    # Create TensorDataset and DataLoader for efficient batch processing
    dataset = TripletDataset(triplets, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Store predictions
    all_logits = []

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for batch in dataloader:
            input_ids, target = batch
            input_ids = input_ids.transpose(0, 1)
            input_ids = input_ids.to(device)
            target = target.to(device)
            output = model(input_ids)
            logits = output
            all_logits.append(logits)

    # Concatenate all logits from each batch
    all_logits = torch.cat(all_logits, dim=0)
    
    return all_logits

def evaluate_model2(model, tokenizer, test_data, entity_list, device, batch_size=32):
    hits_at_1 = 0
    hits_at_10 = 0
    reciprocal_ranks = []
    average_precisions = []  

    # Iterate through test data in batches
    for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
        batch_data = test_data[i:i+batch_size]

        # Prepare masked queries for the batch
        masked_head_queries = [f"[CLS] [MASK] [SEP1] {r} [SEP2] {o} [END]" for s, r, o in batch_data]
        masked_tail_queries = [f"[CLS] {s} [SEP1] {r} [SEP2] [MASK] [END]" for s, r, o in batch_data]

        # Predict rankings for the masked head entities
        head_predictions = model_predict(masked_head_queries, entity_list, model, tokenizer, device)

        # Predict rankings for the masked tail entities
        tail_predictions = model_predict(masked_tail_queries, entity_list, model, tokenizer, device)

        # Iterate through each triplet in the batch
        for j, (s, r, o) in enumerate(batch_data):
            # Get ranks of the correct entities
            head_rank = 1 + head_predictions[j].index(s)
            tail_rank = 1 + tail_predictions[j].index(o)

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


def model_predict(masked_queries, entity_list, model, tokenizer, device):
    """
    Predicts the ranking of entities for masked queries in batch.
    """
    
    # Tokenize the input batch
    input_ids = [tokenizer.tokenize(query) for query in masked_queries]
    input_ids = torch.stack(input_ids).to(device)

    # Ensure input is of shape [seq_len, batch_size]
    input_ids = input_ids.transpose(0, 1)

    with torch.no_grad():
        logits = model(input_ids)  # Assuming logits shape is [seq_len, batch_size, ntoken]

    batch_results = []
    for i in range(logits.size(1)):  # Iterate over batch dimension
        # Softmax over the vocabulary dimension
        scores = F.softmax(logits[:, i, :], dim=-1)

        # Get the last timestep scores assuming [MASK] or [END] token is the target
        last_scores = scores[-1]

        entity_scores = []
        for entity_id in entity_list:
            entity_index = tokenizer.token2idx.get(entity_id, None)
            if entity_index is not None:
                # Safely extract score for entity_id if it exists in tokenizer
                entity_score = last_scores[entity_index].item()
                entity_scores.append((entity_id, entity_score))

        # Sort entities by their scores in descending order
        ranked_entities = sorted(entity_scores, key=lambda x: x[1], reverse=True)
        batch_results.append([entity_id for entity_id, _ in ranked_entities])

    return batch_results