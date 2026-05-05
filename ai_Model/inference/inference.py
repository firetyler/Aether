import math
import torch


def temperature_sampling(logits, temperature=1.0):
    """Applicera temperature scaling på logits"""
    if temperature != 1.0:
        logits = logits / temperature
    return torch.softmax(logits, dim=-1)


def top_k_sampling(probs, k=10):
    """Top-K sampling: ta bara de k mest sannolika tokens"""
    topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
    return topk_probs, topk_indices


def nucleus_sampling(probs, p=0.9):
    """Nucleus (top-p) sampling: ta tokens tills cumulative prob > p"""
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Hitta cutoff
    sorted_indices_to_remove = cumsum_probs > p
    sorted_indices_to_remove[..., 0] = False  # Behåll minst en token
    
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    probs[indices_to_remove] = 0.0
    
    return probs / probs.sum(dim=-1, keepdim=True)


def generate_text(model, tokenizer, prompt, max_length=50, method="top_k", temperature=0.7, k=10, p=0.9):
    """Generer text med olika sampling-metoder.
    
    Args:
        model: Transformer modell
        tokenizer: BPE tokenizer
        prompt: Starttext
        max_length: Max antal tokens att generera
        method: "top_k", "nucleus", "beam" eller "greedy"
        temperature: Temperatur för sampling (högre = mer kreativt)
        k: Antal top tokens för top-k
        p: Cumulative probability för nucleus
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Tokenisera prompt
    input_ids = tokenizer.encode(prompt)
    if isinstance(input_ids, list):
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    else:
        input_ids = input_ids.to(device)

    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(generated)
            next_token_logits = outputs[:, -1, :]
            
            if method == "top_k":
                probs = temperature_sampling(next_token_logits, temperature)
                topk_probs, topk_indices = top_k_sampling(probs, k=k)
                next_token_id = topk_indices[0][torch.multinomial(topk_probs[0], 1)].unsqueeze(0)
            
            elif method == "nucleus":
                probs = temperature_sampling(next_token_logits, temperature)
                probs = nucleus_sampling(probs.clone(), p=p)
                next_token_id = torch.multinomial(probs[0], 1).unsqueeze(0)
            
            elif method == "greedy":
                next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
            
            else:  # beam_search (simplified)
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token_id = probs.topk(1, dim=-1)[1]
            
            generated = torch.cat([generated, next_token_id], dim=1)

    return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)


def beam_search_generate(model, tokenizer, prompt, max_length=50, beam_width=3, eos_token_id=None):
    """Beam search text generation"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Tokenisera prompt
    input_ids = tokenizer.encode(prompt)
    if isinstance(input_ids, list):
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    else:
        input_ids = input_ids.to(device)
    
    sequences = [(input_ids, 0.0)]  # (sequence_tensor, score)

    for _ in range(max_length):
        all_candidates = []
        for seq, score in sequences:
            with torch.no_grad():
                outputs = model(seq)
                logits = outputs[:, -1, :]
                probs = torch.softmax(logits, dim=-1)

                topk_probs, topk_indices = torch.topk(probs, beam_width, dim=-1)

                for i in range(beam_width):
                    token_id = topk_indices[0][i].item()
                    token_prob = topk_probs[0][i].item()

                    new_token = torch.tensor([[token_id]], device=device)
                    new_seq = torch.cat([seq, new_token], dim=1)
                    new_score = score - math.log(token_prob + 1e-8)

                    all_candidates.append((new_seq, new_score))

        sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_width]

        if eos_token_id is not None:
            if all(seq[0][0, -1].item() == eos_token_id for seq in sequences):
                break

    best_seq = sequences[0][0]
    return tokenizer.decode(best_seq[0].tolist(), skip_special_tokens=True)

