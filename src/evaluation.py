import torch

def calculate_perplexity(model, tokenizer, stride=512):
    print("[PPL Evaluation] Calculando Perplexidade (PPL)...")
    reference_text = (
        "Machine learning operations (MLOps) is a set of practices that aims to deploy and maintain "
        "machine learning models in production reliably and efficiently. The word is a compound of "
        "'machine learning' and the continuous development practice of DevOps in the software field."
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encodings = tokenizer(reference_text, return_tensors="pt")
    max_length = model.config.max_position_embeddings if hasattr(model.config, "max_position_embeddings") else 2048
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood * trg_len)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc).item()
    return ppl
