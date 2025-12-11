import torch
import torch.nn.functional as F

def rolling_consistency(embeddings, window=5):
    sims = []
    for i in range(len(embeddings) - window):
        ref = embeddings[i]
        nxt = embeddings[i+1:i+window+1]
        cos = F.cosine_similarity(ref.unsqueeze(0), nxt)
        sims.append(cos.mean().item())
    return torch.tensor(sims)