#!/usr/bin/env python3
import os
import torch
import torch.nn.functional as F
from transformers import AutoConfig
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

# — — —  Setup device
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps"  if torch.backends.mps.is_available()
    else "cpu"
)

# — — —  Model IDs
FINE_TUNED = "vidore/colqwen2.5-v0.2"
BASE_VL    = "Qwen/Qwen2.5-VL-3B-Instruct"

# — — —  Load config + model
config = AutoConfig.from_pretrained(BASE_VL)
model  = ColQwen2_5.from_pretrained(
    FINE_TUNED,
    config=config,
    torch_dtype=torch.float16,
).eval().to(device)

# — — —  Load processor (handles both images & text)
processor = ColQwen2_5_Processor.from_pretrained(FINE_TUNED)

# — — —  Load your precomputed image embeddings
image_embeddings = torch.load(os.path.join("output", "image_embeddings.pt"))

def compute_similarity_scores(q_vec, image_embeddings):
    q_vec = F.normalize(q_vec, p=2, dim=0)  # 128-d unit vector
    all_scores = []
    for page_embed in image_embeddings:
        # page_embed: [1, num_patches, 128]
        p = page_embed.squeeze(0).to(device)   # → [num_patches, 128]
        p = F.normalize(p, p=2, dim=1)         # unit-norm each patch
        sim = p @ q_vec                        # [num_patches]
        all_scores.append(F.softmax(sim, dim=0))
    return all_scores

def main():
    ### 1) Tokenize your text query ###
    query = "Who is David Cochard?"
    text_inputs = processor.tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        padding=True,
    ).to(device)

    ### 2) Run through the model’s *decoder* (the LM head) ###
    #    then pull out last_hidden_state (2048-dim) and project to 128-d.
    with torch.no_grad():
        # get the underlying causal-LM decoder
        decoder = model.get_decoder()
        outputs = decoder(**text_inputs)
        hidden_states = outputs.last_hidden_state       # [1, seq_len, 2048]

        # project into the shared 128-d space
        # (in your LoRA model this layer is named `custom_text_proj`)
        proj_states = model.custom_text_proj(hidden_states)  # [1, seq_len, 128]

    ### 3) Mean-pool + normalize to a single 128-d query vector ###
    q_vec = proj_states.mean(dim=1).squeeze(0)   # [128]
    q_vec = F.normalize(q_vec, p=2, dim=0)

    ### 4) Compute per-patch similarity and print top-5 ###
    similarity_scores = compute_similarity_scores(q_vec, image_embeddings)
    for page_idx, scores in enumerate(similarity_scores):
        top5 = scores.topk(5)
        print(f"\nPage {page_idx}")
        print("  Top-5 scores:     ", top5.values.cpu().tolist())
        print("  Top-5 patch idxs:", top5.indices.cpu().tolist())

if __name__ == "__main__":
    main()
