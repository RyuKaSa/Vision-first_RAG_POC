#!/usr/bin/env python3
import os
import json
import glob
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from PyPDF2 import PdfMerger
from pdf2image import convert_from_bytes
from io import BytesIO

# — — —  Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOC_DIR    = "documents"
OUTPUT_DIR = "output"
EMBED_FILE = os.path.join(OUTPUT_DIR, "image_patch_embeddings.pt")
META_FILE  = os.path.join(OUTPUT_DIR, "metadata.json")
TOP_DIR    = os.path.join(OUTPUT_DIR, "top_patches")

os.makedirs(TOP_DIR, exist_ok=True)
# clean top_patches folder
for file in os.listdir(TOP_DIR):
    os.remove(os.path.join(TOP_DIR, file))

# — — —  Load CLIP model & processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# — — —  Load embeddings and metadata
all_embeds = torch.load(EMBED_FILE)  # list of torch.Tensor [512]
with open(META_FILE, 'r') as f:
    metadata = json.load(f)           # list of dicts

# — — —  Re-render PDF pages to images
pdf_paths = sorted(glob.glob(os.path.join(DOC_DIR, "*.pdf")))
if not pdf_paths:
    raise FileNotFoundError(f"No PDFs in {DOC_DIR}")
merger = PdfMerger()
for p in pdf_paths:
    merger.append(p)
buf = BytesIO()
merger.write(buf)
merger.close()
buf.seek(0)
page_images = convert_from_bytes(buf.read(), dpi=150)

# — — —  Encode text query
def embed_text(query: str) -> torch.Tensor:
    inputs = processor(text=query, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        txt_feats = model.get_text_features(**inputs)  # [1, 512]
    vec = txt_feats.squeeze(0)
    return F.normalize(vec, p=2, dim=-1)

# — — —  Main
def main():
    query = input("Enter your query: ")
    q_vec = embed_text(query)

    emb_tensor = torch.stack(all_embeds).to(device)  # [N, 512]
    emb_tensor = F.normalize(emb_tensor, p=2, dim=-1)

    sims = emb_tensor @ q_vec                         # [N]
    probs = F.softmax(sims, dim=0)
    topk = torch.topk(probs, k=10)

    print("=== TOP 10 PATCHES ===")
    for rank, (idx, score) in enumerate(zip(topk.indices, topk.values), start=1):
        i = idx.item()
        sc = score.item()
        meta = metadata[i]
        page = meta['page']
        l, u, r, d = meta['coords']
        print(f"[{rank}] Page {page}, Patch {meta['patch_index']}, Score: {sc:.4f}")

        # Crop from in-memory page image
        page_img = page_images[page]
        patch_img = page_img.crop((l, u, r, d))
        out_path = os.path.join(TOP_DIR, f"top{rank}_page{page}_patch{meta['patch_index']}.png")
        patch_img.save(out_path)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()