#!/usr/bin/env python3
import os
import json
import math
import torch
import torch.nn.functional as F
import glob
from io import BytesIO
from PyPDF2 import PdfMerger
from pdf2image import convert_from_bytes
from transformers import CLIPModel, CLIPProcessor

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOC_DIR = "documents"
OUTPUT_DIR = "output"
MAX_PATCHES_PER_PAGE = 500

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load CLIP model & processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Gather PDFs
pdf_paths = sorted(glob.glob(os.path.join(DOC_DIR, "*.pdf")))
if not pdf_paths:
    raise FileNotFoundError(f"No PDF files found in {DOC_DIR}")

# Merge PDFs into a single buffer
merger = PdfMerger()
for path in pdf_paths:
    merger.append(path)
buf = BytesIO()
merger.write(buf)
merger.close()
buf.seek(0)

# Convert merged PDF pages to images
doc_images = convert_from_bytes(buf.read(), dpi=150)
print(f"[INFO] Converted {len(doc_images)} pages from {len(pdf_paths)} PDFs")

# Containers for embeddings and metadata
all_embeddings = []
metadata = []

for page_idx, img in enumerate(doc_images):
    width, height = img.size
    # Compute adaptive square patch size
    target_patch_area = (width * height) / MAX_PATCHES_PER_PAGE
    patch_size = int(math.sqrt(target_patch_area))
    if patch_size < 1:
        patch_size = 1

    # Adjust to ensure not exceeding max patches
    n_cols = width // patch_size
    n_rows = height // patch_size
    while n_cols * n_rows > MAX_PATCHES_PER_PAGE:
        patch_size += 1
        n_cols = width // patch_size
        n_rows = height // patch_size

    print(f"[INFO] Page {page_idx}: patch_size={patch_size}, cols={n_cols}, rows={n_rows}, total_patches={n_cols*n_rows}")

    patches = []
    coords = []
    for r in range(n_rows):
        for c in range(n_cols):
            left = c * patch_size
            upper = r * patch_size
            right = left + patch_size
            lower = upper + patch_size
            patch = img.crop((left, upper, right, lower))
            patches.append(patch)
            coords.append((left, upper, right, lower))

    # Encode patches in batch
    inputs = processor(images=patches, return_tensors="pt").to(device)
    with torch.no_grad():
        embeds = model.get_image_features(**inputs)
        embeds = F.normalize(embeds, p=2, dim=-1).cpu()

    # Store embeddings and metadata
    for idx, (embed, (l, u, r, d)) in enumerate(zip(embeds, coords)):
        all_embeddings.append(embed)
        metadata.append({
            "page": page_idx,
            "patch_index": idx,
            "coords": [l, u, r, d],
            "patch_size": patch_size
        })

# Save results
torch.save(all_embeddings, os.path.join(OUTPUT_DIR, "image_patch_embeddings.pt"))
with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"[INFO] Saved {len(all_embeddings)} patch embeddings to {OUTPUT_DIR}")
