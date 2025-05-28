#!/usr/bin/env python3
import os
import json
import torch
import torch.nn.functional as F
import glob
from io import BytesIO
from PyPDF2 import PdfMerger
from pdf2image import convert_from_bytes
from transformers import CLIPModel, CLIPProcessor

# — — —  Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOC_DIR = "documents"
OUTPUT_DIR = "output"
# Number of patches per page (must be a perfect square, e.g., 100 = 10x10)
PATCHES_PER_PAGE = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

# — — —  Load CLIP model & processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# — — —  Gather PDFs
pdf_paths = sorted(glob.glob(os.path.join(DOC_DIR, "*.pdf")))
if not pdf_paths:
    raise FileNotFoundError(f"No PDF files found in {DOC_DIR}")

# — — —  Merge PDFs into a single buffer
merger = PdfMerger()
for path in pdf_paths:
    merger.append(path)
buf = BytesIO()
merger.write(buf)
merger.close()
buf.seek(0)

# — — —  Convert merged PDF pages to images
doc_images = convert_from_bytes(buf.read(), dpi=150)
print(f"[INFO] Converted {len(doc_images)} pages from {len(pdf_paths)} PDFs")

# — — —  Calculate grid size
grid_size = int(PATCHES_PER_PAGE ** 0.5)
if grid_size * grid_size != PATCHES_PER_PAGE:
    raise ValueError("PATCHES_PER_PAGE must be a perfect square")

# Containers for embeddings and metadata
all_embeddings = []  # list of torch.Tensor [dim]
metadata = []        # list of dicts

# — — —  Process each page
for page_idx, img in enumerate(doc_images):
    width, height = img.size
    patch_w = width // grid_size
    patch_h = height // grid_size

    patches = []
    coords = []
    for r in range(grid_size):
        for c in range(grid_size):
            left = c * patch_w
            upper = r * patch_h
            right = left + patch_w
            lower = upper + patch_h
            patch = img.crop((left, upper, right, lower))
            patches.append(patch)
            coords.append((left, upper, right, lower))

    print(f"[INFO] Page {page_idx}: {len(patches)} patches")

    # Encode patches in batch
    inputs = processor(images=patches, return_tensors="pt").to(device)
    with torch.no_grad():
        embeds = model.get_image_features(**inputs)  # shape: [P, D]
        embeds = F.normalize(embeds, p=2, dim=-1)
        embeds = embeds.cpu()

    # Store embeddings and metadata
    for idx, (embed, (l, u, r, d)) in enumerate(zip(embeds, coords)):
        all_embeddings.append(embed)
        metadata.append({
            "page": page_idx,
            "patch_index": idx,
            "coords": [l, u, r, d]
        })

# — — —  Save results
torch.save(all_embeddings, os.path.join(OUTPUT_DIR, "image_patch_embeddings.pt"))
with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"[INFO] Saved {len(all_embeddings)} patch embeddings to {OUTPUT_DIR}")
