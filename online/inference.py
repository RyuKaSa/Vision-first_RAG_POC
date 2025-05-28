#!/usr/bin/env python3
import os
import json
import glob
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from pdf2image import convert_from_bytes
from PyPDF2 import PdfMerger
from io import BytesIO

# — — —  Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOC_DIR    = "documents"
OUTPUT_DIR = "output"
EMBED_FILE = os.path.join(OUTPUT_DIR, "image_patch_embeddings.pt")
META_FILE  = os.path.join(OUTPUT_DIR, "metadata.json")
TOP_DIR    = os.path.join(OUTPUT_DIR, "top_rows")
DPI        = 150  # must match indexing DPI
TOP_K      = 4

os.makedirs(TOP_DIR, exist_ok=True)
# clean output folder
for f in os.listdir(TOP_DIR):
    os.remove(os.path.join(TOP_DIR, f))

# — — —  Load CLIP model & processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# — — —  Load embeddings and metadata
all_embeds = torch.load(EMBED_FILE)
with open(META_FILE, 'r') as f:
    metadata = json.load(f)

# — — —  Merge PDFs and render pages
tmp = BytesIO()
merger = PdfMerger()
for p in sorted(glob.glob(os.path.join(DOC_DIR, "*.pdf"))):
    merger.append(p)
merger.write(tmp)
merger.close()
tmp.seek(0)
page_images = convert_from_bytes(tmp.read(), dpi=DPI)
print(f"[INFO] Loaded {len(page_images)} pages")

# — — —  Text embed helper
def embed_text(query: str) -> torch.Tensor:
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    return F.normalize(feats.squeeze(0), p=2, dim=-1)

# — — —  Cluster patches into row groups (by vertical proximity)
def cluster_patches(patches, max_dist=20):
    rows = []
    for idx, score, meta in patches:
        l,u,_,d = meta['coords']
        mid_y = (u + d) / 2
        placed = False
        for row in rows:
            _,_,m0 = row[0]
            u0,d0 = m0['coords'][1], m0['coords'][3]
            mid_y0 = (u0 + d0) / 2
            if meta['page']==m0['page'] and abs(mid_y - mid_y0) < max_dist:
                row.append((idx, score, meta))
                placed = True
                break
        if not placed:
            rows.append([(idx, score, meta)])
    return rows

# — — —  Main
def main():
    query = input("Enter your query: ")
    q_vec = embed_text(query)

    # compute similarities
    emb_tensor = torch.stack(all_embeds).to(device)
    emb_tensor = F.normalize(emb_tensor, p=2, dim=-1)
    sims = emb_tensor @ q_vec
    probs = F.softmax(sims, dim=0)
    topk = torch.topk(probs, k=TOP_K)

    # gather top-k patches
    patches = [(idx.item(), val.item(), metadata[idx.item()])
               for idx,val in zip(topk.indices, topk.values)]

    # cluster into rows
    rows = cluster_patches(patches)

    print("=== TOP ROWS OF PATCHES ===")
    for i, row in enumerate(rows, start=1):
        # Determine vertical bounds from row patches
        page_no = row[0][2]['page']
        ys = [m['coords'][1] for _,_,m in row] + [m['coords'][3] for _,_,m in row]
        y_min, y_max = min(ys), max(ys)

        # Crop full width of page between y_min and y_max for context
        page_img = page_images[page_no]
        width, height = page_img.size
        bbox = (0, y_min, width, y_max)
        row_crop = page_img.crop(bbox)

        # Save and print metadata
        out_path = os.path.join(TOP_DIR, f"row{i}.png")
        row_crop.save(out_path)
        print(f"[Row {i}] page={page_no}, patches={len(row)} -> saved: {out_path}")
        for idx, score, meta in row:
            print(f"    patch {meta['patch_index']}, coords={meta['coords']}, score={score:.4f}")
        print("-"*40)
    print("Done. Rows saved with full-width context.")

if __name__ == '__main__':
    main()