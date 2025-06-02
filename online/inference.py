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

# OCR
import pytesseract
import requests
from PIL import Image

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

# --- OCR + LLM helpers ---------------------------------------------

def ocr_image(img):
    """Return raw UTF-8 text from a PIL.Image."""
    return pytesseract.image_to_string(img, lang='eng', config='--psm 6').strip()

def call_ollama(prompt: str,
                model: str = "gemma3:4b-it-qat",
                host: str = "http://localhost:11434") -> str:
    """Blocking call to Ollama /api/generate with stream=False."""
    url = f"{host}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["response"]


# — — —  Main
def main():
    query = input("Enter your query: ")
    q_vec = embed_text(query)

    # --- compute similarities once --------------------------
    emb_tensor = F.normalize(torch.stack(all_embeds).to(device), p=2, dim=-1)

    q_full  = embed_text(query)
    keywords = call_ollama(
        f"return 1–3 key noun phrases that appear verbatim in: {query}"
    ).strip()
    q_key   = embed_text(keywords)

    q_vec   = F.normalize(q_full + 0.5 * q_key, p=2, dim=-1)
    sims    = emb_tensor @ q_vec            # cosine because vectors are unit-norm
    topk_val, topk_idx = torch.topk(sims, k=TOP_K * 2)   # grab a few extra

    patches = [(i.item(), v.item(), metadata[i.item()])
            for i, v in zip(topk_idx, topk_val)]

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

    # ── collect OCR text from the saved crops ───────────────────────
    context_chunks = []
    for i in range(1, len(rows)+1):
        crop_path = os.path.join(TOP_DIR, f"row{i}.png")
        with Image.open(crop_path) as im:
            txt = ocr_image(im)
            if txt:
                context_chunks.append(txt)

    context = "\n".join(context_chunks)[:8000]  # keep it short; 8k ≈ 2k tokens

    # ── plug it into Gemma 3 ────────────────────────────────────────
    sys_msg = ("You are a terse Q&A assistant. If the context lacks the answer, "
               "say 'Not found'.")
    prompt = f"{sys_msg}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    answer = call_ollama(prompt)

    print("\n=== RAW OCR CONTEXT ===")
    print(context)

    print("\n=== LLM ANSWER ===")
    print(answer)


if __name__ == '__main__':
    main()