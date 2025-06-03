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
DPI        = 150
TOP_K      = 6
NUM_NEIGHBOR_LAYERS = 1 # 0 for tight box, 1 for 1-layer avg patch padding

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
    inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    return F.normalize(feats.squeeze(0), p=2, dim=-1)

# — — —  Cluster patches into row groups (by vertical proximity)
def cluster_patches(patches, num_neighbor_layers=1, dist_thresh=100):
    """
    Clusters patches into tight rectangular groups based on spatial proximity (L2 in center coords).
    Then expands each group based on average patch size and `num_neighbor_layers`.
    """
    if not patches:
        return []

    from math import hypot

    def center(meta):
        l, u, r, d = meta['coords']
        return ((l + r) / 2, (u + d) / 2)

    def close_enough(meta1, meta2):
        x1, y1 = center(meta1)
        x2, y2 = center(meta2)
        return hypot(x1 - x2, y1 - y2) <= dist_thresh and meta1['page'] == meta2['page']

    # Greedy clustering (disjoint sets based on proximity)
    clusters = []
    used = set()

    for i, (idx1, score1, meta1) in enumerate(patches):
        if i in used:
            continue
        cluster = [(idx1, score1, meta1)]
        used.add(i)
        for j, (idx2, score2, meta2) in enumerate(patches):
            if j in used:
                continue
            if close_enough(meta1, meta2):
                cluster.append((idx2, score2, meta2))
                used.add(j)
        clusters.append(cluster)

    # Build bounding boxes
    grouped = []
    for cluster in clusters:
        boxes = [meta['coords'] for _, _, meta in cluster]
        page_idx = cluster[0][2]['page']

        widths  = [r - l for l, u, r, d in boxes]
        heights = [d - u for l, u, r, d in boxes]
        avg_w = sum(widths) / len(widths) if widths else 0.0
        avg_h = sum(heights) / len(heights) if heights else 0.0

        margin_x = num_neighbor_layers * avg_w
        margin_y = num_neighbor_layers * avg_h

        lefts   = [b[0] for b in boxes]
        uppers  = [b[1] for b in boxes]
        rights  = [b[2] for b in boxes]
        lowers  = [b[3] for b in boxes]

        l = max(0.0, min(lefts) - margin_x)
        u = max(0.0, min(uppers) - margin_y)
        r = max(l, max(rights) + margin_x)
        d = max(u, max(lowers) + margin_y)

        grouped.append({
            "page": page_idx,
            "bbox": (l, u, r, d),
            "patches": cluster
        })

    return grouped

# --- OCR + LLM helpers ---------------------------------------------

def ocr_image(img):
    return pytesseract.image_to_string(img, lang='eng', config='--psm 6 -c preserve_interword_spaces=1').strip()

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
    topk_val, topk_idx = torch.topk(sims, k=TOP_K)

    patches = [(i.item(), v.item(), metadata[i.item()])
            for i, v in zip(topk_idx, topk_val)]

    # cluster into rows
    groups = cluster_patches(patches, num_neighbor_layers=NUM_NEIGHBOR_LAYERS)

    print("=== TOP ROWS OF PATCHES ===")
    for i, group in enumerate(groups, start=1):
        page_no = group["page"]
        l, u, r, d = group["bbox"]

        page_img = page_images[page_no]
        bbox = (l, u, r, d)
        group_crop = page_img.crop(bbox)

        out_path = os.path.join(TOP_DIR, f"group{i}.png")
        group_crop.save(out_path)
        print(f"[Group {i}] page={page_no}, bbox={bbox}, patches={len(group['patches'])} -> saved: {out_path}")
        for idx, score, meta in group['patches']:
            print(f"    patch {meta['patch_index']}, coords={meta['coords']}, score={score:.4f}")
        print("-"*40)
    print("Done. Rows saved with full-width context.")

    # ── collect OCR text from the saved crops ───────────────────────
    context_chunks = []
    for i in range(1, len(groups)+1):
        crop_path = os.path.join(TOP_DIR, f"group{i}.png")
        with Image.open(crop_path) as im:
            txt = ocr_image(im)
            if txt:
                print(f"[DEBUG] group{i} raw OCR: {repr(txt[:2000])}")
                context_chunks.append(txt)

    context = "\n".join(context_chunks)[:8000]  # keep it short; 8k ≈ 2k tokens

    # ── plug it into Gemma 3 ────────────────────────────────────────
    # lets first clean the OCR output, using a run of LLM
    cleaned_context = call_ollama(f"normalize the following OCR text for readability while keeping all names, numbers, and symbols intact. Do not omit any lines. Preserve proper nouns and formatting: {context}")
    # print(f"[INFO] Cleaned context: {cleaned_context}")

    sys_msg = ("You are a terse Q&A assistant. If the context lacks the answer, "
               "say 'Not found'.")
    prompt = f"{sys_msg}\n\nContext:\n{cleaned_context}\n\nQuestion: {query}\nAnswer:"
    answer = call_ollama(prompt)

    print("\n=== CLEANED OCR CONTEXT ===")
    print(cleaned_context)

    print("\n=== LLM ANSWER ===")
    print(answer)


if __name__ == '__main__':
    main()