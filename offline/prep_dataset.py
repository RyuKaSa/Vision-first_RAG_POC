#!/usr/bin/env python3
import os
import json
import time
import torch
import numpy as np
import glob
from io import BytesIO
from PyPDF2 import PdfMerger
from pdf2image import convert_from_bytes
import statistics
from transformers import AutoConfig
from transformers.utils.import_utils import is_flash_attn_2_available
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    dispatch_model
)
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

# — — —  Setup device & attention impl
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps"  if torch.backends.mps.is_available()
    else "cpu"
)
attn = "flash_attention_2" if is_flash_attn_2_available() else None

# — — —  Paths and directories
DOC_DIR    = "documents"
OUTPUT_DIR = "output"
OFFLOAD    = "./offload"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OFFLOAD,    exist_ok=True)

# — — —  Model identifiers
FINE_TUNED = "vidore/colqwen2.5-v0.2"
BASE_VL    = "Qwen/Qwen2.5-VL-3B-Instruct"

# — — —  1) Load the proper VL config (with vision_config)
config = AutoConfig.from_pretrained(BASE_VL)

# — — —  2) Build empty-weights shell and infer device_map if CUDA
if torch.cuda.is_available():
    with init_empty_weights():
        empty_model = ColQwen2_5._from_config(config)
    empty_model.tie_weights()  # avoid “weights not tied” warning
    device_map = infer_auto_device_map(
        empty_model,
        no_split_module_classes=["Qwen2Layer"],
        max_memory={0: "4GiB", "cpu": "12GiB"}
    )
    # 3a) Load fine-tuned weights
    model = ColQwen2_5.from_pretrained(
        FINE_TUNED,
        config=config,
        torch_dtype=torch.float16,
        offload_folder=OFFLOAD,
        attn_implementation=attn,
    ).eval()
    dispatch_model(
        model,
        device_map=device_map,
        offload_folder=OFFLOAD
    )
else:
    # 3b) MPS/CPU only: load entire model on device
    model = ColQwen2_5.from_pretrained(
        FINE_TUNED,
        config=config,
        torch_dtype=torch.float16,
        attn_implementation=attn,
    ).eval().to(device)

# — — —  4) Processor
processor = ColQwen2_5_Processor.from_pretrained(FINE_TUNED)

# — — —  5) gather all PDFs
pdf_paths = sorted(glob.glob(os.path.join(DOC_DIR, "*.pdf")))
if not pdf_paths:
    raise FileNotFoundError(f"No PDF files found in {DOC_DIR}")

# — — —  6) Merge all PDFs into a single PDF
merger = PdfMerger()
for path in pdf_paths:
    merger.append(path)
buffer = BytesIO()
merger.write(buffer)
merger.close()
buffer.seek(0)

images = convert_from_bytes(buffer.read(), dpi=150)

print(f"[INFO] Converted {len(images)} pages from {len(pdf_paths)} PDFs")

# Prepare containers for embeddings and stats
all_embeds      = []
all_page_stats  = []

for i, img in enumerate(images):
    stats = {"page": i}
    t_start = time.perf_counter()

    # 5.1) Preprocess timing
    t_p0 = time.perf_counter()
    inputs = processor.process_images([img]).to(device, torch.float16)
    t_p1 = time.perf_counter()

    # 5.2) Patch-level uniformity metrics
    arr = np.array(img)
    ph, pw = 32, 32
    patches = [
        arr[y:y+ph, x:x+pw]
        for y in range(0, arr.shape[0], ph)
        for x in range(0, arr.shape[1], pw)
    ]
    means = [p.mean() for p in patches]
    stds  = [p.std()  for p in patches]
    total_patches = len(patches)

    # 5.3) Forward-pass timing and embedding
    t_f0 = time.perf_counter()
    with torch.no_grad():
        embed = model(**inputs)
    t_f1 = time.perf_counter()

    # 5.4) Collect embedding diagnostics
    em_np = embed.cpu().numpy().reshape(-1, embed.shape[-1])
    norms = np.linalg.norm(em_np, axis=1)
    zeros = int((norms == 0).sum())

    # 5.5) Append embedding and stats
    all_embeds.append(embed.cpu())
    stats.update({
        "preprocess_s":       t_p1 - t_p0,
        "forward_s":          t_f1 - t_f0,
        "total_page_s":       t_f1 - t_start,

        "total_patches":      total_patches,
        "patch_int_mean":     float(np.mean(means)),
        "patch_int_std":      float(np.mean(stds)),
        "patch_int_min":      float(np.min(means)),
        "patch_int_max":      float(np.max(means)),

        "embed_shape":        list(embed.shape),
        "zero_embedding_ct":  zeros,
        "embed_norm_mean":    float(norms.mean()),
        "embed_norm_std":     float(norms.std()),
    })
    all_page_stats.append(stats)
    print(f"[INFO] Page {i} stats: {stats}")

# — — —  6) Save embeddings and metadata
torch.save(all_embeds, os.path.join(OUTPUT_DIR, "image_embeddings.pt"))
with open(os.path.join(OUTPUT_DIR, "metadata_stats.json"), "w") as f:
    summary = {
        "pages":            len(all_page_stats),
        "avg_preprocess_s": statistics.mean(p["preprocess_s"] for p in all_page_stats),
        "avg_forward_s":    statistics.mean(p["forward_s"]   for p in all_page_stats),
        "avg_total_s":      statistics.mean(p["total_page_s"] for p in all_page_stats),
        "avg_patches":      statistics.mean(p["total_patches"] for p in all_page_stats),
        "total_zero_embeds":sum(p["zero_embedding_ct"] for p in all_page_stats),
        "throughput_pps":   len(all_page_stats) / sum(p["total_page_s"] for p in all_page_stats),
    }
    json.dump({"per_page": all_page_stats, "summary": summary}, f, indent=2)

print("=== SUMMARY ===")
print(json.dumps(summary, indent=2))
