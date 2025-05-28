# Vision-First Retrieval POC

A proof-of-concept that uses CLIP to retrieve and cluster image patches from PDF pages into full-width “rows” of interest, for downstream processing.

## Project structure

```bash
.
├── .gitignore           # Git ignore rules (keep folders, ignore files inside)
├── README.md            # Project overview and instructions
├── offline/             # Offline data preparation scripts
│   └── prep_dataset.py  # (e.g. dataset download and formatting)
├── online/              # Inference code and live scripts
│   └── inference.py     # Retrieves top-K patches, clusters rows, saves crops
├── documents/           # Source PDFs (keep folder, ignore contents)
├── output/              # Generated artifacts (ignore files, keep folders)
│   ├── image_patch_embeddings.pt  # CLIP embeddings
│   ├── metadata.json              # Patch-coord metadata
│   └── top_rows/                  # Saved full-row images
└── list_files.py        # Utility to list project files
```


## Getting started

1. **Create & activate your venv**  
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2. **Install dependencies**

   ```bash
   pip install torch transformers pdf2image PyPDF2
   ```

3. **Prepare your PDFs**
   Drop your `.pdf` files into the `documents/` folder.

   ```bash
   python offline/prep_data.py
   ```

4. **Run inference**

   ```bash
   python online/inference.py
   ```

   * You’ll be prompted for a text query (e.g. `what is ...`).
   * The script will output the top-k patches, cluster them into rows, and save full-width row crops under `output/top_rows/`.
   * Check the console for which row images were saved.

## What’s next?

* Hook these row crops into a vision-language model or OCR+LLM pipeline.
* Tune the vertical clustering distance (`max_dist`) or `TOP_K` in `inference.py`.
* Add tests to verify patch-to-row grouping.

---

*Proof-of-Concept only – no OCR/text-extraction is performed here.*
