# Vision-First RAG POC

A proof-of-concept that uses CLIP to retrieve and cluster image patches from PDF pages into full-width "rows" of interest, with OCR and local LLM-powered question answering.

## Project structure

```bash
.
├── .gitignore           # Git ignore rules (keep folders, ignore files inside)
├── README.md            # Project overview and instructions
├── offline/             # Offline data preparation scripts
│   └── prep_dataset.py  # (e.g. dataset download and formatting)
├── online/              # Inference code and live scripts
│   └── inference.py     # Retrieves patches, clusters rows, performs OCR + LLM Q&A
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
   pip install torch transformers pdf2image PyPDF2 pytesseract requests
   ```

3. **Install System Dependencies**
   
   You'll need Tesseract OCR and Ollama installed:
   
   - macOS: `brew install tesseract ollama`
   - Linux: `sudo apt install tesseract-ocr` and install Ollama from [ollama.ai](https://ollama.ai)

4. **Start Ollama**
   
   Make sure Ollama is running and has the Gemma model pulled:
   ```bash
   ollama serve  # in one terminal
   ollama pull gemma:3b  # in another terminal
   ```

5. **Prepare your PDFs**
   Drop your `.pdf` files into the `documents/` folder.

   ```bash
   python offline/prep_data.py
   ```

6. **Run inference**

   ```bash
   python online/inference.py
   ```

   * You'll be prompted for a text query (e.g. `what is ...`).
   * The script will:
     1. Find and cluster relevant image patches into rows
     2. Save full-width row crops under `output/top_rows/`
     3. Perform OCR on the retrieved rows
     4. Use a local Ollama LLM to answer your query based on the OCR'd text
   * Check the console for:
     - Saved row images
     - Raw OCR text
     - LLM-generated answer

## What's next?

* Tune the vertical clustering distance (`max_dist`) or `TOP_K` in `inference.py`
* Try different LLM models via Ollama
* Improve OCR quality with preprocessing
* Add tests to verify patch-to-row grouping

---

*This is a proof-of-concept combining CLIP-based visual retrieval with local OCR + LLM for quick document Q&A.*
