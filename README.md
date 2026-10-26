# Marakwet AI – Kalenjin ⇄ English Translation

Marakwet AI builds a parallel English–Kalenjin (Marakwet) dataset from Bible PDFs and uses it to fine-tune a neural machine translation model ([NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M)). The trained model can translate **Kalenjin → English** and **English → Kalenjin**, from the command line or from a Streamlit web app that also contains a small Kalenjin dictionary.

## How it works

```
Kalenjin PDF ──► kalenjin_bible_extractor.py ──► extracts/*.json ──┐
                                                                   ├─► combine_bible.py ─► cleaned_extracts/*.json
English (KJV) ─► kjv_bible_extractor.py ───────► en_extracts/*.json ┘            │
                                                                                  ▼
                                                           prepare_train_data.py ─► model/data/train.csv, validation.csv
                                                                                  │
                                                           model/train_translation.py (fine-tune NLLB)
                                                                                  │
                                                  kalenjin-translator/  (Kalenjin → English)
                                                  kalenjin-translator-en-kln/  (English → Kalenjin)
                                                                                  │
                                                  model/translate.py (CLI)   app/app.py (web app)
```

## Project layout

| Path | Purpose |
|------|---------|
| `data-collections/kalenjin_bible_extractor.py` | Extracts Kalenjin verses from the PDF into one JSON file per book (`extracts/`). |
| `data-collections/kjv_bible_extractor.py` | Downloads/extracts the English KJV Bible per book (`en_extracts/`). |
| `data-collections/combine_bible.py` | Aligns English and Kalenjin verses into `cleaned_extracts/`. |
| `data-collections/prepare_train_data.py` | Builds `model/data/train.csv` and `validation.csv` (90/10 split) from `cleaned_extracts/`. |
| `data-collections/extract_glossary.py` | Builds a rough word-level glossary from the aligned verses. |
| `data-collections/book_mappings.py`, `kln_book_pages_ranges.py` | Book name mappings and PDF page ranges. |
| `model/test_tokenizer.py` | Shows how the tokenizer splits a Kalenjin verse (look for `<unk>`). |
| `model/train_translation.py` | Fine-tunes NLLB in either direction. |
| `model/translate.py` | Translate from the command line (or import `Translator`). |
| `app/app.py` | Streamlit web app: Translate tab and Dictionary tab. |
| `not-used/` | Old extractors kept for reference. |

Generated folders (`extracts/`, `en_extracts/`, `cleaned_extracts/`, `kalenjin-translator*/`, `model/data/`) and all `*.json`, `*.csv`, `*.pdf` files are git-ignored.

Verse format in `cleaned_extracts/*.json`:

```json
{
  "id": "1 John#1#1",
  "book_kln": "Johana 1",
  "book_en": "1 John",
  "chapter": 1,
  "verse": 1,
  "text_en": "That which was from the beginning...",
  "text_kln": "Noto ne kimi koboch taunet..."
}
```

## Requirements

- **Python 3.11** (3.12+ is not supported by the pinned `torch==2.1.0` / `scikit-learn==1.3.2`; they have no prebuilt wheels and the build fails)
- ~5 GB free disk (model download ≈ 2.4 GB, plus checkpoints)
- 16 GB RAM recommended. Training on CPU is very slow; use a GPU (NVIDIA CUDA) or Google Colab for full training.
- Python packages are listed in [requirements.txt](requirements.txt): pdfplumber, tqdm, python-dotenv, scikit-learn, pandas, pyarrow, transformers, torch, datasets, accelerate, sentencepiece, streamlit.

## Setup

### macOS / Linux

```bash
# Install Python 3.11 first, e.g. `brew install python@3.11` or `pyenv install 3.11.9`
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> pyenv users: make sure Python was built with `lzma` support (`brew install xz` before `pyenv install`), otherwise `datasets` fails with `No module named '_lzma'`.

### Windows (PowerShell)

```powershell
# Install Python 3.11 from python.org (tick "Add to PATH") or: winget install Python.Python.3.11
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1      # cmd.exe: .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

All commands below are run from the repository root. On Windows use `python` instead of `python3`.

## 1. Prepare the data

Only needed if you don't already have `extracts/`, `en_extracts/` and `cleaned_extracts/`.

1. Copy `.env.example` to `.env` and fill in the paths (`KLN_BIBLE_PATH`, `KLN_BIBLE_OUTPUT_DIR`, `EN_BIBLE_OUTPUT_DIR`, `EN_BIBLE_URL`, …). Put the Kalenjin PDF in `data-collections/`.
2. Run the extractors, then combine:

```bash
python data-collections/kalenjin_bible_extractor.py
python data-collections/kjv_bible_extractor.py
python data-collections/combine_bible.py
python data-collections/prepare_train_data.py
```


`prepare_train_data.py` also replaces the curly apostrophe `’` with `'`, because the NLLB tokenizer turns `’` into `<unk>` (very common in Kalenjin, e.g. `ng'alek`). Check tokenization any time with:

```bash
python model/test_tokenizer.py
```

## 2. Train

```bash
# Kalenjin → English  (saves to kalenjin-translator/)
python model/train_translation.py --direction kln-en

# English → Kalenjin  (saves to kalenjin-translator-en-kln/)
python model/train_translation.py --direction en-kln
```

Options:

| Flag | Default | Meaning |
|------|---------|---------|
| `--direction` | `kln-en` | `kln-en` or `en-kln` |
| `--epochs` | `3` | Number of training epochs |
| `--max-samples` | all | Train on a small subset, e.g. `--max-samples 200 --epochs 1` for a quick sanity check |

The first run downloads the base model (~2.4 GB). NLLB has no Kalenjin language code, so Swahili (`swh_Latn`) is used as a stand-in tag for Kalenjin; the model learns the language during fine-tuning. To change it, edit the language codes in `train_translation.py` and `translate.py`.

### Modify and retrain

- Fix or add data → rerun `prepare_train_data.py` → retrain.
- Tune `learning_rate`, `per_device_train_batch_size` (lower it if you run out of memory) and `--epochs` in `train_translation.py`.
- Compare the validation loss printed after each epoch; if it rises while training loss falls, you are overfitting – use fewer epochs.

## 3. Translate locally

```bash
python model/translate.py --direction kln-en "Amunet ne nyolu ke tugul"
python model/translate.py --direction en-kln "God is good"
python model/translate.py --direction kln-en          # interactive; empty line to quit
```

From Python:

```python
import sys; sys.path.append("model")
from translate import Translator
print(Translator("en-kln")("In the beginning"))
```

## 4. Web app

```bash
streamlit run app/app.py
```

Open http://localhost:8501. The **Translate** tab lets you pick the direction and translate text (it needs the matching trained model folder in the repo root); the **Dictionary** tab searches `app/kalenjin_data.json`.

## Known gaps

- Missing Kalenjin books: Komong'u (Exodus), Ng'atutik (Deuteronomy), Tesalonikaek 2 (2 Thessalonians).
- Failed to combine: Acts (`boisionikab_che_kiyoktoi.json`, English file missing), Job (`job.json`, both files missing), Ephesians (`epesoek.json`, English file missing).
- The corpus is Bible text only, so translations of everyday vocabulary will be weak until more data is added.

## Research to do

- https://link.springer.com/rwe/10.1007/978-94-024-2068-5_197
