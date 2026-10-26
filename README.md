# Kalenjin Language & Dialects – Kalenjin ⇄ English Translation

This project builds a parallel English–Kalenjin dataset from Bible PDFs and uses it to fine-tune a neural machine translation model ([NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M)). The trained model can translate **Kalenjin → English** and **English → Kalenjin**, from the command line or from a Streamlit web app. It also learns a **Kalenjin ⇄ English word dictionary** from the aligned verses, and collects user **feedback** (corrections) for future training.

> **Scope:** for now the project is a generic Kalenjin tool (Nandi, Kipsigis, Keiyo, Marakwet, Tugen, Pokot, …). The long-term goal is a stable **Marakwet dialect** version, so dialect and sub-dialect are collected with every suggestion and answer. Audio and video are planned later.


Current data: 66 Bible books, about 31,000 aligned verse pairs (about 28,000 train / 3,100 validation).

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
                                                                                  │
                                                                     model/data/feedback.csv (user corrections)

train.csv ─► model/build_dictionary.py ─► model/data/dictionary.csv ─► app/app.py (Dictionary tab)
```

The model and the dictionary are **independent**: the dictionary is built with word-alignment statistics straight from `train.csv`; it is not part of the model weights and needs no training.

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
| `model/train_translation.py` | Fine-tunes NLLB in either direction (supports `--offset` / `--resume` for chunks). |
| `model/train_chunks.py` | Trains on the whole dataset one chunk at a time, resuming after each chunk and after interruptions. |
| `model/translate.py` | Translate from the command line (or import `Translator`). |
| `model/build_dictionary.py` | Learns word-to-word meanings (IBM Model 1 alignment, both directions) into `model/data/dictionary.csv`. |
| `app/app.py` | Streamlit web app: Translate tab (with feedback form) and Dictionary tab. |
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

- **Python 3.11** (3.12+ is not supported by the pinned `torch==2.5.1` / `scikit-learn==1.3.2`; they have no prebuilt wheels and the build fails). Do not use older torch (2.1) on Apple Silicon: its MPS backend gave NaN losses
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
| `--offset` | `0` | First training row to use (with `--max-samples`, selects one chunk) |
| `--resume` | off | Continue from the model already saved in the output folder instead of the base model |
| `--batch-size` / `--grad-accum` | `2` / `4` | Lower the batch size if memory runs out |
| `--train-embeddings` | off | Also train the large embedding matrix (needs much more memory) |

### Train on all the data, chunk by chunk

```bash
caffeinate -i python model/train_chunks.py --direction kln-en --chunk 2000   # macOS; drop `caffeinate -i` elsewhere
python model/train_chunks.py --direction en-kln --chunk 2000
```

It walks through `train.csv` 2000 rows at a time, resuming from the saved model after each chunk. Progress is stored in `model/data/progress_<direction>.txt`: press Ctrl+C and rerun the same command to continue (an interrupted chunk restarts). Use `--reset` to start over. On an Apple Silicon Mac a chunk takes about 30 minutes (about 7 hours for the full pass); a GPU or Google Colab is much faster. After any chunk you can test with `translate.py`. `eval_loss` must be a finite number that trends down; `nan` means something is wrong.
The first run downloads the base model (~2.4 GB). NLLB has no Kalenjin language code, so Swahili (`swh_Latn`) is used as a stand-in tag for Kalenjin; the model learns the language during fine-tuning. To change it, edit the language codes in `train_translation.py` and `translate.py`.

### Modify and retrain

- Fix or add data → rerun `prepare_train_data.py` → retrain (`train_chunks.py --reset`).
- Tune `learning_rate`, `per_device_train_batch_size` (lower it if you run out of memory) and `--epochs` in `train_translation.py`.
- Compare the validation loss printed after each epoch; if it rises while training loss falls, you are overfitting – use fewer epochs.

## 3. Translate locally

The saved model is complete (weights, tokenizer, config), so it works without the base-model download. Until you have trained on enough data, translations will be unreliable.

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

## 4. Word dictionary

```bash
python model/build_dictionary.py                      # ~2 min, writes model/data/dictionary.csv
python model/build_dictionary.py --lookup bik tugul   # look up words (Kalenjin or English)
```

It aligns words across all verse pairs (IBM Model 1, run in both directions; a pair is kept only if each word predicts the other). Examples learned: `bik` → people, `tugul` → all, `sait` → hour, `agenge` → one. Columns: `kalenjin, english, score, pair_count, kalenjin_count, example_kln, example_en`. Words seen in fewer than 3 verses are skipped, and very common English function words are filtered. Different forms of a word (`bik`, `bikab`, `bikyik`) are separate entries. Rebuild after `train.csv` changes.

## 5. Web app

```bash
streamlit run app/app.py
```

Open http://localhost:8501. The **Translate** tab lets you pick the direction and translate text (it needs the matching trained model folder in the repo root); the **Dictionary** tab searches the word dictionary learned from the Bible verses (build it first with `python model/build_dictionary.py`; look words up in the terminal with `--lookup bik tugul`). 

**Feedback:** under each translation, a form lets users submit the correct meaning (for missing words or wrong spelling). Entries are appended to `model/data/feedback.csv` with columns `time, direction, source, model_output, correction, dialect, sub_dialect, note, ip, user_agent`. Review them later and add good ones to `train.csv` or the dictionary.

**Dialect poll:** each dictionary entry has an "Is this right in your dialect?" form (yes / no + what you say / not sure, plus dialect and optional sub-dialect). Votes go to `model/data/word_votes.csv` (`time, kalenjin, english, vote, dialect, sub_dialect, suggestion, ip, user_agent`). The IP and user agent are stored only for later spam filtering; treat these CSVs as private and tell contributors the data is collected.

## 6. Quiz

The third tab runs rounds of 5 questions from the learned dictionary. About 70% are multiple choice with a known answer (confident entries; proper names are skipped) and are scored. The rest are open questions about uncertain words, where people type what the word means in their dialect. Answers go to `model/data/qa_answers.csv` (`time, question_type, kalenjin, expected, answer, correct, dialect, sub_dialect, ip, user_agent`). Many matching open answers from different people suggest a verified meaning.

## 7. Collecting data from a hosted app

Local CSVs are lost when a hosted app restarts, so every submission (feedback, dialect votes, quiz answers) is also pushed to a **private** GitHub repo, one small CSV file per submission under `data/<kind>/<date>/` (no write conflicts). Set these as secrets on the host (never commit them):

- `GITHUB_TOKEN`: a fine-grained token with *Contents: read and write* on the dataset repo only
- `DATASET_REPO`: e.g. `ezkemboi/kalenjin-dataset`

Without them the app only writes the local CSVs. To pull everything to your machine and merge it:

```bash
python model/pull_collected.py --repo ezkemboi/kalenjin-dataset
```

This writes `model/data/collected/{feedback,word_votes,qa_answers}.csv`, deduplicated. Review them (the IP column helps filter spam), add approved translation corrections to `train.csv`, retrain, and rebuild the dictionary. The repo must stay private because it contains IPs.

## Known gaps

- The corpus is Bible text only (one Kalenjin translation vs. English KJV), so everyday vocabulary (greetings, modern words) is weak until more data is added; user feedback is the planned way to fill this.
- `swh_Latn` (Swahili) is a stand-in language tag because NLLB has no Kalenjin code (`kln_Latn` is not in its vocabulary).
- The dictionary only knows words that appear in at least 3 verses and does not group word forms into stems.

## Research to do

- https://link.springer.com/rwe/10.1007/978-94-024-2068-5_197
