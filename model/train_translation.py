import argparse

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

MODEL_NAME = "facebook/nllb-200-distilled-600M"
MAX_LEN = 128

parser = argparse.ArgumentParser()
parser.add_argument("--direction", choices=["kln-en", "en-kln"], default="kln-en")
parser.add_argument("--epochs", type=float, default=3)
parser.add_argument("--max-samples", type=int, default=None, help="use a small subset for a quick test run")
parser.add_argument("--batch-size", type=int, default=2, help="per-device batch size; lower it if memory runs out")
parser.add_argument("--grad-accum", type=int, default=4, help="gradient accumulation steps")
parser.add_argument("--train-embeddings", action="store_true", help="also train the large shared embedding matrix (needs much more memory)")
cli = parser.parse_args()

# NLLB has no Kalenjin code; a Latin-script code (Swahili) stands in for Kalenjin
if cli.direction == "kln-en":
    SRC_COL, TGT_COL, SRC_LANG, TGT_LANG = "text_kln", "text_en", "swh_Latn", "eng_Latn"
    OUTPUT_DIR = "kalenjin-translator"
else:
    SRC_COL, TGT_COL, SRC_LANG, TGT_LANG = "text_en", "text_kln", "eng_Latn", "swh_Latn"
    OUTPUT_DIR = "kalenjin-translator-en-kln"

# Step 1: Load the CSVs produced by data-collections/prepare_train_data.py
train_df = pd.read_csv("model/data/train.csv")[[SRC_COL, TGT_COL]]
val_df = pd.read_csv("model/data/validation.csv")[[SRC_COL, TGT_COL]]
if cli.max_samples:
    train_df, val_df = train_df.head(cli.max_samples), val_df.head(max(cli.max_samples // 10, 8))
train_ds, val_ds = Dataset.from_pandas(train_df), Dataset.from_pandas(val_df)

# Step 2: Load base model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
if not cli.train_embeddings:
    # The 256k-token embedding matrix (tied to the output layer) dominates memory
    model.get_input_embeddings().weight.requires_grad = False

# Step 3: Tokenize (padding is handled per batch by the data collator)
def preprocess(batch):
    return tokenizer(
        batch[SRC_COL],
        text_target=batch[TGT_COL],
        truncation=True,
        max_length=MAX_LEN,
    )

train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
val_tok = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)

# Step 4: Training args
args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=cli.batch_size,
    gradient_accumulation_steps=cli.grad_accum,
    per_device_eval_batch_size=cli.batch_size,
    num_train_epochs=cli.epochs,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
    generation_max_length=MAX_LEN,
    logging_dir="./logs",
    push_to_hub=False,
)

# Step 5: Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

# Step 6: Train and save
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Train examples: {len(train_ds)}, validation examples: {len(val_ds)}")
