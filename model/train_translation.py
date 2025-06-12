from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import torch
from load_dataset import load_translation_dataset

import transformers
print("Transformers version:", transformers.__version__)


# Step 1: Load dataset
dataset = load_translation_dataset("./files/initial.json") # "./files/initial.json", "./cleaned_extracts"
dataset = dataset.train_test_split(test_size=0.1)

# Step 1.5: Flatten the nested translation fields
def flatten_translation(example):
    example["kln"] = example["translation"]["kln"]
    example["en"] = example["translation"]["en"]
    return example

dataset = dataset.map(flatten_translation)

# Step 2: Load base model and tokenizer
model_name = "Helsinki-NLP/opus-mt-ROMANCE-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Step 3: Preprocess (now this works with flat dataset)
def preprocess(batch):
    inputs = tokenizer(batch["kln"], truncation=True, padding="max_length", max_length=64)
    targets = tokenizer(batch["en"], truncation=True, padding="max_length", max_length=64)
    inputs["labels"] = targets["input_ids"]
    return inputs

print("Example sample:", dataset["train"][0])
print("Column names:", dataset["train"].column_names)

# Step 3.5: Tokenize
tokenized = dataset.map(preprocess, batched=True)

# Step 4: Training args
args = Seq2SeqTrainingArguments(
    output_dir="kalenjin-translator",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
    logging_dir="./logs",
    push_to_hub=False
)

# Step 5: Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

# Step 6: Train
trainer.train()
# Step 7: Save model and tokenizer
trainer.save_model("./kalenjin-translator")
tokenizer.save_pretrained("./kalenjin-translator")
print(f"Total training examples: {len(dataset)}")
print(dataset[0])
