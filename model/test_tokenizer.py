from transformers import AutoTokenizer
import pandas as pd

# Load Meta's NLLB-200 base model tokenizer
MODEL_NAME = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Pull a sample verse from your newly generated training csv
try:
    df = pd.read_csv("model/data/train.csv")
    sample_kln = df['text_kln'].iloc[0]
    print(f"Testing Verse: {sample_kln}\n")
    
    # Check tokens
    tokens = tokenizer.tokenize(sample_kln)
    print(f"Token Breakdown:\n{tokens}")
except Exception as e:
    print(f"Run step 1 consolidation first. Error details: {e}")
