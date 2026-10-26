import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

def compile_dataset(json_dir="cleaned_extracts"):
    all_data = []
    
    # 1. Loop through your generated book files
    if not os.path.exists(json_dir):
        print(f"Error: {json_dir} folder not found. Run data-collections/combine_bible.py first.")
        return
        
    for file_name in os.listdir(json_dir):
        if file_name.endswith('.json'):
            with open(os.path.join(json_dir, file_name), 'r', encoding='utf-8') as f:
                try:
                    book_data = json.load(f)
                    all_data.extend(book_data)
                except Exception as e:
                    print(f"Skipping corrupted file {file_name}: {e}")
                    
    # 2. Convert to Pandas Matrix
    df = pd.DataFrame(all_data)
    
    # Clean up structure to match required model input fields
    # Dropping rows where text is blank or missing
    df = df.dropna(subset=['text_en', 'text_kln'])
    df = df[(df['text_en'].str.strip() != "") & (df['text_kln'].str.strip() != "")]
    
    # The tokenizer maps the curly apostrophe to <unk>; use a plain one
    for col in ('text_en', 'text_kln'):
        df[col] = df[col].str.replace('\u2019', "'", regex=False)

    print(f"Total compiled parallel sentences: {len(df)}")
    
    # 3. ML Best Practice: Split data into Train and Validation sets (90% Train, 10% Eval)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    # Save as CSVs inside your model directory
    os.makedirs("model/data", exist_ok=True)
    train_df.to_csv("model/data/train.csv", index=False)
    val_df.to_csv("model/data/validation.csv", index=False)
    print("Created model/data/train.csv and model/data/validation.csv successfully!")

if __name__ == "__main__":
    # Point this to the combined bilingual JSONs
    compile_dataset("cleaned_extracts") 
