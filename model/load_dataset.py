from datasets import Dataset
import json
import os

# pairs = [{"translation": {"kln": item["text_kln"], "en": item["text_en"]}} for item in data]
# return Dataset.from_list(pairs)

def load_translation_dataset(mixed_path="./files/initial.json", folder_path="./cleaned_extracts"):
    data = []

    # Load the single JSON file (assuming it's a list of items)
    if os.path.exists(mixed_path):
        with open(mixed_path, 'r', encoding='utf-8') as f:
            try:
                file_data = json.load(f)
                for item in file_data:
                    if "text_kln" in item and "text_en" in item:
                        data.append({
                            "translation": {
                                "kln": item["text_kln"].strip(),
                                "en": item["text_en"].strip()
                            }
                        })
            except Exception as e:
                print("Error reading file:", mixed_path, "->", e)

    # Load all JSON files in the cleaned_extracts folder
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                path = os.path.join(folder_path, filename)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        item = json.load(f)
                        if "text_kln" in item and "text_en" in item:
                            data.append({
                                "translation": {
                                    "kln": item["text_kln"].strip(),
                                    "en": item["text_en"].strip()
                                }
                            })
                except Exception as e:
                    print("Skipping file", filename, "->", e)

    return Dataset.from_list(data)
