import os
import json
from glob import glob
from collections import defaultdict

def extract_word_pairs(kln_text, en_text):
    kln_words = kln_text.strip(";.,:!?").split()
    en_words = en_text.strip(";.,:!?").split()

    # Align positionally (best effort)
    min_len = min(len(kln_words), len(en_words))
    return list(zip(kln_words[:min_len], en_words[:min_len]))

def build_glossary_from_files(folder_path, output_dir="glossary_extracts", output_filename="glossary.json"):
    glossary = defaultdict(set)

    os.makedirs(output_dir, exist_ok=True)

    for filepath in glob(os.path.join(folder_path, "*.json")):
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                entries = json.load(f)
            except Exception as e:
                print(f"⚠️ Failed to read {filepath}: {e}")
                continue

        for item in entries:
            kln = item.get("text_kln")
            en = item.get("text_en")
            if kln and en:
                pairs = extract_word_pairs(kln, en)
                for kln_word, en_word in pairs:
                    glossary[kln_word.strip(";.,:!?").lower()].add(en_word.strip(";.,:!?").lower())

    # Convert to list
    result = [{"text_kln": kln, "text_en": sorted(list(ens))} for kln, ens in glossary.items()]

    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ Glossary created: {output_path} ({len(result)} entries)")

# Run script
if __name__ == "__main__":
    build_glossary_from_files(folder_path="./cleaned_extracts")
