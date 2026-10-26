"""Translate with a fine-tuned model. Usage:
  python model/translate.py --direction kln-en "Amunet ne nyolu ke tugul"
  python model/translate.py --direction en-kln          (interactive)
"""
import argparse

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DIRECTIONS = {
    "kln-en": ("kalenjin-translator", "swh_Latn", "eng_Latn"),
    "en-kln": ("kalenjin-translator-en-kln", "eng_Latn", "swh_Latn"),
}


class Translator:
    def __init__(self, direction="kln-en"):
        self.model_dir, src, self.tgt = DIRECTIONS[direction]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, src_lang=src)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir)

    def __call__(self, text):
        # Training data uses a plain apostrophe
        text = text.replace("\u2019", "'")
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        out = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.tgt),
            max_length=128,
            num_beams=4,
            no_repeat_ngram_size=3,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=DIRECTIONS, default="kln-en")
    parser.add_argument("text", nargs="*")
    args = parser.parse_args()
    translator = Translator(args.direction)
    if args.text:
        print(translator(" ".join(args.text)))
    else:
        while (line := input("> ").strip()):
            print(translator(line))
