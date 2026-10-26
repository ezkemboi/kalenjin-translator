"""Translate with a fine-tuned model. Usage:
  python model/translate.py --direction kln-en "Amunet ne nyolu ke tugul"
  python model/translate.py --direction en-kln          (interactive)
"""
import argparse
import warnings

import torch

warnings.filterwarnings("ignore", category=FutureWarning)

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402

hf_logging.set_verbosity_error()

DIRECTIONS = {
    "kln-en": ("kalenjin-translator", "swh_Latn", "eng_Latn"),
    "en-kln": ("kalenjin-translator-en-kln", "eng_Latn", "swh_Latn"),
}


class Translator:
    def __init__(self, direction="kln-en"):
        self.model_dir, src, self.tgt = DIRECTIONS[direction]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, src_lang=src)
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir).to(self.device).eval()

    def __call__(self, text):
        # Training data uses a plain apostrophe
        text = text.replace("\u2019", "'")
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        n = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.tgt),
                max_length=min(128, int(n * 2) + 10),
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
