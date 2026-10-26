"""Run training inside Google Colab. Models and progress live on Google Drive, so a
session timeout loses nothing: rerun the same command to continue.

One-time: put train.csv and validation.csv in  MyDrive/kalenjin/  (from model/data/).

In a Colab cell (GPU runtime):
  !git clone https://github.com/ezkemboi/kalenjin-translator
  %cd kalenjin-translator
  %run model/colab_run.py --direction kln-en
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--direction", choices=["kln-en", "en-kln"], default="kln-en")
parser.add_argument("--chunk", type=int, default=2000)
parser.add_argument("--grad-accum", type=int, default=2)
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--reset", action="store_true")
args, _ = parser.parse_known_args()

from google.colab import drive  # noqa: E402

drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive/kalenjin")
DRIVE.mkdir(parents=True, exist_ok=True)

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers==4.44.2",
                       "accelerate", "sentencepiece"])

Path("model/data").mkdir(parents=True, exist_ok=True)
for name in ("train.csv", "validation.csv"):
    src = DRIVE / name
    if not src.exists():
        sys.exit(f"Missing {src}. Upload train.csv and validation.csv to MyDrive/kalenjin/ first.")
    shutil.copy(src, f"model/data/{name}")

# Save the model and the progress file straight to Drive through symlinks
model_dir = "kalenjin-translator" if args.direction == "kln-en" else "kalenjin-translator-en-kln"
progress = f"progress_{args.direction}.txt"
(DRIVE / model_dir).mkdir(exist_ok=True)
for link, target in ((Path(model_dir), DRIVE / model_dir), (Path("model/data") / progress, DRIVE / progress)):
    if link.is_symlink() or link.exists():
        link.unlink() if link.is_symlink() else shutil.rmtree(link) if link.is_dir() else link.unlink()
    link.symlink_to(target)
if args.reset and (DRIVE / progress).exists():
    (DRIVE / progress).unlink()

cmd = [sys.executable, "model/train_chunks.py", "--direction", args.direction,
       "--chunk", str(args.chunk), "--grad-accum", str(args.grad_accum)]
if args.cpu:
    cmd.append("--cpu")
# train_chunks.py unlinks a missing progress file only with --reset, which is handled above
sys.exit(subprocess.call(cmd))
