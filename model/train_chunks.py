"""Train on the full dataset one chunk at a time, continuing from the last saved model.
Progress is stored in model/data/progress_<direction>.txt, so you can stop (Ctrl+C) and rerun to continue.

  python model/train_chunks.py --direction kln-en --chunk 2000
  python model/train_chunks.py --direction kln-en --chunk 2000 --reset   (start over)
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--direction", choices=["kln-en", "en-kln"], default="kln-en")
parser.add_argument("--chunk", type=int, default=2000, help="training rows per run")
parser.add_argument("--epochs", type=float, default=1, help="epochs per chunk")
parser.add_argument("--reset", action="store_true")
parser.add_argument("--cpu", action="store_true", help="train on CPU (slower, avoids MPS NaN)")
parser.add_argument("--grad-accum", type=int, default=4)
args = parser.parse_args()

progress = Path(f"model/data/progress_{args.direction}.txt")
if args.reset and progress.exists():
    progress.unlink()
offset = int(progress.read_text()) if progress.exists() else 0
total = len(pd.read_csv("model/data/train.csv"))

while offset < total:
    print(f"\n=== {args.direction}: rows {offset}-{min(offset + args.chunk, total)} of {total} ===", flush=True)
    cmd = [sys.executable, "model/train_translation.py", "--direction", args.direction,
           "--max-samples", str(args.chunk), "--offset", str(offset), "--epochs", str(args.epochs)]
    cmd += ["--grad-accum", str(args.grad_accum)]
    if args.cpu:
        cmd.append("--cpu")
    if offset > 0:
        cmd.append("--resume")
    if subprocess.run(cmd).returncode != 0:
        sys.exit("Chunk failed; fix the error and rerun to continue from the same chunk.")
    offset += args.chunk
    progress.write_text(str(offset))

print("All data seen. Test with: python model/translate.py --direction", args.direction)
