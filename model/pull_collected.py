"""Pull submissions from the private dataset repo and merge them into one CSV per kind.

    python model/pull_collected.py --repo ezkemboi/kalenjin-dataset

Needs git access to the repo (gh auth login, or an SSH key). Writes model/data/collected/<kind>.csv (raw, keeps IP - private, for spam checks) and
model/data/collected/clean/<kind>.csv (safe for training: no IP / user agent, emails and
phone numbers scrubbed, IP replaced by a salted hash `contributor_id`).

Set PII_SALT (any secret string) so contributor ids cannot be reversed.
"""
import argparse
import glob
import hashlib
import os
import re
import subprocess

import pandas as pd

CLONE_DIR = "model/data/dataset_repo"
OUT_DIR = "model/data/collected"
CLEAN_DIR = f"{OUT_DIR}/clean"
PII_COLS = ["ip", "user_agent"]
SALT = os.environ.get("PII_SALT", "")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")


def scrub(text):
    return PHONE.sub("[phone]", EMAIL.sub("[email]", str(text)))


def clean(df):
    df = df.copy()
    if "ip" in df.columns:
        df["contributor_id"] = df["ip"].map(
            lambda ip: hashlib.sha256((SALT + str(ip)).encode()).hexdigest()[:10] if ip else ""
        )
    df = df.drop(columns=[c for c in PII_COLS if c in df.columns])
    for col in df.select_dtypes(include="object").columns:
        if col != "contributor_id":
            df[col] = df[col].map(scrub)
    return df


if not SALT:
    print("Warning: PII_SALT not set; contributor ids can be guessed. Set it for real use.")

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="ezkemboi/kalenjin-dataset")
args = ap.parse_args()

if os.path.isdir(os.path.join(CLONE_DIR, ".git")):
    subprocess.run(["git", "-C", CLONE_DIR, "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", f"https://github.com/{args.repo}.git", CLONE_DIR], check=True)

os.makedirs(CLEAN_DIR, exist_ok=True)
for kind in ("feedback", "word_votes", "qa_answers"):
    files = glob.glob(f"{CLONE_DIR}/data/{kind}/**/*.csv", recursive=True)
    if not files:
        print(f"{kind}: no files")
        continue
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True).fillna("")
    df = df.drop_duplicates().sort_values("time")
    df.to_csv(f"{OUT_DIR}/{kind}.csv", index=False)
    print(f"{kind}: {len(df)} rows -> {OUT_DIR}/{kind}.csv")
    clean(df).to_csv(f"{CLEAN_DIR}/{kind}.csv", index=False)
    print(f"  clean (no IP/user agent) -> {CLEAN_DIR}/{kind}.csv")
