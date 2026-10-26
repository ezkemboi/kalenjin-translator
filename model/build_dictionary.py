"""Learn a Kalenjin <-> English word dictionary from the aligned verses (IBM Model 1, both directions).

  python model/build_dictionary.py            -> model/data/dictionary.csv
  python model/build_dictionary.py --lookup bik tugul sait
"""
import argparse
import csv
import re
from collections import Counter, defaultdict
from math import sqrt

import pandas as pd

OUT = "model/data/dictionary.csv"
STOP_EN = {"the", "of", "and", "to", "a", "in", "that", "he", "is", "shall", "for", "it", "unto", "his", "i", "thou", "thy", "be", "with", "not", "they", "them", "ye", "my", "all"}


def tokens(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)*", str(text).replace("\u2019", "'").lower())


def model1(pairs, iters=6):
    """t[src][tgt] = P(tgt | src); a NULL source word absorbs unaligned target words."""
    t = defaultdict(lambda: defaultdict(lambda: 1.0))
    for _ in range(iters):
        count, total = defaultdict(lambda: defaultdict(float)), defaultdict(float)
        for src, tgt in pairs:
            src = ["<null>"] + src
            for y in tgt:
                z = sum(t[x][y] for x in src)
                for x in src:
                    c = t[x][y] / z
                    count[x][y] += c
                    total[x] += c
        t = defaultdict(lambda: defaultdict(lambda: 1e-6),
                        {x: defaultdict(lambda: 1e-6, {y: c / total[x] for y, c in ys.items()}) for x, ys in count.items()})
    return t


def build():
    df = pd.read_csv("model/data/train.csv").dropna(subset=["text_kln", "text_en"])
    kln = [tokens(x) for x in df.text_kln]
    en = [tokens(x) for x in df.text_en]
    keep = [i for i in range(len(df)) if kln[i] and en[i]]
    kln, en, verses = [kln[i] for i in keep], [en[i] for i in keep], df.text_en.iloc[keep].tolist()
    kln_text = df.text_kln.iloc[keep].tolist()

    t_ke = model1(list(zip(kln, en)))
    t_ek = model1(list(zip(en, kln)))

    freq = Counter(w for s in kln for w in set(s))
    cooc = Counter((k, e) for ks, es in zip(kln, en) for k in set(ks) for e in set(es))
    example = {}
    for i, ks in enumerate(kln):
        if len(ks) <= 14:
            for k in ks:
                example.setdefault(k, i)

    rows = []
    for k, n in freq.items():
        if n < 3:
            continue
        cands = []
        for e, p in t_ke[k].items():
            if e in STOP_EN and p < 0.5:
                continue
            c = cooc[(k, e)]
            if c < 2:
                continue
            score = sqrt(p * t_ek[e][k])
            cands.append((score, e, c))
        for score, e, c in sorted(cands, reverse=True)[:3]:
            if score >= 0.05:
                i = example.get(k)
                rows.append([k, e, round(score, 3), c, n, kln_text[i] if i is not None else "", verses[i] if i is not None else ""])
    rows.sort(key=lambda r: (-r[4], -r[2]))
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kalenjin", "english", "score", "pair_count", "kalenjin_count", "example_kln", "example_en"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} entries for {len({r[0] for r in rows})} Kalenjin words to {OUT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lookup", nargs="*")
    a = p.parse_args()
    if a.lookup:
        d = pd.read_csv(OUT)
        for w in a.lookup:
            hit = d[(d.kalenjin == w.lower()) | (d.english == w.lower())]
            print(hit[["kalenjin", "english", "score", "pair_count"]].to_string(index=False) if len(hit) else f"{w}: not found")
    else:
        build()
