"""From-scratch BPE *training* (Sennrich et al., 1994).

The algorithm GPT-4o's vocabulary was learned with, in miniature:
count every adjacent symbol pair, merge the most frequent pair
everywhere, repeat. Deterministic; ties broken by first-seen order.

Usage:
    python3 python/bpe_train.py corpus.txt --merges 500 --out merges.json
"""

import argparse, json, re, sys
from collections import Counter


def train(corpus, num_merges, verbose=True):
    words = Counter(re.findall(r"\S+|\s+", corpus))
    splits = {w: list(w) for w in words}
    merges = []
    for m in range(num_merges):
        pair_counts = Counter()
        for w, freq in words.items():
            s = splits[w]
            for a, b in zip(s, s[1:]):
                pair_counts[(a, b)] += freq
        if not pair_counts:
            break
        (a, b), _ = pair_counts.most_common(1)[0]
        merges.append([a, b])
        merged = a + b
        for w in words:
            s, out = splits[w], []
            i = 0
            while i < len(s):
                if i < len(s) - 1 and s[i] == a and s[i + 1] == b:
                    out.append(merged); i += 2
                else:
                    out.append(s[i]); i += 1
            splits[w] = out
        if verbose and m % 100 == 0:
            print(f"merge {m}: {a!r} + {b!r} -> {merged!r}", flush=True)
    vocab = sorted({sym for s in splits.values() for sym in s})
    return {"merges": merges, "vocab": vocab}


def encode(text, merges):
    rank = {(a, b): i for i, (a, b) in enumerate(merges)}
    out = []
    for w in re.findall(r"\S+|\s+", text):
        parts = list(w)
        while True:
            best, best_i = None, -1
            for i, (a, b) in enumerate(zip(parts, parts[1:])):
                r = rank.get((a, b))
                if r is not None and (best is None or r < best):
                    best, best_i = r, i
            if best is None:
                break
            parts = parts[:best_i] + [parts[best_i] + parts[best_i + 1]] + parts[best_i + 2:]
        out.extend(parts)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--merges", type=int, default=500)
    ap.add_argument("--out", default="merges.json")
    args = ap.parse_args()
    corpus = open(args.corpus, encoding="utf-8").read()
    result = train(corpus, args.merges)
    json.dump(result, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"vocab size: {len(result['vocab'])}, merges: {len(result['merges'])} -> {args.out}")
    demo = "the quick brown fox"
    print(f"demo: {demo!r} -> {encode(demo, result['merges'])}")
