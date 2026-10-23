"""Download a corpus, train a tokenizer on it, and write train/validation token files.

    python prepare_data.py                              # Tiny Shakespeare, character tokenizer
    python prepare_data.py --tokenizer bpe --vocab 2048 # same text, BPE tokenizer
    python prepare_data.py --text my_corpus.txt --out data/mine

Output folder: input.txt, tokenizer.json, train.bin, val.bin, meta.json (source, licence,
checksum, split sizes - roadmap section 8: document every dataset).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tinygpt import BPETokenizer, CharTokenizer

ROOT = Path(__file__).resolve().parent

CORPORA = {
    "shakespeare": {
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "description": "Tiny Shakespeare: ~1.1 MB of concatenated Shakespeare plays (char-rnn dataset)",
        "license": "Public domain (works of William Shakespeare)",
    },
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="shakespeare", choices=sorted(CORPORA))
    ap.add_argument("--text", type=Path, help="use a local UTF-8 text file instead of downloading a corpus")
    ap.add_argument("--out", type=Path, help="output folder (default: data/<corpus>[_bpe])")
    ap.add_argument("--tokenizer", default="char", choices=["char", "bpe"])
    ap.add_argument("--vocab", type=int, default=2048, help="BPE vocabulary size")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    args = ap.parse_args()

    name = args.text.stem if args.text else args.corpus
    out = args.out or ROOT / "data" / (name + ("_bpe" if args.tokenizer == "bpe" else ""))
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "input.txt"

    if args.text:
        raw.write_text(args.text.read_text(encoding="utf-8"), encoding="utf-8")
        source = {"url": None, "description": f"local file {args.text}", "license": "document it here"}
    else:
        source = CORPORA[args.corpus]
        if not raw.exists():
            print(f"downloading {source['url']}")
            urllib.request.urlretrieve(source["url"], raw)
    text = raw.read_text(encoding="utf-8")
    print(f"corpus: {len(text):,} characters")

    if args.tokenizer == "char":
        tok = CharTokenizer.train(text)
    else:
        print(f"training BPE tokenizer to {args.vocab} tokens ...")
        tok = BPETokenizer.train(text, args.vocab, special_tokens=["<|endoftext|>"], verbose=True)
    tok.save(out / "tokenizer.json")

    ids = np.array(tok.encode(text), dtype=np.uint16)
    # split by position, not randomly: the validation text is a continuous passage the model
    # never trains on, so validation loss measures generalisation without leakage
    n_val = int(len(ids) * args.val_fraction)
    ids[:-n_val].tofile(out / "train.bin")
    ids[-n_val:].tofile(out / "val.bin")

    meta = {
        "source": source,
        "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "prepared_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tokenizer": tok.kind,
        "vocab_size": tok.vocab_size,
        "tokens_total": int(len(ids)),
        "tokens_train": int(len(ids) - n_val),
        "tokens_val": n_val,
        "split": f"last {args.val_fraction:.0%} of the text is validation",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"vocab {tok.vocab_size} | train {meta['tokens_train']:,} tokens | val {n_val:,} tokens -> {out}")


if __name__ == "__main__":
    main()
