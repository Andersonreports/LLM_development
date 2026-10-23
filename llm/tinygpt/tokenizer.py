"""Tokenizers, written from scratch: text <-> list of integer token IDs.

CharTokenizer  one token per character. Simplest possible; good for first experiments.
BPETokenizer   byte-level Byte Pair Encoding (the method GPT models use). Starts from the
               256 possible bytes and repeatedly merges the most frequent adjacent pair into
               a new token, so common words/sub-words become single tokens. Any text can be
               encoded (unknown characters fall back to their bytes).
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# Split text into word-like chunks before BPE, so merges never cross word boundaries
# (" the" can become one token, but "e t" across two words cannot). Simplified GPT-2 pattern.
_CHUNK_RE = re.compile(r"""'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d{1,3}| ?[^\s\w]+|\s+(?!\S)|\s+""")


class CharTokenizer:
    kind = "char"

    def __init__(self, chars: list[str]):
        self.itos = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def train(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        missing = set(text) - self.stoi.keys()
        if missing:
            raise ValueError(f"characters not in the vocabulary: {''.join(sorted(missing))!r}")
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"type": self.kind, "chars": self.itos}, ensure_ascii=False), encoding="utf-8")


class BPETokenizer:
    kind = "bpe"

    def __init__(self, merges: list[tuple[int, int]], special_tokens: list[str] | None = None):
        self.merges = [tuple(m) for m in merges]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}  # lower rank = merged earlier
        # vocab[id] = the bytes that token stands for
        self.vocab: list[bytes] = [bytes([b]) for b in range(256)]
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self.special = {tok: len(self.vocab) + i for i, tok in enumerate(special_tokens or [])}
        self._special_re = (re.compile("(" + "|".join(map(re.escape, self.special)) + ")")
                            if self.special else None)
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special)

    @classmethod
    def train(cls, text: str, vocab_size: int, special_tokens: list[str] | None = None,
              verbose: bool = False) -> "BPETokenizer":
        n_merges = vocab_size - 256 - len(special_tokens or [])
        if n_merges < 0:
            raise ValueError("vocab_size must be at least 256 + number of special tokens")
        # count each distinct chunk once, with its frequency - much faster than the raw text
        chunk_counts = Counter(_CHUNK_RE.findall(text))
        words = [list(w.encode("utf-8")) for w in chunk_counts]
        freqs = list(chunk_counts.values())

        pair_counts: Counter = Counter()
        where: dict[tuple[int, int], set[int]] = defaultdict(set)  # pair -> words containing it
        for wi, w in enumerate(words):
            for pair in zip(w, w[1:]):
                pair_counts[pair] += freqs[wi]
                where[pair].add(wi)

        merges: list[tuple[int, int]] = []
        for step in range(n_merges):
            if not pair_counts:
                break  # nothing left to merge (tiny corpus)
            best, count = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if count < 2:
                break
            new_id = 256 + len(merges)
            merges.append(best)
            # update only the words that contain the merged pair
            for wi in list(where[best]):
                w, f = words[wi], freqs[wi]
                for pair in zip(w, w[1:]):
                    pair_counts[pair] -= f
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    where[pair].discard(wi)
                words[wi] = w = _merge(w, best, new_id)
                for pair in zip(w, w[1:]):
                    pair_counts[pair] += f
                    where[pair].add(wi)
            where.pop(best, None)
            if verbose and (step + 1) % 200 == 0:
                print(f"  merge {step + 1}/{n_merges}: {best} -> {new_id} (seen {count}x)")
        return cls(merges, special_tokens)

    def _encode_chunk(self, chunk: str) -> list[int]:
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # apply the earliest-learned merge available, exactly as during training
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            ids = _merge(ids, pair, 256 + self.ranks[pair])
        if len(self._cache) < 100_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        parts = self._special_re.split(text) if self._special_re else [text]
        for part in parts:
            if part in self.special:
                out.append(self.special[part])
            elif part:
                for chunk in _CHUNK_RE.findall(part):
                    out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids: list[int]) -> str:
        inv_special = {i: t for t, i in self.special.items()}
        data = b"".join(inv_special[i].encode("utf-8") if i in inv_special else self.vocab[i] for i in ids)
        # a generated sequence may stop in the middle of a multi-byte character
        return data.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"type": self.kind, "merges": self.merges,
                                          "special_tokens": list(self.special)}), encoding="utf-8")


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def load_tokenizer(path: str | Path) -> CharTokenizer | BPETokenizer:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d["type"] == "char":
        return CharTokenizer(d["chars"])
    if d["type"] == "bpe":
        return BPETokenizer(d["merges"], d.get("special_tokens"))
    raise ValueError(f"unknown tokenizer type {d['type']!r}")
