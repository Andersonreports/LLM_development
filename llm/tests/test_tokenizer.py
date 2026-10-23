from __future__ import annotations

import pytest

from tinygpt import BPETokenizer, CharTokenizer, load_tokenizer

TEXT = ("The proband carries a heterozygous frameshift variant. The variant is absent from gnomAD. "
        "The father and the mother were tested; the variant is de novo. ") * 20


def test_char_round_trip_and_save(tmp_path):
    tok = CharTokenizer.train(TEXT)
    ids = tok.encode("The variant")
    assert tok.decode(ids) == "The variant"
    tok.save(tmp_path / "t.json")
    assert load_tokenizer(tmp_path / "t.json").encode("The variant") == ids
    with pytest.raises(ValueError):
        tok.encode("Z!")                        # character never seen in training


def test_bpe_round_trip_including_unseen_unicode():
    tok = BPETokenizer.train(TEXT, vocab_size=300)
    for s in [TEXT[:200], "Café – naïve 🧬 c.664del p.Arg222fs", "", "   spaces   "]:
        assert tok.decode(tok.encode(s)) == s


def test_bpe_compresses_and_respects_vocab_size():
    tok = BPETokenizer.train(TEXT, vocab_size=320, special_tokens=["<|endoftext|>"])
    assert tok.vocab_size <= 320
    ids = tok.encode(TEXT)
    assert len(ids) < len(TEXT.encode("utf-8")) / 2   # merges make text much shorter than raw bytes
    assert max(ids) < tok.vocab_size


def test_bpe_special_tokens_and_save(tmp_path):
    tok = BPETokenizer.train(TEXT, vocab_size=300, special_tokens=["<|endoftext|>"])
    ids = tok.encode("de novo<|endoftext|>The")
    assert tok.special["<|endoftext|>"] in ids
    assert tok.decode(ids) == "de novo<|endoftext|>The"
    tok.save(tmp_path / "b.json")
    tok2 = load_tokenizer(tmp_path / "b.json")
    assert tok2.encode(TEXT[:300]) == tok.encode(TEXT[:300])
    assert tok2.vocab_size == tok.vocab_size
