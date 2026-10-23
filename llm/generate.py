"""Generate text from a trained checkpoint.

    python generate.py --prompt "ROMEO:"
    python generate.py --prompt "ROMEO:" --tokens 500 --temperature 0.7 --top-k 40 --samples 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tinygpt import GPT, GPTConfig, load_tokenizer

ROOT = Path(__file__).resolve().parent


def load(ckpt_path: Path, device: str) -> tuple[GPT, object]:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ck["config"])).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, load_tokenizer(ck["tokenizer"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=ROOT / "checkpoints" / "shakespeare_char" / "best.pt")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--tokens", type=int, default=300, help="number of tokens to generate")
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = always the most likely token")
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    if a.seed is not None:
        torch.manual_seed(a.seed)
    model, tok = load(a.ckpt, a.device)
    start = torch.tensor([tok.encode(a.prompt)], device=a.device)
    for i in range(a.samples):
        out = model.generate(start, a.tokens, temperature=a.temperature, top_k=a.top_k, top_p=a.top_p)
        print(f"--- sample {i + 1} ---")
        print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
