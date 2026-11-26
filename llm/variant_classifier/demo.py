"""End-to-end demo on synthetic data:

    python -m variant_classifier.demo

Trains the ordinal model twice - once with the honest gene-grouped split and once with a
careless random split - so you can see the random split's numbers are inflated by leakage.
Then prints feature importance. No real data, nothing clinical."""
from __future__ import annotations

from .synth import make
from .train import cross_val, permutation_importance_qwk, _report


def main() -> None:
    df = make()
    print(f"synthetic dataset: {len(df):,} variants across {df['gene'].nunique()} genes")
    print(f"tier counts (B,LB,VUS,LP,P): {df['tier'].value_counts().sort_index().to_dict()}")

    print("\n################ HONEST: gene-grouped split ################")
    honest = cross_val(df, split="gene")
    _report(honest)

    print("\n################ LEAKY: random split (do NOT trust) ################")
    leaky = cross_val(df, split="random")
    _report(leaky)

    print(f"\n>>> leakage check: random-split QWK {leaky['model']['qwk']:.3f} is inflated vs "
          f"the honest gene-split QWK {honest['model']['qwk']:.3f}.")

    print("\nfeature importance (QWK drop when shuffled, gene signal included):")
    for feat, drop in permutation_importance_qwk(df):
        print(f"  {feat:22} {drop:+.4f}")


if __name__ == "__main__":
    main()
