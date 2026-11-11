"""Prepare the SMILES dataset: python scripts/prepare_data.py [--source URL|path]."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mol.data import DEFAULT_URL, download, prepare


def main():
    p = argparse.ArgumentParser(description="Download and prepare SMILES training data")
    p.add_argument("--source", default=DEFAULT_URL, help="URL or local .csv/.txt of SMILES")
    p.add_argument("--raw_dir", default="data/raw")
    p.add_argument("--out_dir", default="data/processed")
    p.add_argument("--max_len", type=int, default=100)
    p.add_argument("--n_bins", type=int, default=10)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1337)
    a = p.parse_args()

    source = a.source
    if source.startswith("http://") or source.startswith("https://"):
        name = os.path.basename(source.split("?")[0]) or "dataset.csv"
        source = download(source, os.path.join(a.raw_dir, name))
    prepare(source, a.out_dir, a.max_len, a.n_bins, a.limit, a.val_frac, a.seed)


if __name__ == "__main__":
    main()
