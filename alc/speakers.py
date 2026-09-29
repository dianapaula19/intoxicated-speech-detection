"""Summarise ALC speakers from the annotation files (one row per speaker).

    python -m alc.speakers ALC annotation_analysis.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from .annotations import FIELDS, read_labels


def speaker_table(root):
    rows = []
    for annot in sorted(Path(root).rglob("*_h_00_annot.json")):
        labels = read_labels(annot)
        rows.append({field: labels.get(field) for field in FIELDS})
    df = pd.DataFrame(rows, columns=list(FIELDS))
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    for column in ("aak", "bak"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    # The first recording of each speaker is kept; the columns describe the
    # speaker, except alc/aak/bak, which describe that first session.
    return df.drop_duplicates(subset=["spn"], keep="first").reset_index(drop=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corpus", help="ALC root folder")
    parser.add_argument("output", help="output CSV")
    args = parser.parse_args(argv)

    df = speaker_table(args.corpus)
    print(df.describe(include="all"))
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} speakers to {args.output}")


if __name__ == "__main__":
    main()
