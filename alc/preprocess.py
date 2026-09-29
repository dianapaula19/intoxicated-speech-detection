"""Extract MFCC features from the ALC corpus.

    python -m alc.preprocess ALC data/features.npz
"""

import argparse

from .dataset import build
from .features import FeatureConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corpus", help="ALC root folder")
    parser.add_argument("output", help="output .npz file")
    parser.add_argument("--sample-rate", type=int, default=FeatureConfig.sample_rate)
    parser.add_argument("--n-mfcc", type=int, default=FeatureConfig.n_mfcc)
    parser.add_argument("--n-frames", type=int, default=FeatureConfig.n_frames)
    parser.add_argument("--no-deltas", action="store_true")
    args = parser.parse_args(argv)

    config = FeatureConfig(sample_rate=args.sample_rate, n_mfcc=args.n_mfcc,
                           n_frames=args.n_frames, deltas=not args.no_deltas)
    build(args.corpus, args.output, config)


if __name__ == "__main__":
    main()
