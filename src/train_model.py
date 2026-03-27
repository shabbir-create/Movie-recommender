from __future__ import annotations

import argparse

import pandas as pd

from recommender import MovieRecommender


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and save movie recommender model")
    parser.add_argument(
        "--data",
        type=str,
        default="data/movies_tmdb_clean.csv",
        help="Path to movie dataset CSV",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/movie_recommender.joblib",
        help="Where to save the model artifact",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    movies_df = pd.read_csv(args.data)

    recommender = MovieRecommender()
    recommender.fit(movies_df)
    recommender.save(args.model_path)

    print(f"Model trained on {len(movies_df)} rows and saved to: {args.model_path}")


if __name__ == "__main__":
    main()
