from __future__ import annotations

import argparse

import pandas as pd

from recommender import MovieRecommender


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Content-based Movie Recommendation System"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/movies_sample.csv",
        help="Path to movie dataset CSV",
    )
    parser.add_argument(
        "--mode",
        choices=["similar", "personalized"],
        required=True,
        help="Recommendation mode",
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Movie title for similar recommendations",
    )
    parser.add_argument(
        "--liked",
        nargs="+",
        help="One or more liked movie titles for personalized recommendations",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of recommendations to return",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/movie_recommender.joblib",
        help="Path for saving/loading the trained model",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save trained model after fitting from CSV",
    )
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Load model from --model-path instead of training from CSV",
    )
    return parser


def print_recommendations(results: list, heading: str) -> None:
    print()
    print(heading)
    print("-" * len(heading))
    if not results:
        print("No recommendations found.")
        return

    for idx, rec in enumerate(results, start=1):
        year_part = f" ({rec.year})" if rec.year else ""
        print(f"{idx:>2}. {rec.title}{year_part}  score={rec.score:.4f}")


def build_or_load_recommender(args: argparse.Namespace) -> MovieRecommender:
    if args.load_model:
        return MovieRecommender.load(args.model_path)

    movies_df = pd.read_csv(args.data)
    recommender = MovieRecommender()
    recommender.fit(movies_df)

    if args.save_model:
        recommender.save(args.model_path)
        print(f"Saved model to: {args.model_path}")

    return recommender


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    recommender = build_or_load_recommender(args)

    if args.mode == "similar":
        if not args.title:
            parser.error("--title is required for mode=similar")
        recommendations = recommender.recommend_similar(args.title, top_n=args.top_n)
        print_recommendations(
            recommendations,
            heading=f"Top {args.top_n} movies similar to '{args.title}'",
        )
        return

    if not args.liked:
        parser.error("--liked is required for mode=personalized")
    recommendations = recommender.recommend_for_user(args.liked, top_n=args.top_n)
    print_recommendations(
        recommendations,
        heading=f"Top {args.top_n} personalized recommendations",
    )


if __name__ == "__main__":
    main()
