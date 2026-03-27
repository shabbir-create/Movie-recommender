from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_CONTENT_COLUMNS = ["genres", "keywords", "cast", "director", "overview"]


@dataclass
class RecommendationResult:
    title: str
    score: float
    year: int | None = None


class MovieRecommender:
    def __init__(
        self,
        title_column: str = "title",
        content_columns: list[str] | None = None,
    ) -> None:
        self.title_column = title_column
        self.content_columns = content_columns or DEFAULT_CONTENT_COLUMNS
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.movies_df: pd.DataFrame | None = None
        self.tfidf_matrix = None
        self.title_to_idx: dict[str, int] = {}

    def fit(self, movies_df: pd.DataFrame) -> None:
        if self.title_column not in movies_df.columns:
            raise ValueError(
                f"Dataset must contain a '{self.title_column}' column."
            )

        df = movies_df.copy()
        df = df.dropna(subset=[self.title_column]).drop_duplicates(
            subset=[self.title_column]
        )

        available_content_cols = [
            col for col in self.content_columns if col in df.columns
        ]
        if not available_content_cols:
            raise ValueError(
                "No content columns found. Provide at least one of: "
                f"{', '.join(self.content_columns)}"
            )

        for col in available_content_cols:
            df[col] = df[col].fillna("").astype(str).str.lower()

        df[self.title_column] = df[self.title_column].astype(str).str.strip()
        df["combined_features"] = df[available_content_cols].agg(" ".join, axis=1)
        df = df.reset_index(drop=True)

        self.movies_df = df
        self.tfidf_matrix = self.vectorizer.fit_transform(df["combined_features"])
        self.title_to_idx = {
            title.lower(): idx for idx, title in enumerate(df[self.title_column])
        }

    def save(self, model_path: str) -> None:
        self._ensure_fitted()
        target = Path(model_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "title_column": self.title_column,
            "content_columns": self.content_columns,
            "vectorizer": self.vectorizer,
            "movies_df": self.movies_df,
            "tfidf_matrix": self.tfidf_matrix,
            "title_to_idx": self.title_to_idx,
        }
        joblib.dump(artifact, target)

    @classmethod
    def load(cls, model_path: str) -> "MovieRecommender":
        source = Path(model_path)
        if not source.exists():
            raise FileNotFoundError(f"Saved model not found at: {source}")

        artifact = joblib.load(source)
        required_keys = {
            "title_column",
            "content_columns",
            "vectorizer",
            "movies_df",
            "tfidf_matrix",
            "title_to_idx",
        }
        missing = required_keys - set(artifact.keys())
        if missing:
            raise ValueError(
                "Invalid model artifact. Missing keys: " + ", ".join(sorted(missing))
            )

        recommender = cls(
            title_column=artifact["title_column"],
            content_columns=artifact["content_columns"],
        )
        recommender.vectorizer = artifact["vectorizer"]
        recommender.movies_df = artifact["movies_df"]
        recommender.tfidf_matrix = artifact["tfidf_matrix"]
        recommender.title_to_idx = artifact["title_to_idx"]
        return recommender

    def recommend_similar(
        self,
        title: str,
        top_n: int = 10,
    ) -> list[RecommendationResult]:
        self._ensure_fitted()
        normalized = title.strip().lower()
        if normalized not in self.title_to_idx:
            raise ValueError(f"Movie '{title}' not found in dataset.")

        idx = self.title_to_idx[normalized]
        cosine_scores = cosine_similarity(
            self.tfidf_matrix[idx], self.tfidf_matrix
        ).flatten()

        ranked_indices = cosine_scores.argsort()[::-1]
        recommendations: list[RecommendationResult] = []
        for rec_idx in ranked_indices:
            if rec_idx == idx:
                continue
            row = self.movies_df.iloc[rec_idx]
            recommendations.append(
                RecommendationResult(
                    title=row[self.title_column],
                    score=float(cosine_scores[rec_idx]),
                    year=self._safe_year(row),
                )
            )
            if len(recommendations) >= top_n:
                break
        return recommendations

    def recommend_for_user(
        self,
        liked_titles: Iterable[str],
        top_n: int = 10,
    ) -> list[RecommendationResult]:
        self._ensure_fitted()
        liked_indices = []
        for title in liked_titles:
            normalized = title.strip().lower()
            idx = self.title_to_idx.get(normalized)
            if idx is not None:
                liked_indices.append(idx)

        if not liked_indices:
            raise ValueError("None of the liked titles were found in the dataset.")

        user_profile_vector = np.asarray(
            self.tfidf_matrix[liked_indices].mean(axis=0)
        )
        cosine_scores = cosine_similarity(
            user_profile_vector, self.tfidf_matrix
        ).flatten()

        ranked_indices = cosine_scores.argsort()[::-1]
        liked_set = set(liked_indices)
        recommendations: list[RecommendationResult] = []
        for rec_idx in ranked_indices:
            if rec_idx in liked_set:
                continue
            row = self.movies_df.iloc[rec_idx]
            recommendations.append(
                RecommendationResult(
                    title=row[self.title_column],
                    score=float(cosine_scores[rec_idx]),
                    year=self._safe_year(row),
                )
            )
            if len(recommendations) >= top_n:
                break
        return recommendations

    def available_titles(self) -> list[str]:
        self._ensure_fitted()
        return sorted(self.movies_df[self.title_column].tolist())

    def _safe_year(self, row: pd.Series) -> int | None:
        if "year" not in row:
            return None
        try:
            return int(row["year"])
        except (TypeError, ValueError):
            return None

    def _ensure_fitted(self) -> None:
        if self.movies_df is None or self.tfidf_matrix is None:
            raise RuntimeError("Call fit() before requesting recommendations.")
