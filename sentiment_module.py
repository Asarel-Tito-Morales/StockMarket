from __future__ import annotations

from typing import Any, Dict

import pandas as pd

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except Exception:
    SentimentIntensityAnalyzer = None


class SentimentScorer:
    def __init__(self, config: Dict[str, Any], logger: Any) -> None:
        self.config = config
        self.logger = logger
        self.model_name = config.get("SENTIMENT_MODEL", "vader").lower()
        self.positive_threshold = float(config.get("SENTIMENT_POSITIVE_THRESHOLD", 0.05))
        self.negative_threshold = float(config.get("SENTIMENT_NEGATIVE_THRESHOLD", -0.05))

        if self.model_name != "vader":
            raise ValueError(f"Unsupported SENTIMENT_MODEL: {self.model_name}. Use 'vader'.")

        if SentimentIntensityAnalyzer is None:
            raise ImportError(
                "nltk.sentiment.vader is not available. Install nltk and download vader_lexicon."
            )
        self.analyzer = SentimentIntensityAnalyzer()

    def score(self, news_df: pd.DataFrame) -> pd.DataFrame:
        if news_df.empty:
            return news_df.copy()

        df = news_df.copy()
        text = (
            df["title"].fillna("") + ". " +
            df["description"].fillna("") + ". " +
            df["content"].fillna("")
        ).str.strip()

        scores = text.apply(self.analyzer.polarity_scores)
        df["sentiment_neg"] = scores.apply(lambda x: x["neg"])
        df["sentiment_neu"] = scores.apply(lambda x: x["neu"])
        df["sentiment_pos"] = scores.apply(lambda x: x["pos"])
        df["sentiment_compound"] = scores.apply(lambda x: x["compound"])

        def label(compound: float) -> str:
            if compound >= self.positive_threshold:
                return "Positive"
            if compound <= self.negative_threshold:
                return "Negative"
            return "Neutral"

        df["sentiment_label"] = df["sentiment_compound"].apply(label)
        return df
