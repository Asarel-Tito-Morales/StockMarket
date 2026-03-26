from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from utils import get_alignment_timestamp


class DataMerger:
    def __init__(self, config: Dict[str, Any], logger: Any) -> None:
        self.config = config
        self.logger = logger
        self.interval_minutes = int(config.get("RESAMPLE_INTERVAL_MINUTES", 30))
        self.matching_rule = config.get("MATCHING_RULE", "floor").lower()

    def align_news_timestamps(self, news_df: pd.DataFrame) -> pd.DataFrame:
        if news_df.empty:
            return news_df.copy()

        df = news_df.copy()
        df["raw_news_timestamp"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        df["aligned_timestamp"] = df["raw_news_timestamp"].apply(
            lambda ts: get_alignment_timestamp(ts, self.interval_minutes, self.matching_rule)
        )
        return df

    def resample_stock(self, stock_df: pd.DataFrame) -> pd.DataFrame:
        if stock_df.empty:
            return stock_df.copy()

        frames = []
        freq = f"{self.interval_minutes}min"
        for ticker, part in stock_df.groupby("ticker"):
            part = part.copy().sort_values("timestamp")
            part = part.set_index("timestamp")

            agg = part.resample(freq).agg(
                {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Adj Close": "last",
                    "Volume": "sum",
                }
            )
            agg["ticker"] = ticker
            agg = agg.dropna(subset=["Open", "High", "Low", "Close"], how="any").reset_index()
            frames.append(agg)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def aggregate_news(self, aligned_news_df: pd.DataFrame) -> pd.DataFrame:
        if aligned_news_df.empty:
            return pd.DataFrame(columns=[
                "ticker", "aligned_timestamp", "news_count", "sentiment_compound_mean",
                "sentiment_compound_sum", "positive_count", "neutral_count", "negative_count"
            ])

        df = aligned_news_df.copy()
        df["is_positive"] = (df["sentiment_label"] == "Positive").astype(int)
        df["is_neutral"] = (df["sentiment_label"] == "Neutral").astype(int)
        df["is_negative"] = (df["sentiment_label"] == "Negative").astype(int)

        grouped = (
            df.groupby(["ticker", "aligned_timestamp"], as_index=False)
              .agg(
                  news_count=("title", "count"),
                  sentiment_compound_mean=("sentiment_compound", "mean"),
                  sentiment_compound_sum=("sentiment_compound", "sum"),
                  positive_count=("is_positive", "sum"),
                  neutral_count=("is_neutral", "sum"),
                  negative_count=("is_negative", "sum"),
              )
        )
        return grouped

    def merge(self, news_df: pd.DataFrame, stock_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        aligned_news = self.align_news_timestamps(news_df)
        stock_30 = self.resample_stock(stock_df)
        news_agg = self.aggregate_news(aligned_news)

        if stock_30.empty:
            return aligned_news, news_agg, stock_30.copy()

        merged = stock_30.merge(
            news_agg,
            how="left",
            left_on=["ticker", "timestamp"],
            right_on=["ticker", "aligned_timestamp"],
        )

        for col in ["news_count", "sentiment_compound_mean", "sentiment_compound_sum", "positive_count", "neutral_count", "negative_count"]:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)

        merged["return_1"] = merged.groupby("ticker")["Close"].pct_change()
        merged["rolling_volatility_3"] = (
            merged.groupby("ticker")["return_1"].rolling(3).std().reset_index(level=0, drop=True)
        )
        merged["price_range_pct"] = (merged["High"] - merged["Low"]) / merged["Close"].replace(0, np.nan)
        merged["volume_change"] = merged.groupby("ticker")["Volume"].pct_change()
        merged["sentiment_balance"] = merged["positive_count"] - merged["negative_count"]

        merged["rolling_volatility_3"] = merged["rolling_volatility_3"].fillna(0)
        merged["price_range_pct"] = merged["price_range_pct"].replace([np.inf, -np.inf], np.nan).fillna(0)
        merged["volume_change"] = merged["volume_change"].replace([np.inf, -np.inf], np.nan).fillna(0)
        merged["return_1"] = merged["return_1"].replace([np.inf, -np.inf], np.nan).fillna(0)

        return aligned_news, news_agg, merged

    def create_target(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        if merged_df.empty:
            return merged_df.copy()

        df = merged_df.copy().sort_values(["ticker", "timestamp"])
        lookahead = int(self.config.get("LOOKAHEAD_PERIODS", 2))
        flat_threshold = float(self.config.get("TARGET_FLAT_THRESHOLD", 0.0025))

        df["future_close"] = df.groupby("ticker")["Close"].shift(-lookahead)
        df["future_return"] = (df["future_close"] - df["Close"]) / df["Close"].replace(0, np.nan)
        df["future_return"] = df["future_return"].replace([np.inf, -np.inf], np.nan)

        def direction_label(x: float) -> str | None:
            if pd.isna(x):
                return None
            if x > flat_threshold:
                return "UP"
            if x < -flat_threshold:
                return "DOWN"
            return "FLAT"

        df["target_direction"] = df["future_return"].apply(direction_label)
        return df
