from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd
import requests


class NewsCollectorError(Exception):
    pass


@dataclass
class BaseNewsCollector:
    config: Dict[str, Any]
    logger: Any

    def fetch(self) -> pd.DataFrame:
        raise NotImplementedError


@dataclass
class NewsApiCollector(BaseNewsCollector):
    def fetch(self) -> pd.DataFrame:
        api_key = self.config.get("NEWS_API_KEY", "")
        if not api_key or api_key == "REPLACE_ME":
            raise NewsCollectorError("NEWS_API_KEY is missing or still set to REPLACE_ME.")

        tickers = self.config["TICKERS"]
        page_size = int(self.config.get("NEWS_PAGE_SIZE", 50))
        language = self.config.get("NEWS_QUERY_LANGUAGE", "en")
        lookback_hours = int(self.config.get("NEWS_LOOKBACK_HOURS", 48))
        tz_name = self.config["TIMEZONE"]

        rows: List[Dict[str, Any]] = []
        session = requests.Session()

        for ticker in tickers:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": ticker,
                "language": language,
                "pageSize": page_size,
                "sortBy": "publishedAt",
                "from": (pd.Timestamp.now(tz=tz_name) - pd.Timedelta(hours=lookback_hours)).isoformat(),
                "apiKey": api_key,
            }
            response = session.get(url, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()

            for article in payload.get("articles", []):
                rows.append(
                    {
                        "ticker": ticker,
                        "source_name": (article.get("source") or {}).get("name"),
                        "author": article.get("author"),
                        "title": article.get("title"),
                        "description": article.get("description"),
                        "content": article.get("content"),
                        "url": article.get("url"),
                        "published_at": article.get("publishedAt"),
                        "raw_payload_provider": "newsapi",
                    }
                )

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        df = df.dropna(subset=["published_at", "title"]).drop_duplicates(subset=["ticker", "url"])
        return df


@dataclass
class AlphaVantageNewsCollector(BaseNewsCollector):
    def fetch(self) -> pd.DataFrame:
        api_key = self.config.get("NEWS_API_KEY", "")
        if not api_key or api_key == "REPLACE_ME":
            raise NewsCollectorError("NEWS_API_KEY is missing or still set to REPLACE_ME.")

        tickers = ",".join(self.config["TICKERS"])
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": tickers,
            "apikey": api_key,
            "limit": 200,
        }
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        rows: List[Dict[str, Any]] = []

        for article in payload.get("feed", []):
            published_at = pd.to_datetime(article.get("time_published"), utc=True, errors="coerce")
            for ts in article.get("ticker_sentiment", []):
                rows.append(
                    {
                        "ticker": ts.get("ticker"),
                        "source_name": article.get("source"),
                        "author": article.get("authors"),
                        "title": article.get("title"),
                        "description": article.get("summary"),
                        "content": article.get("summary"),
                        "url": article.get("url"),
                        "published_at": published_at,
                        "raw_payload_provider": "alphavantage",
                    }
                )

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.dropna(subset=["published_at", "title"]).drop_duplicates(subset=["ticker", "url"])
        return df


def get_news_collector(config: Dict[str, Any], logger: Any) -> BaseNewsCollector:
    source = str(config.get("NEWS_SOURCE", "newsapi")).lower().strip()

    if source == "newsapi":
        return NewsApiCollector(config=config, logger=logger)
    if source == "alphavantage":
        return AlphaVantageNewsCollector(config=config, logger=logger)

    raise NewsCollectorError(
        f"Unsupported NEWS_SOURCE '{source}'. Supported values: newsapi, alphavantage."
    )
