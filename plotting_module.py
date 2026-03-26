from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import plotly.graph_objects as go


class Plotter:
    def __init__(self, config: Dict[str, Any], logger: Any) -> None:
        self.config = config
        self.logger = logger
        self.output_dir = Path(config["OUTPUT_DIR"]).resolve()

    def plot_price_with_sentiment(self, aligned_news_df: pd.DataFrame, merged_df: pd.DataFrame) -> None:
        if merged_df.empty:
            self.logger.warning("Merged data empty; skipping overlay plot.")
            return

        for ticker, price_df in merged_df.groupby("ticker"):
            fig = go.Figure()
            price_df = price_df.sort_values("timestamp")

            fig.add_trace(
                go.Scatter(
                    x=price_df["timestamp"],
                    y=price_df["Close"],
                    mode="lines",
                    name=f"{ticker} Close",
                )
            )

            news_ticker = aligned_news_df[aligned_news_df["ticker"] == ticker].copy()
            if not news_ticker.empty:
                merged_news_price = news_ticker.merge(
                    price_df[["timestamp", "Close"]],
                    how="left",
                    left_on="aligned_timestamp",
                    right_on="timestamp"
                )

                for label in ["Positive", "Neutral", "Negative"]:
                    part = merged_news_price[merged_news_price["sentiment_label"] == label]
                    if part.empty:
                        continue
                    fig.add_trace(
                        go.Scatter(
                            x=part["aligned_timestamp"],
                            y=part["Close"],
                            mode="markers",
                            name=f"{label} News",
                            text=part["title"],
                            hovertemplate="Time=%{x}<br>Price=%{y}<br>Headline=%{text}<extra></extra>",
                        )
                    )

            fig.update_layout(
                title=f"{ticker} Price with Sentiment Events",
                xaxis_title="Time",
                yaxis_title="Price",
                legend_title="Series",
                template="plotly_white",
            )

            out_path = self.output_dir / f"{ticker.lower()}_price_sentiment.html"
            fig.write_html(out_path)
            self.logger.info("Saved overlay chart to %s", out_path)

    def plot_candlestick(self, stock_df: pd.DataFrame) -> None:
        if stock_df.empty:
            self.logger.warning("Stock data empty; skipping candlestick chart.")
            return

        for ticker, part in stock_df.groupby("ticker"):
            part = part.sort_values("timestamp")
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=part["timestamp"],
                        open=part["Open"],
                        high=part["High"],
                        low=part["Low"],
                        close=part["Close"],
                        name=ticker,
                    )
                ]
            )
            fig.update_layout(
                title=f"{ticker} Historical Candlestick",
                xaxis_title="Time",
                yaxis_title="Price",
                template="plotly_white",
            )
            out_path = self.output_dir / f"{ticker.lower()}_candlestick.html"
            fig.write_html(out_path)
            self.logger.info("Saved candlestick chart to %s", out_path)
