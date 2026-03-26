from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import yfinance as yf


class StockCollector:
    def __init__(self, config: Dict[str, Any], logger: Any) -> None:
        self.config = config
        self.logger = logger

    def fetch(self) -> pd.DataFrame:
        tickers: List[str] = self.config["TICKERS"]
        interval = self.config["STOCK_INTERVAL"]
        period = self.config.get("STOCK_PERIOD", "")
        start_date = self.config.get("START_DATE", "")
        end_date = self.config.get("END_DATE", "")
        tz_name = self.config["TIMEZONE"]

        frames = []

        for ticker in tickers:
            kwargs = {
                "tickers": ticker,
                "interval": interval,
                "auto_adjust": False,
                "progress": False,
                "prepost": False,
                "threads": False,
            }

            if start_date and end_date:
                kwargs["start"] = start_date
                kwargs["end"] = end_date
            elif start_date and not end_date:
                kwargs["start"] = start_date
            else:
                kwargs["period"] = period or "1mo"

            self.logger.info("Downloading stock data for %s with args=%s", ticker, kwargs)
            df = yf.download(**kwargs)

            if df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            df = df.reset_index()
            dt_col = "Datetime" if "Datetime" in df.columns else "Date"
            df = df.rename(columns={dt_col: "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df["timestamp"] = df["timestamp"].dt.tz_convert(tz_name)
            df["ticker"] = ticker

            expected = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            for col in expected:
                if col not in df.columns:
                    df[col] = pd.NA

            df = df[["ticker", "timestamp", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
            frames.append(df)

        if not frames:
            return pd.DataFrame(columns=["ticker", "timestamp", "Open", "High", "Low", "Close", "Adj Close", "Volume"])

        stock_df = pd.concat(frames, ignore_index=True)
        return stock_df
