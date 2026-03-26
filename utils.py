from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from zoneinfo import ZoneInfo


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def setup_logging(output_dir: str | Path, log_file_name: str) -> logging.Logger:
    output_dir = ensure_output_dir(output_dir)
    log_path = output_dir / log_file_name

    logger = logging.getLogger("stock_sentiment_agent")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def print_io_directories(config: Dict[str, Any], logger: logging.Logger) -> None:
    logger.info("Reading config from: %s", Path(config["CONFIG_PATH"]).resolve())
    logger.info("Config directory: %s", Path(config["CONFIG_DIR"]).resolve())
    logger.info("Writing outputs to: %s", Path(config["OUTPUT_DIR"]).resolve())


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def get_timezone(tz_name: str) -> ZoneInfo:
    return ZoneInfo(tz_name)


def parse_date_maybe(value: str | None, tz_name: str) -> Optional[pd.Timestamp]:
    if not value:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(tz_name)
    return ts


def to_timezone(series_or_ts, tz_name: str):
    if isinstance(series_or_ts, pd.Series):
        s = pd.to_datetime(series_or_ts, errors="coerce", utc=True)
        return s.dt.tz_convert(tz_name)
    ts = pd.Timestamp(series_or_ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(tz_name)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def append_or_replace_csv(df: pd.DataFrame, path: str | Path, subset: list[str] | None = None) -> None:
    path = Path(path)
    if path.exists():
        old = pd.read_csv(path)
        combined = pd.concat([old, df], ignore_index=True)
        if subset:
            combined = combined.drop_duplicates(subset=subset, keep="last")
        combined.to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


def floor_to_interval(ts: pd.Timestamp, minutes: int) -> pd.Timestamp:
    return ts.floor(f"{minutes}min")


def nearest_to_interval(ts: pd.Timestamp, minutes: int) -> pd.Timestamp:
    return ts.round(f"{minutes}min")


def get_alignment_timestamp(ts: pd.Timestamp, minutes: int, rule: str = "floor") -> pd.Timestamp:
    if rule == "nearest":
        return nearest_to_interval(ts, minutes)
    return floor_to_interval(ts, minutes)


def should_retrain(output_dir: Path, config: Dict[str, Any]) -> bool:
    metrics_path = output_dir / config.get("METRICS_OUTPUT_FILE", "model_metrics.json")
    every_hours = int(config.get("MODEL_RETRAIN_EVERY_HOURS", 24))
    if not metrics_path.exists():
        return True

    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        last_trained = pd.Timestamp(metrics.get("trained_at"))
        if last_trained.tzinfo is None:
            last_trained = last_trained.tz_localize("UTC")
        return pd.Timestamp.utcnow().tz_localize("UTC") - last_trained >= pd.Timedelta(hours=every_hours)
    except Exception:
        return True


def safe_numeric(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)


def utc_now_iso() -> str:
    return pd.Timestamp.utcnow().isoformat()
