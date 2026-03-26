from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class ConfigError(Exception):
    pass


def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    required = [
        "OUTPUT_DIR",
        "NEWS_OUTPUT_FILE",
        "STOCK_OUTPUT_FILE",
        "MERGED_OUTPUT_FILE",
        "MODEL_OUTPUT_FILE",
        "LOG_FILE",
        "TICKERS",
        "NEWS_SOURCE",
        "STOCK_INTERVAL",
        "RESAMPLE_INTERVAL_MINUTES",
        "RUN_EVERY_MINUTES",
        "CONTINUOUS_RUN",
        "TIMEZONE",
        "SENTIMENT_MODEL",
        "PREDICTION_TARGET",
        "LOOKAHEAD_PERIODS",
        "HISTORICAL_CANDLE_MONTHS",
        "ENABLE_PLOTTING",
        "ENABLE_CANDLESTICK",
        "ENABLE_TRAINING",
        "ENABLE_PREDICTION"
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ConfigError(f"Missing required config fields: {missing}")

    output_dir = Path(config["OUTPUT_DIR"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = (path.parent / output_dir).resolve()
    config["OUTPUT_DIR"] = str(output_dir)

    config["CONFIG_PATH"] = str(path)
    config["CONFIG_DIR"] = str(path.parent.resolve())

    if not isinstance(config["TICKERS"], list) or not config["TICKERS"]:
        raise ConfigError("TICKERS must be a non-empty list")

    return config
