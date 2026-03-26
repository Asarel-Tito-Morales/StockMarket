from __future__ import annotations

import argparse
import time
from pathlib import Path

import nltk
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler

from config_loader import load_config
from merger import DataMerger
from model_module import PriceImpactModel
from news_collector import get_news_collector
from plotting_module import Plotter
from sentiment_module import SentimentScorer
from stock_collector import StockCollector
from utils import (
    append_or_replace_csv,
    ensure_output_dir,
    print_io_directories,
    setup_logging,
    should_retrain,
)


def run_pipeline(config_path: str) -> None:
    config = load_config(config_path)
    output_dir = ensure_output_dir(config["OUTPUT_DIR"])
    logger = setup_logging(output_dir, config["LOG_FILE"])
    print_io_directories(config, logger)

    try:
        nltk.download("vader_lexicon", quiet=True)

        news_collector = get_news_collector(config, logger)
        stock_collector = StockCollector(config, logger)
        sentiment_scorer = SentimentScorer(config, logger)
        merger = DataMerger(config, logger)
        modeler = PriceImpactModel(config, logger)
        plotter = Plotter(config, logger)

        logger.info("Starting pipeline run.")

        news_df = news_collector.fetch()
        stock_df = stock_collector.fetch()

        if not news_df.empty:
            news_df = sentiment_scorer.score(news_df)

        news_file = output_dir / config["NEWS_OUTPUT_FILE"]
        stock_file = output_dir / config["STOCK_OUTPUT_FILE"]
        merged_file = output_dir / config["MERGED_OUTPUT_FILE"]

        if not news_df.empty:
            export_news = news_df.copy()
            export_news["published_at"] = export_news["published_at"].astype(str)
            append_or_replace_csv(export_news, news_file, subset=["ticker", "url"])
            logger.info("Saved %s news rows to %s", len(export_news), news_file)
        else:
            logger.warning("No news rows collected.")

        if not stock_df.empty:
            export_stock = stock_df.copy()
            export_stock["timestamp"] = export_stock["timestamp"].astype(str)
            append_or_replace_csv(export_stock, stock_file, subset=["ticker", "timestamp"])
            logger.info("Saved %s stock rows to %s", len(export_stock), stock_file)
        else:
            logger.warning("No stock rows collected.")

        aligned_news_df, news_agg_df, merged_df = merger.merge(news_df, stock_df)
        merged_target_df = merger.create_target(merged_df)

        if not merged_target_df.empty:
            export_merged = merged_target_df.copy()
            for col in ["timestamp", "aligned_timestamp", "raw_news_timestamp"]:
                if col in export_merged.columns:
                    export_merged[col] = export_merged[col].astype(str)
            export_merged.to_csv(merged_file, index=False)
            logger.info("Saved merged rows to %s", merged_file)
        else:
            logger.warning("Merged dataset is empty.")

        if config.get("ENABLE_TRAINING", True) and not merged_target_df.empty and should_retrain(output_dir, config):
            try:
                metrics = modeler.train(merged_target_df, output_dir)
                logger.info("Training complete. Accuracy=%.4f", metrics["accuracy"])
            except Exception as exc:
                logger.exception("Training failed: %s", exc)

        if config.get("ENABLE_PREDICTION", True) and not merged_target_df.empty:
            try:
                predictions = modeler.predict(merged_target_df, output_dir)
                logger.info("Prediction output generated with %s rows.", len(predictions))
            except Exception as exc:
                logger.exception("Prediction failed: %s", exc)

        if config.get("ENABLE_PLOTTING", True):
            try:
                plotter.plot_price_with_sentiment(aligned_news_df, merged_target_df)
                if config.get("ENABLE_CANDLESTICK", True):
                    plotter.plot_candlestick(stock_df)
            except Exception as exc:
                logger.exception("Plotting failed: %s", exc)

        logger.info("Pipeline run finished successfully.")
    except Exception as exc:
        logger.exception("Fatal pipeline error: %s", exc)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Stock sentiment and price correlation agent")
    parser.add_argument("--config", default="json.cfg", help="Path to json.cfg")
    args = parser.parse_args()

    config = load_config(args.config)

    if config.get("CONTINUOUS_RUN", False):
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            func=run_pipeline,
            trigger="interval",
            minutes=int(config.get("RUN_EVERY_MINUTES", 5)),
            args=[args.config],
            max_instances=1,
            coalesce=True,
        )
        scheduler.start()

        # Run once immediately so the user does not wait for the first interval.
        run_pipeline(args.config)

        try:
            while True:
                time.sleep(30)
        except (KeyboardInterrupt, SystemExit):
            scheduler.shutdown()
    else:
        run_pipeline(args.config)


if __name__ == "__main__":
    main()
