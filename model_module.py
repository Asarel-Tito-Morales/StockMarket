from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout

from utils import save_json, utc_now_iso


class PriceImpactModel:
    def __init__(self, config: Dict[str, Any], logger: Any) -> None:
        self.config = config
        self.logger = logger
        self.random_seed = int(config.get("RANDOM_SEED", 42))
        tf.random.set_seed(self.random_seed)
        np.random.seed(self.random_seed)

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        model_df = df.copy()

        feature_cols = [
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
            "news_count",
            "sentiment_compound_mean",
            "sentiment_compound_sum",
            "positive_count",
            "neutral_count",
            "negative_count",
            "return_1",
            "rolling_volatility_3",
            "price_range_pct",
            "volume_change",
            "sentiment_balance",
        ]

        model_df["ticker"] = model_df["ticker"].astype(str)
        ticker_dummies = pd.get_dummies(model_df["ticker"], prefix="ticker")
        X = pd.concat([model_df[feature_cols], ticker_dummies], axis=1)
        y = model_df["target_direction"]

        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        return X, y, list(X.columns)

    def build_model(self, input_dim: int, num_classes: int) -> tf.keras.Model:
        model = Sequential([
            Dense(128, activation="relu", input_shape=(input_dim,)),
            Dropout(0.25),
            Dense(64, activation="relu"),
            Dropout(0.20),
            Dense(32, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ])
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train(self, merged_target_df: pd.DataFrame, output_dir: str | Path) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        train_df = merged_target_df.dropna(subset=["target_direction"]).copy()

        if train_df.empty or len(train_df) < 25:
            raise ValueError("Not enough labeled rows to train the model. Increase lookback/history.")

        X, y_raw, feature_names = self.prepare_features(train_df)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=float(self.config.get("TRAIN_TEST_SPLIT", 0.2)),
            random_state=self.random_seed,
            stratify=y if len(np.unique(y)) > 1 else None,
        )

        model = self.build_model(input_dim=X_train.shape[1], num_classes=len(label_encoder.classes_))
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=int(self.config.get("EARLY_STOPPING_PATIENCE", 5)),
            restore_best_weights=True,
        )

        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=int(self.config.get("EPOCHS", 20)),
            batch_size=int(self.config.get("BATCH_SIZE", 32)),
            verbose=0,
            callbacks=[early_stopping],
        )

        y_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)

        metrics = {
            "trained_at": utc_now_iso(),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "classes": label_encoder.classes_.tolist(),
            "classification_report": classification_report(
                y_test, y_pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "feature_names": feature_names,
            "history": {
                "loss": [float(x) for x in history.history.get("loss", [])],
                "val_loss": [float(x) for x in history.history.get("val_loss", [])],
                "accuracy": [float(x) for x in history.history.get("accuracy", [])],
                "val_accuracy": [float(x) for x in history.history.get("val_accuracy", [])],
            },
        }

        model_path = output_dir / self.config["MODEL_OUTPUT_FILE"]
        scaler_path = output_dir / self.config.get("SCALER_OUTPUT_FILE", "feature_scaler.joblib")
        metrics_path = output_dir / self.config.get("METRICS_OUTPUT_FILE", "model_metrics.json")
        encoder_path = output_dir / "label_encoder.joblib"
        features_path = output_dir / "feature_names.json"

        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(label_encoder, encoder_path)
        save_json({"feature_names": feature_names}, features_path)
        save_json(metrics, metrics_path)

        self.logger.info("Saved model to %s", model_path)
        return metrics

    def predict(self, merged_target_df: pd.DataFrame, output_dir: str | Path) -> pd.DataFrame:
        output_dir = Path(output_dir)

        model_path = output_dir / self.config["MODEL_OUTPUT_FILE"]
        scaler_path = output_dir / self.config.get("SCALER_OUTPUT_FILE", "feature_scaler.joblib")
        encoder_path = output_dir / "label_encoder.joblib"
        features_path = output_dir / "feature_names.json"

        if not all(path.exists() for path in [model_path, scaler_path, encoder_path, features_path]):
            raise FileNotFoundError("Model artifacts not found. Train the model first.")

        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(encoder_path)
        feature_names = pd.read_json(features_path, typ="series")
        if isinstance(feature_names, pd.Series) and "feature_names" in feature_names.index:
            feature_names = feature_names["feature_names"]

        scored_df = merged_target_df.copy()
        X, _, current_features = self.prepare_features(scored_df)

        for name in feature_names:
            if name not in X.columns:
                X[name] = 0
        X = X[list(feature_names)]

        X_scaled = scaler.transform(X)
        probs = model.predict(X_scaled, verbose=0)
        pred_idx = np.argmax(probs, axis=1)
        labels = label_encoder.inverse_transform(pred_idx)

        scored_df["predicted_direction"] = labels
        for i, class_name in enumerate(label_encoder.classes_):
            scored_df[f"prob_{class_name.lower()}"] = probs[:, i]

        pred_file = output_dir / self.config.get("PREDICTIONS_OUTPUT_FILE", "predictions.csv")
        scored_df.to_csv(pred_file, index=False)
        return scored_df
