import argparse
import os
import time
import numpy as np
import pandas as pd
import joblib
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data",   type=str, required=True)
    parser.add_argument("--val_data",     type=str, required=True)
    parser.add_argument("--test_data",    type=str, required=True)
    parser.add_argument("--deploy_data",  type=str, required=True)
    parser.add_argument("--model_output", type=str, required=True)
    parser.add_argument("--C", type=float, default=1.183)
    parser.add_argument("--max_iter", type=int, default=1000)
    return parser.parse_args()

def load_data(folder_path):
    parquet_path = os.path.join(folder_path, "data.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"data.parquet not found in: {folder_path}")
    return pd.read_parquet(parquet_path)

def create_labels(df):
    if "overall" not in df.columns:
        raise RuntimeError("Column 'overall' is missing.")
    df = df.copy()
    df["label"] = (df["overall"] >= 4).astype(int)
    return df

def build_features(df):
    parts = []

    # SBERT columns
    sbert_cols = sorted([c for c in df.columns if c.startswith("sbert_")])
    if sbert_cols:
        parts.append(df[sbert_cols].values.astype(float))
        print(f"  SBERT: {len(sbert_cols)} dims")

    # TF-IDF columns
    tfidf_cols = sorted([c for c in df.columns if c.startswith("tfidf_")])
    if tfidf_cols:
        parts.append(df[tfidf_cols].values.astype(float))
        print(f"  TF-IDF: {len(tfidf_cols)} dims")

    # Sentiment columns - EXCLUDING compound to reduce leakage
    sentiment_cols = [c for c in df.columns if c in [
        "sentiment_neg", "sentiment_neu", "sentiment_pos"
    ]]
    if sentiment_cols:
        parts.append(df[sentiment_cols].values.astype(float))
        print(f"  Sentiment (no compound): {sentiment_cols}")

    # Length columns
    length_cols = [c for c in df.columns if c in [
        "review_length_chars", "review_length_words",
        "review_length", "word_count"
    ]]
    if length_cols:
        parts.append(df[length_cols].values.astype(float))
        print(f"  Length: {length_cols}")

    if not parts:
        raise RuntimeError("No feature columns found.")

    X = np.hstack(parts)
    X = np.nan_to_num(X, nan=0.0)
    print(f"  Feature matrix: {X.shape}")

    # Sanity check - print label correlation with first few features
    return X

def evaluate(model, X, y, split):
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    acc  = accuracy_score(y, preds)
    auc  = roc_auc_score(y, proba)
    prec = precision_score(y, preds, zero_division=0)
    rec  = recall_score(y, preds, zero_division=0)
    f1   = f1_score(y, preds, zero_division=0)
    mlflow.log_metric(f"{split}_accuracy",  acc)
    mlflow.log_metric(f"{split}_auc",       auc)
    mlflow.log_metric(f"{split}_precision", prec)
    mlflow.log_metric(f"{split}_recall",    rec)
    mlflow.log_metric(f"{split}_f1",        f1)
    print(f"{split}: acc={acc:.4f} auc={auc:.4f} f1={f1:.4f}")

def main():
    args = parse_args()
    start_time = time.time()

    mlflow.start_run()
    mlflow.log_param("C",        args.C)
    mlflow.log_param("max_iter", args.max_iter)

    print("Loading data...")
    train_df = create_labels(load_data(args.train_data))
    val_df   = create_labels(load_data(args.val_data))
    test_df  = create_labels(load_data(args.test_data))

    print(f"Label distribution: {dict(pd.Series(train_df['label']).value_counts())}")
    print(f"overall distribution: {dict(pd.Series(train_df['overall']).value_counts())}")

    print("Building features...")
    X_train = build_features(train_df);  y_train = train_df["label"]
    X_val   = build_features(val_df);    y_val   = val_df["label"]
    X_test  = build_features(test_df);   y_test  = test_df["label"]

    print("Training model...")
    model = LogisticRegression(C=args.C, max_iter=args.max_iter,
                               random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("Evaluating...")
    evaluate(model, X_train, y_train, "train")
    evaluate(model, X_val,   y_val,   "val")
    evaluate(model, X_test,  y_test,  "test")

    print("Saving model...")
    os.makedirs(args.model_output, exist_ok=True)
    model_path = os.path.join(args.model_output, "model.pkl")
    joblib.dump(model, model_path)

    runtime = time.time() - start_time
    mlflow.log_metric("training_runtime_seconds", runtime)
    print(f"Done. Total time: {runtime:.1f}s")
    mlflow.end_run()

if __name__ == "__main__":
    main()
