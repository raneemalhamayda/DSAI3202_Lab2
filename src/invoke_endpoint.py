import json
import os
import numpy as np
import pandas as pd
import requests
import urllib3
from sklearn.metrics import accuracy_score, f1_score

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ENDPOINT_URL = os.environ.get("ENDPOINT_URL", "https://amazon-review-ep-60300390.qatarcentral.inference.ml.azure.com/score")
API_KEY = os.environ.get("ENDPOINT_API_KEY", "")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

def load_data(folder_path):
    parquet_path = os.path.join(folder_path, "data.parquet")
    return pd.read_parquet(parquet_path)

def create_labels(df):
    df = df.copy()
    df["label"] = (df["overall"] >= 4).astype(int)
    return df

def build_features(df):
    parts = []
    sbert_cols = sorted([c for c in df.columns if c.startswith("sbert_")])
    if sbert_cols:
        parts.append(df[sbert_cols].values.astype(float))
    tfidf_cols = sorted([c for c in df.columns if c.startswith("tfidf_")])
    if tfidf_cols:
        parts.append(df[tfidf_cols].values.astype(float))
    sentiment_cols = [c for c in df.columns if c in [
        "sentiment_neg", "sentiment_neu", "sentiment_pos"
    ]]
    if sentiment_cols:
        parts.append(df[sentiment_cols].values.astype(float))
    length_cols = [c for c in df.columns if c in [
        "review_length_chars", "review_length_words"
    ]]
    if length_cols:
        parts.append(df[length_cols].values.astype(float))
    X = np.hstack(parts)
    X = np.nan_to_num(X, nan=0.0)
    return X

def main():
    import sys
    deploy_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not deploy_path:
        print("Usage: python invoke_endpoint.py <path_to_deploy_folder>")
        return

    print(f"Loading deployment data from: {deploy_path}")
    df = create_labels(load_data(deploy_path))
    X = build_features(df)
    y_true = df["label"].values
    print(f"Samples: {len(X)}, Features: {X.shape[1]}")

    batch_size = 100
    all_preds = []

    print(f"Sending {len(X)} samples in batches of {batch_size}...")
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        payload = {"data": batch.tolist()}
        response = requests.post(
            ENDPOINT_URL,
            headers=headers,
            json=payload,
            verify=False
        )
        if response.status_code != 200:
            print(f"Error on batch {i}: {response.text}")
            continue
        result = response.json()
        all_preds.extend(result["predictions"])
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(X)}...")

    all_preds = np.array(all_preds)
    acc = accuracy_score(y_true[:len(all_preds)], all_preds)
    f1  = f1_score(y_true[:len(all_preds)], all_preds, zero_division=0)

    print(f"\nDeployment dataset results:")
    print(f"  Samples evaluated: {len(all_preds)}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

if __name__ == "__main__":
    main()
