import requests
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ENDPOINT_URL = "https://amazon-review-ep-60300390.qatarcentral.inference.ml.azure.com/score"
API_KEY = "3ErJmcV29w0syo6ct3LuEi3Z7o4NBsTf6pOPkWkiFC6N7GlQ5OCVJQQJ99CCAAAAAAAAAAAAINFRAZML1q4u"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

def load_data(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".parquet"):
                full = os.path.join(root, f)
                print(f"Loading: {full}")
                return pd.read_parquet(full)
    raise FileNotFoundError(f"No parquet file found in {path}")

def build_features(df):
    sbert_cols = sorted([c for c in df.columns if c.startswith("sbert_")], key=lambda x: int(x.split("_")[1]))
    tfidf_cols = sorted([c for c in df.columns if c.startswith("tfidf_")], key=lambda x: int(x.split("_")[1]))
    sent_cols  = [c for c in ["sentiment_neg", "sentiment_neu", "sentiment_pos"] if c in df.columns]
    len_cols   = [c for c in ["review_length_chars", "review_length_words"] if c in df.columns]

    print(f"  sbert cols: {len(sbert_cols)}, tfidf cols: {len(tfidf_cols)}, sentiment cols: {len(sent_cols)}, length cols: {len(len_cols)}")

    parts = [df[sbert_cols].values, df[tfidf_cols].values, df[sent_cols].values, df[len_cols].values]
    return np.hstack(parts)

def main():
    print("Loading deployment dataset...")
    df = load_data("./deploy_data")
    print(f"Rows: {len(df)}, Cols: {len(df.columns)}")

    df["label"] = (df["overall"] >= 4).astype(int)
    y_true = df["label"].values

    print("Building features...")
    X = build_features(df)
    print(f"Feature matrix shape: {X.shape}")

    all_preds = []
    batch_size = 100
    total = len(X)
    print(f"Sending {total} rows to endpoint in batches of {batch_size}...")

    for i in range(0, total, batch_size):
        batch = X[i:i+batch_size]
        payload = {"data": batch.tolist()}
        response = requests.post(ENDPOINT_URL, headers=headers, data=json.dumps(payload))
        if response.status_code != 200:
            print(f"Error on batch {i}: {response.status_code} {response.text}")
            continue
        result = response.json()
        preds = result.get("predictions", [])
        all_preds.extend(preds)
        if i % 500 == 0:
            print(f"  Processed {i}/{total}...")

    print(f"\nTotal predictions received: {len(all_preds)}")
    y_pred = np.array(all_preds[:len(y_true)])
    acc = accuracy_score(y_true[:len(y_pred)], y_pred)
    f1  = f1_score(y_true[:len(y_pred)], y_pred)

    print(f"\n=== Deployment Results ===")
    print(f"Deployment Accuracy: {acc:.4f}")
    print(f"Deployment F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()
