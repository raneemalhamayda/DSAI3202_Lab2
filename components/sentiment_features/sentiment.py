import argparse, os, glob
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def read_parquet_any(path: str) -> pd.DataFrame:
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True)
        if not files:
            raise FileNotFoundError(f"No parquet files found under: {path}")
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return pd.read_parquet(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--text_col", type=str, default="reviewText")
    args = ap.parse_args()

    df = read_parquet_any(args.data)

    for k in ["asin", "reviewerID", args.text_col]:
        if k not in df.columns:
            raise ValueError(f"Missing {k}. Found: {list(df.columns)}")

    analyzer = SentimentIntensityAnalyzer()
    scores = df[args.text_col].astype(str).map(analyzer.polarity_scores)

    out = pd.DataFrame({
        "asin": df["asin"].astype(str),
        "reviewerID": df["reviewerID"].astype(str),
        "sentiment_neg": scores.map(lambda d: d["neg"]).astype("float32"),
        "sentiment_neu": scores.map(lambda d: d["neu"]).astype("float32"),
        "sentiment_pos": scores.map(lambda d: d["pos"]).astype("float32"),
        "sentiment_compound": scores.map(lambda d: d["compound"]).astype("float32"),
    })

    for ts in ["unixReviewTime", "reviewTime", "timestamp"]:
        if ts in df.columns:
            out[ts] = df[ts].values
            break

    os.makedirs(args.out, exist_ok=True)
    out_file = os.path.join(args.out, "data.parquet")
    out.to_parquet(out_file, index=False)

    print("Wrote:", out_file, "rows:", len(out))

if __name__ == "__main__":
    main()
