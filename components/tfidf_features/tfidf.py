import argparse, os, glob
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def read_parquet_any(path: str) -> pd.DataFrame:
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True)
        if not files:
            raise FileNotFoundError(f"No parquet files found under: {path}")
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return pd.read_parquet(path)

def write_out(df_in: pd.DataFrame, X, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    cols = [f"tfidf_{i}" for i in range(X.shape[1])]
    arr = X.toarray().astype(np.float32)   # OK for max_features ~500
    out = pd.concat(
        [
            df_in[["asin", "reviewerID"]].astype(str).reset_index(drop=True),
            pd.DataFrame(arr, columns=cols),
        ],
        axis=1
    )

    # keep a timestamp column if present
    for ts in ["unixReviewTime", "reviewTime", "timestamp"]:
        if ts in df_in.columns:
            out[ts] = df_in[ts].values
            break

    out_file = os.path.join(out_dir, "data.parquet")
    out.to_parquet(out_file, index=False)
    print("Wrote:", out_file, "rows:", len(out), "features:", X.shape[1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, required=True)
    ap.add_argument("--val", type=str, required=True)
    ap.add_argument("--test", type=str, required=True)
    ap.add_argument("--train_out", type=str, required=True)
    ap.add_argument("--val_out", type=str, required=True)
    ap.add_argument("--test_out", type=str, required=True)
    ap.add_argument("--text_col", type=str, default="reviewText")
    ap.add_argument("--max_features", type=int, default=500)
    ap.add_argument("--ngram_max", type=int, default=2)
    args = ap.parse_args()

    train_df = read_parquet_any(args.train)
    val_df = read_parquet_any(args.val)
    test_df = read_parquet_any(args.test)

    for df_ in [train_df, val_df, test_df]:
        for k in ["asin", "reviewerID", args.text_col]:
            if k not in df_.columns:
                raise ValueError(f"Missing {k} in columns: {list(df_.columns)}")

    vec = TfidfVectorizer(
        max_features=args.max_features,
        stop_words="english",
        ngram_range=(1, args.ngram_max),
        dtype=np.float32,
    )

    # Fit ONLY on train (avoid leakage)
    vec.fit(train_df[args.text_col].astype(str))

    X_train = vec.transform(train_df[args.text_col].astype(str))
    X_val = vec.transform(val_df[args.text_col].astype(str))
    X_test = vec.transform(test_df[args.text_col].astype(str))

    write_out(train_df, X_train, args.train_out)
    write_out(val_df, X_val, args.val_out)
    write_out(test_df, X_test, args.test_out)

if __name__ == "__main__":
    main()
