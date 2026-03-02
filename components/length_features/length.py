import argparse, os, glob, re
import pandas as pd

WORD_RE = re.compile(r"\w+")

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
            raise ValueError(f"Missing required column: {k}. Found: {list(df.columns)}")

    text = df[args.text_col].astype(str)

    out = pd.DataFrame({
        "asin": df["asin"].astype(str),
        "reviewerID": df["reviewerID"].astype(str),
        "review_length_chars": text.str.len().astype("int32"),
        "review_length_words": text.map(lambda s: len(WORD_RE.findall(s))).astype("int32"),
    })

    # carry a timestamp column if present (helps later for feature store)
    for ts in ["unixReviewTime", "reviewTime", "timestamp"]:
        if ts in df.columns:
            out[ts] = df[ts].values
            break

    os.makedirs(args.out, exist_ok=True)
    out_file = os.path.join(args.out, "data.parquet")
    out.to_parquet(out_file, index=False)

    print("Wrote:", out_file)
    print("Rows:", len(out), "Cols:", len(out.columns))

if __name__ == "__main__":
    main()
