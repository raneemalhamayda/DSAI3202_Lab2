import argparse, os, re, glob
import pandas as pd

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
NUM_RE = re.compile(r"\b\d+(\.\d+)?\b")
PUNCT_RE = re.compile(r"[^\w\s]")  # punctuation -> space

def read_parquet_any(path: str) -> pd.DataFrame:
    # AML uri_folder comes as a folder path
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True)
        if not files:
            raise FileNotFoundError(f"No parquet files found under: {path}")
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return pd.read_parquet(path)

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    s = URL_RE.sub(" URL ", s)
    s = NUM_RE.sub(" NUM ", s)
    s = PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--min_chars", type=int, default=10)
    ap.add_argument("--text_col", type=str, default="reviewText")
    args = ap.parse_args()

    df = read_parquet_any(args.data)

    if args.text_col not in df.columns:
        raise ValueError(f"Expected column '{args.text_col}'. Found: {list(df.columns)}")

    df[args.text_col] = df[args.text_col].map(normalize_text)

    # filter very short rows
    df = df[df[args.text_col].str.len() >= args.min_chars].copy()

    os.makedirs(args.out, exist_ok=True)
    out_file = os.path.join(args.out, "data.parquet")
    df.to_parquet(out_file, index=False)

    print("Normalized rows:", len(df))
    print("Wrote:", out_file)

if __name__ == "__main__":
    main()
