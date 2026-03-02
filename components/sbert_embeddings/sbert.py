import argparse, os, glob
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

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
    ap.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_rows", type=int, default=5000)  # keeps runtime reasonable
    args = ap.parse_args()

    df = read_parquet_any(args.data)
    for k in ["asin", "reviewerID", args.text_col]:
        if k not in df.columns:
            raise ValueError(f"Missing {k}. Found: {list(df.columns)}")

    df = df.head(args.max_rows).copy()

    model = SentenceTransformer(args.model_name)
    texts = df[args.text_col].astype(str).tolist()

    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        show_progress_bar=True
    ).astype(np.float32)

    cols = [f"sbert_{i}" for i in range(emb.shape[1])]
    out = pd.concat(
        [
            df[["asin", "reviewerID"]].astype(str).reset_index(drop=True),
            pd.DataFrame(emb, columns=cols),
        ],
        axis=1
    )

    for ts in ["unixReviewTime", "reviewTime", "timestamp"]:
        if ts in df.columns:
            out[ts] = df[ts].values
            break

    os.makedirs(args.out, exist_ok=True)
    out_file = os.path.join(args.out, "data.parquet")
    out.to_parquet(out_file, index=False)

    print("Wrote:", out_file, "rows:", len(out), "dim:", emb.shape[1])

if __name__ == "__main__":
    main()
