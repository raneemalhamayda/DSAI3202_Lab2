import argparse, os, glob
import pandas as pd

TS_COLS = ["unixReviewTime", "reviewTime", "timestamp", "reviewTime_x", "reviewTime_y", "unixReviewTime_x", "unixReviewTime_y"]

def read_parquet_any(path: str) -> pd.DataFrame:
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True)
        if not files:
            raise FileNotFoundError(f"No parquet files found under: {path}")
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return pd.read_parquet(path)

def dedup(df: pd.DataFrame):
    return df.drop_duplicates(subset=["asin", "reviewerID"], keep="first")

def drop_ts(df: pd.DataFrame) -> pd.DataFrame:
    # drop any timestamp-ish columns so they don't collide during merges
    cols_to_drop = [c for c in df.columns if c in TS_COLS]
    return df.drop(columns=cols_to_drop, errors="ignore")

def keep_one_ts(df: pd.DataFrame) -> pd.DataFrame:
    # keep exactly one timestamp column and rename to event_time
    for c in ["unixReviewTime", "reviewTime", "timestamp"]:
        if c in df.columns:
            out = df.copy()
            out = out.rename(columns={c: "event_time"})
            # remove any other timestamp variants
            out = out.drop(columns=[x for x in TS_COLS if x in out.columns and x != "event_time"], errors="ignore")
            return out
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=str, required=True)
    ap.add_argument("--sentiment", type=str, required=True)
    ap.add_argument("--tfidf", type=str, required=True)
    ap.add_argument("--sbert", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    length_df = dedup(read_parquet_any(args.length))
    sent_df   = dedup(read_parquet_any(args.sentiment))
    tfidf_df  = dedup(read_parquet_any(args.tfidf))
    sbert_df  = dedup(read_parquet_any(args.sbert))

    # Keep ONE timestamp column only from length_df (rename to event_time)
    length_df = keep_one_ts(length_df)

    # Drop timestamps from other dfs to avoid duplicate columns
    sent_df  = drop_ts(sent_df)
    tfidf_df = drop_ts(tfidf_df)
    sbert_df = drop_ts(sbert_df)

    # Merge order: start small, merge big last
    merged = sbert_df.merge(length_df, on=["asin","reviewerID"], how="inner")
    merged = merged.merge(sent_df,   on=["asin","reviewerID"], how="inner")

    # Filter TF-IDF to matching keys before merging (saves memory)
    keys = merged[["asin","reviewerID"]]
    tfidf_small = tfidf_df.merge(keys, on=["asin","reviewerID"], how="inner")

    merged = merged.merge(tfidf_small, on=["asin","reviewerID"], how="inner")

    # Final safety: ensure no duplicate column names remain
    merged = merged.loc[:, ~merged.columns.duplicated()].copy()

    os.makedirs(args.out, exist_ok=True)
    out_file = os.path.join(args.out, "data.parquet")
    merged.to_parquet(out_file, index=False, engine="pyarrow", compression="snappy")

    print("Merged rows:", len(merged))
    print("Merged cols:", len(merged.columns))
    print("Wrote:", out_file)

if __name__ == "__main__":
    main()
