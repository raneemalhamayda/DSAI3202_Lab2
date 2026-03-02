import argparse
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

def read_parquet_any(path: str) -> pd.DataFrame:
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True)
        if not files:
            raise FileNotFoundError(f"No parquet files found under: {path}")
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return pd.read_parquet(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = read_parquet_any(args.data)

    for c in ["reviewText", "asin", "reviewerID"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    train_val_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.seed, shuffle=True)
    val_ratio_of_trainval = args.val_size / (1.0 - args.test_size)
    train_df, val_df = train_test_split(train_val_df, test_size=val_ratio_of_trainval, random_state=args.seed, shuffle=True)

    os.makedirs(args.train, exist_ok=True)
    os.makedirs(args.val, exist_ok=True)
    os.makedirs(args.test, exist_ok=True)

    train_df.to_parquet(os.path.join(args.train, "data.parquet"), index=False)
    val_df.to_parquet(os.path.join(args.val, "data.parquet"), index=False)
    test_df.to_parquet(os.path.join(args.test, "data.parquet"), index=False)

    print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))

if __name__ == "__main__":
    main()
