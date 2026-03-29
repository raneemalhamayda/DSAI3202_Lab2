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
    parser.add_argument("--deploy", type=str, required=True)  # NEW
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = read_parquet_any(args.data)

    for c in ["reviewText", "asin", "reviewerID", "overall"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Deploy split = most recent 10% by review_year
    if "review_year" not in df.columns:
        # derive it if not present
        df["review_year"] = pd.to_datetime(df["unixReviewTime"], unit="s").dt.year

    df = df.sort_values("review_year").reset_index(drop=True)
    cutoff = int(len(df) * 0.90)
    deploy_df = df.iloc[cutoff:]
    remaining = df.iloc[:cutoff]

    # Split remaining into 60 / 15 / 15
    # 15/90 = 0.1667 of remaining goes to test, then half of that to val
    train_temp, test_df = train_test_split(remaining, test_size=0.1667, random_state=args.seed)
    train_df, val_df = train_test_split(train_temp, test_size=0.1667, random_state=args.seed)

    os.makedirs(args.train, exist_ok=True)
    os.makedirs(args.val, exist_ok=True)
    os.makedirs(args.test, exist_ok=True)
    os.makedirs(args.deploy, exist_ok=True)

    train_df.to_parquet(os.path.join(args.train, "data.parquet"), index=False)
    val_df.to_parquet(os.path.join(args.val, "data.parquet"), index=False)
    test_df.to_parquet(os.path.join(args.test, "data.parquet"), index=False)
    deploy_df.to_parquet(os.path.join(args.deploy, "data.parquet"), index=False)

    print("Train:", len(train_df), "Val:", len(val_df), 
          "Test:", len(test_df), "Deploy:", len(deploy_df))

if __name__ == "__main__":
    main()
