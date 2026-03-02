import argparse, os
import pandas as pd

def infer_type(dtype):
    if pd.api.types.is_integer_dtype(dtype): return "long"
    if pd.api.types.is_float_dtype(dtype): return "float"
    if pd.api.types.is_bool_dtype(dtype): return "boolean"
    return "string"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--source_uri", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--timestamp_col", default="event_time")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)

    index_cols = ["asin", "reviewerID"]
    ts_col = args.timestamp_col if args.timestamp_col in df.columns else None
    feature_cols = [c for c in df.columns if c not in index_cols and c != ts_col]

    lines = []
    lines.append("$schema: http://azureml/sdk-2-0/FeatureSetSpec.json")
    lines.append("")
    lines.append("source:")
    lines.append("  type: parquet")
    lines.append(f"  path: {args.source_uri}")
    if ts_col:
        lines.append("  timestamp_column:")
        lines.append(f"    name: {ts_col}")
    lines.append("")
    lines.append("features:")
    for c in feature_cols:
        lines.append(f"  - name: {c}")
        lines.append(f"    type: {infer_type(df[c].dtype)}")
    lines.append("")
    lines.append("index_columns:")
    for c in index_cols:
        lines.append(f"  - name: {c}")
        lines.append("    type: string")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "FeatureSetSpec.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Wrote:", out_path)
    print("Features:", len(feature_cols))
    print("Timestamp used:", ts_col)

if __name__ == "__main__":
    main()
