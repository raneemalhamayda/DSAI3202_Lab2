# DSAI3202 Lab 2 — Data Ingestion in Azure (Electronics Reviews)

This repository documents the full workflow for Lab 2: ingesting Amazon Electronics review data into Azure Blob Storage, fixing the product metadata file into valid JSON Lines, and running an Azure Data Factory (ADF) mapping data flow to write partitioned Parquet output.

---

## Overview
**Goal:**
1. Upload raw review JSON to Azure Blob Storage.
2. Fix `meta_Electronics` into valid line-delimited JSON and upload the corrected file.
3. Use Azure Data Factory (ADF) Mapping Data Flow to read raw JSON and write partitioned Parquet output.

**Storage account:** `amazonelectron1446226083`  
**Containers used:** `raw`, `processed`

> **Security note:** Do NOT commit SAS tokens, access keys, or any secrets to GitHub. All commands below use `<SAS_TOKEN>` placeholders.

---

## Part A — Upload raw reviews file to Blob Storage

### A1) Verify local file exists
```bash
ls -lh reviews_Electronics_5.json

## A2) Upload to the raw container using AzCopy
azcopy copy "./reviews_Electronics_5.json" \
"https://amazonelectron1446226083.blob.core.windows.net/raw/reviews_Electronics_5.json?<SAS_TOKEN>" \
--overwrite=true

## A3) Verify upload
azcopy list "https://amazonelectron1446226083.blob.core.windows.net/raw?<SAS_TOKEN>" | grep -i reviews

Part B — Fix and re-upload product metadata (Required)
B1) Download the metadata file (Stanford SNAP)

wget -O meta_Electronics.json.gz \
https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz


B2) Upload the metadata .gz into the raw container
azcopy copy "./meta_Electronics.json.gz" \
"https://amazonelectron1446226083.blob.core.windows.net/raw/meta_Electronics.json.gz?<SAS_TOKEN>" \
--overwrite=true

B3) Decompress locally
gunzip -f meta_Electronics.json.gz
# Produces: meta_Electronics.json

B4) Convert to valid line-delimited JSON (meta_Electronics_fixed.json)

The original file uses Python dictionary formatting (e.g., single quotes), so it is not valid JSON. Convert each line into valid JSON:

python3 << 'EOF'
import ast, json

input_file = "meta_Electronics.json"
output_file = "meta_Electronics_fixed.json"

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        obj = ast.literal_eval(line)
        fout.write(json.dumps(obj) + "\n")

print("Metadata conversion complete:", output_file)
EOF

B5) Upload the fixed metadata file to raw
azcopy copy "./meta_Electronics_fixed.json" \
"https://amazonelectron1446226083.blob.core.windows.net/raw/meta_Electronics_fixed.json?<SAS_TOKEN>" \
--overwrite=true

B6) Verify raw contains all expected inputs
azcopy list "https://amazonelectron1446226083.blob.core.windows.net/raw?<SAS_TOKEN>" \
| grep -i -E "reviews|meta"


Expected:

reviews_Electronics_5.json

meta_Electronics.json.gz

meta_Electronics_fixed.json

Part C — Azure Data Factory (ADF): JSON → Parquet partitioned by year
C1) Create the processed container (if not already created)

In Azure Portal → Storage account → Containers → Add container

Name: processed

C2) Configure datasets

Source dataset: dsReviewsRawJson

Linked service points to storage account amazonelectron1446226083

Container / File system: raw

File path: reviews_Electronics_5.json

Enable:

Allow schema drift

Infer drifted column types

Sink dataset: ds_reviews_processed (name may vary)

Linked service points to storage account amazonelectron1446226083

Container / File system: processed

Folder path: reviews/

Output format: Parquet

Fix applied: The pipeline initially failed with
Job failed due to reason: at Source 'dsReviewsRawJson': 'container' or 'fileSystem' is required
Resolved by setting the dataset container/file system to raw.

C3) Run the pipeline / data flow

Turn on Data flow debug if required for Data Preview.

Run the pipeline and confirm Status = Succeeded.

C4) Verify Parquet output partitions

In Azure Portal:

Storage account → Containers → processed → reviews

Expect folders such as:

review_year=1999

review_year=2000

review_year=2001



CLI check (optional):

azcopy list "https://amazonelectron1446226083.blob.core.windows.net/processed/reviews?<SAS_TOKEN>" \
--recursive | grep -i "review_year=" | head

--note: added the screenshots in the screenshots folder

