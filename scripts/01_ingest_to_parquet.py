#!/usr/bin/env python3
import os, sys, io, glob, boto3, pandas as pd, numpy as np, pyarrow as pa, pyarrow.parquet as pq

LOCAL_DATA_DIR = os.getenv("LOCAL_DATA_DIR", "/home/ubuntu/dataset/training_setB")
OUTPUT_LOCAL_PATH = os.getenv("OUTPUT_LOCAL_PATH", "/home/ubuntu/data/sepsis2019/all_patients_setB.parquet")
S3_BUCKET = os.getenv("S3_BUCKET", "physionet-sepsis-forecasting")
S3_KEY = os.getenv("S3_OUTPUT_KEY", "sepsis2019/raw/all_patients_setB.parquet")
UPLOAD_TO_S3 = os.getenv("UPLOAD_TO_S3", "1")

s3 = boto3.client("s3")

def normalize_cols(cols): 
    return [c.strip().replace(' ', '_').replace('/', '_') for c in cols]

def list_psv_files(root): 
    return sorted(glob.glob(os.path.join(root, "**", "*.psv"), recursive=True))

def infer_patient_id(path): 
    return os.path.splitext(os.path.basename(path))[0]

def union_header(files):
    union=set()
    for fp in files:
        hdr=pd.read_csv(fp, sep="|", nrows=0); hdr.columns=normalize_cols(hdr.columns); union|=set(hdr.columns.tolist())
    return ["patient_id"]+sorted(list(union-{"patient_id"}))

def cast_dtypes_for_stability(df, min_numeric_ratio=0.9):
    for c in df.columns:
        if c == "patient_id":
            continue
        s_num = pd.to_numeric(df[c], errors="coerce")
        if s_num.notna().mean() >= min_numeric_ratio:
            df[c] = s_num.astype("float64")
    return df

def main():
    os.makedirs(os.path.dirname(OUTPUT_LOCAL_PATH), exist_ok=True)
    files=list_psv_files(LOCAL_DATA_DIR)
    if not files: print(f"[ERROR] No .psv files found under {LOCAL_DATA_DIR}", file=sys.stderr); sys.exit(1)
    print(f"[INFO] Found {len(files)} patient files. Building unified schema...")
    all_cols=union_header(files)
    print(f"[INFO] Unified columns: {len(all_cols)} (including patient_id)")
    writer=None; rows_total=0
    try:
        for i, path in enumerate(files,1):
            pid=infer_patient_id(path)
            df=pd.read_csv(path, sep="|"); df.columns=normalize_cols(df.columns)
            if "patient_id" not in df.columns: df.insert(0,"patient_id",pid)
            missing=[c for c in all_cols if c not in df.columns]
            for m in missing: df[m]=np.nan
            df=df[all_cols]
            df=cast_dtypes_for_stability(df)
            table=pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer=pq.ParquetWriter(OUTPUT_LOCAL_PATH, table.schema, compression="snappy")
            else:
                if table.schema != writer.schema: table = table.cast(writer.schema)
            writer.write_table(table)
            rows_total += df.shape[0]
            if i % 500 == 0: print(f"[INFO] Processed {i}/{len(files)} patients...")
    finally:
        if writer is not None: writer.close()
    print(f"[DONE] Wrote {rows_total} rows into {OUTPUT_LOCAL_PATH}")
    if UPLOAD_TO_S3 == "1":
        print(f"[INFO] Uploading to s3://{S3_BUCKET}/{S3_KEY} ...")
        with open(OUTPUT_LOCAL_PATH,"rb") as f: s3.put_object(Bucket=S3_BUCKET, Key=S3_KEY, Body=f.read())
        print(f"[DONE] Uploaded to s3://{S3_BUCKET}/{S3_KEY}")

if __name__ == "__main__": main()
