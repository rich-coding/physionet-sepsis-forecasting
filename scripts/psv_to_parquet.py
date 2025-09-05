#!/usr/bin/env python3
"""
Convierte archivos .psv de pacientes en archivos Parquet unificados por set (A y B),
garantizando columnas y tipos consistentes. Opcionalmente sube los archivos a S3.

- Lee todos los archivos .psv de data/raw/training_setA y data/raw/training_setB.
- Unifica columnas y tipos de datos.
- Guarda los Parquet en data/raw/all_patients_setA.parquet y data/raw/all_patients_setB.parquet.
- Verifica que ambos Parquet tengan las mismas columnas.
- (Opcional) Sube los archivos Parquet a S3.

Variables de entorno soportadas:
    S3_BUCKET: Nombre del bucket S3
    S3_KEY_A: Ruta destino en S3 para setA
    S3_KEY_B: Ruta destino en S3 para setB
    UPLOAD_TO_S3: "1" para subir a S3, "0" para solo local
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import boto3
except ImportError:
    boto3 = None

# Configuración de rutas
LOCAL_DATA_DIR_A = "data/raw/training_setA"
LOCAL_DATA_DIR_B = "data/raw/training_setB"
OUTPUT_LOCAL_PATH_A = "data/raw/all_patients_setA.parquet"
OUTPUT_LOCAL_PATH_B = "data/raw/all_patients_setB.parquet"
S3_BUCKET = os.getenv("S3_BUCKET", "physionet-sepsis-forecasting")
S3_KEY_A = os.getenv("S3_KEY_A", "sepsis2019/raw/all_patients_setA.parquet")
S3_KEY_B = os.getenv("S3_KEY_B", "sepsis2019/raw/all_patients_setB.parquet")
UPLOAD_TO_S3 = os.getenv("UPLOAD_TO_S3", "0")

def normalize_cols(cols):
    """Normaliza nombres de columnas: quita espacios y reemplaza '/' por '_'."""
    return [c.strip().replace(' ', '_').replace('/', '_') for c in cols]

def list_psv_files(root):
    """Lista todos los archivos .psv en la carpeta raíz (recursivo)."""
    return sorted(glob.glob(os.path.join(root, "**", "*.psv"), recursive=True))

def infer_patient_id(path):
    """Extrae el patient_id del nombre de archivo."""
    return os.path.splitext(os.path.basename(path))[0]

def union_header(files):
    """Obtiene el conjunto unificado de columnas de todos los archivos .psv."""
    union = set()
    for fp in files:
        hdr = pd.read_csv(fp, sep="|", nrows=0)
        hdr.columns = normalize_cols(hdr.columns)
        union |= set(hdr.columns.tolist())
    return ["patient_id"] + sorted(list(union - {"patient_id"}))

def cast_dtypes_for_stability(df, min_numeric_ratio=0.9):
    """Convierte columnas a float64 si la mayoría de los valores son numéricos."""
    for c in df.columns:
        if c == "patient_id":
            continue
        s_num = pd.to_numeric(df[c], errors="coerce")
        if s_num.notna().mean() >= min_numeric_ratio:
            df[c] = s_num.astype("float64")
    return df

def psvs_to_parquet(psv_dir, output_path, all_cols=None):
    """Convierte todos los .psv de una carpeta en un único Parquet."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    files = list_psv_files(psv_dir)
    if not files:
        print(f"[ERROR] No .psv files found under {psv_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Found {len(files)} patient files in {psv_dir}.")
    if all_cols is None:
        all_cols = union_header(files)
    print(f"[INFO] Unified columns: {len(all_cols)} (including patient_id)")
    writer = None
    rows_total = 0
    try:
        for i, path in enumerate(files, 1):
            pid = infer_patient_id(path)
            df = pd.read_csv(path, sep="|")
            df.columns = normalize_cols(df.columns)
            if "patient_id" not in df.columns:
                df.insert(0, "patient_id", pid)
            missing = [c for c in all_cols if c not in df.columns]
            for m in missing:
                df[m] = np.nan
            df = df[all_cols]
            df = cast_dtypes_for_stability(df)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
            else:
                if table.schema != writer.schema:
                    table = table.cast(writer.schema)
            writer.write_table(table)
            rows_total += df.shape[0]
            if i % 500 == 0:
                print(f"[INFO] Processed {i}/{len(files)} patients...")
    finally:
        if writer is not None:
            writer.close()
    print(f"[DONE] Wrote {rows_total} rows into {output_path}")
    return all_cols

def verify_columns(path_a, path_b):
    """Verifica que las columnas de ambos Parquet sean iguales."""
    df_a = pd.read_parquet(path_a)
    df_b = pd.read_parquet(path_b)
    cols_a = list(df_a.columns)
    cols_b = list(df_b.columns)
    print(f"[CHECK] Columns in setA: {cols_a}")
    print(f"[CHECK] Columns in setB: {cols_b}")
    assert cols_a == cols_b, "[ERROR] Column mismatch between setA and setB!"
    print("[CHECK] Column consistency verified between setA and setB.")

def upload_to_s3(local_path, bucket, key):
    """Sube un archivo local a S3."""
    if boto3 is None:
        print("[ERROR] boto3 is not installed. Cannot upload to S3.", file=sys.stderr)
        return
    print(f"[INFO] Uploading to s3://{bucket}/{key} ...")
    s3 = boto3.client("s3")
    with open(local_path, "rb") as f:
        s3.put_object(Bucket=bucket, Key=key, Body=f.read())
    print(f"[DONE] Uploaded to s3://{bucket}/{key}")

def main():
    # Procesar setA
    print("\n[PROCESS] Converting training_setA ...")
    all_cols_A = psvs_to_parquet(LOCAL_DATA_DIR_A, OUTPUT_LOCAL_PATH_A)
    # Procesar setB usando el mismo esquema de columnas
    print("\n[PROCESS] Converting training_setB ...")
    all_cols_B = psvs_to_parquet(LOCAL_DATA_DIR_B, OUTPUT_LOCAL_PATH_B, all_cols=all_cols_A)
    # Verificar columnas
    verify_columns(OUTPUT_LOCAL_PATH_A, OUTPUT_LOCAL_PATH_B)
    # Subir a S3 si está habilitado
    if UPLOAD_TO_S3 == "1":
        upload_to_s3(OUTPUT_LOCAL_PATH_A, S3_BUCKET, S3_KEY_A)
        upload_to_s3(OUTPUT_LOCAL_PATH_B, S3_BUCKET, S3_KEY_B)

if __name__ == "__main__":
    main()
