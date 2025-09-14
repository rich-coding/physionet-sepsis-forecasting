from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from joblib import dump
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from mlflow.models.signature import infer_signature

from utils import best_threshold_fbeta, metrics_block, plot_and_log_curves, save_metadata

FEATURES = ['patient_id','SepsisLabel','ICULOS','Temp','BaseExcess','DBP','FiO2',
            'Gender','Age','HCO3','HR','HospAdmTime','Magnesium','O2Sat','Resp']
NUMERIC_FEATURES = ['ICULOS','Temp','BaseExcess','DBP','FiO2','Age','HCO3','HR','HospAdmTime','Magnesium','O2Sat','Resp']
CATEGORIC_FEATURE = 'Gender' 
TARGET = 'SepsisLabel'
ID_COL = 'patient_id'

def load_data() -> pd.DataFrame:
    raw_dir = Path("data/raw")
    files = [
        raw_dir / "all_patients_setA.parquet",
        raw_dir / "all_patients_setB.parquet",
    ]
    dfs = []
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"No se encontró {f}")
        dfs.append(pd.read_parquet(f))
    df = pd.concat(dfs, ignore_index=True)

    # Asegurar columnas
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")

    # --- Normalización de tipos ---
    # patient_id puede venir string o numérico: lo mantenemos como string para el split por paciente
    df[ID_COL] = df[ID_COL].astype(str)

    # El target debe ser entero 0/1
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).round().clip(0, 1).astype(int)

    # Features numéricos en float (Gender lo dejaremos 0/1 int)
   
    for c in NUMERIC_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Gender a entero 0/1 (tolerante si viene float)
    df[CATEGORIC_FEATURE] = pd.to_numeric(df[CATEGORIC_FEATURE], errors="coerce").fillna(0).round().clip(0, 1).astype(int)

    return df

def split_by_patient(df: pd.DataFrame, valid_frac: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    pats = df[ID_COL].astype(str).unique()
    rng.shuffle(pats)
    n_valid = int(len(pats)*valid_frac)
    valid_pats = set(pats[:n_valid])
    train = df[~df[ID_COL].isin(valid_pats)].copy()
    valid = df[df[ID_COL].isin(valid_pats)].copy()

    return train, valid

def build_preprocess():
    feats = [c for c in FEATURES if c not in (ID_COL, TARGET)]
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    return feats, pipe

def main():
    models_dir = Path(os.getenv("MODELS_DIR", "models/production"))
    mlruns_uri = os.getenv("MLFLOW_TRACKING_URI", "file:" + str(Path("mlruns").absolute()))
    mlflow.set_tracking_uri(mlruns_uri)
    mlflow.set_experiment("sepsis2019_production")

    df = load_data()
    train_df, valid_df = split_by_patient(df, valid_frac=0.2, seed=42)

    feats, preproc = build_preprocess()
    X_train = train_df[feats]
    y_train = train_df[TARGET].astype(int)
    X_valid = valid_df[feats]
    y_valid = valid_df[TARGET].astype(int)

    classes = np.array([0, 1])
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {0: cw[0], 1: cw[1]}

    hgb_params = dict(
        learning_rate=0.05,
        max_iter=1000,
        max_leaf_nodes=80,
        min_samples_leaf=60,
        l2_regularization=1.0,
        class_weight=class_weight,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=80,
        scoring="average_precision",
        random_state=42,
    )

    X_train_t = preproc.fit_transform(X_train)
    X_valid_t = preproc.transform(X_valid)

    hgb = HistGradientBoostingClassifier(**hgb_params)

    with mlflow.start_run(run_name="HGB_weighted"):
        mlflow.set_tags({
            "project": "DesarrolloSoluciones",
            "component": "sepsis_6h",
            "model_family": "HistGradientBoosting",
            "dataset": "physionet2019",
            "split": "train/valid",
            "owner": "deayala",
        })

        hgb.fit(X_train_t, y_train)
        p_valid = hgb.predict_proba(X_valid_t)[:, 1]

        f2_opt, thr_opt = best_threshold_fbeta(y_valid, p_valid, beta=2.0)
        block = metrics_block(y_valid, p_valid, threshold=thr_opt)

        mlflow.log_params({
            **hgb_params,
            "reweighting": "class_weight -> sample_weight",
            "cw_pos": float(class_weight[1]),
            "cw_neg": float(class_weight[0]),
        })
        mlflow.log_metrics({
            "valid_auroc": block["auroc"],
            "valid_auprc": block["auprc"],
            "valid_accuracy_at_bestF2thr": block["acc_at_best_f2_thr"],
            "valid_f2_at_bestF2thr": block["f2_at_best_f2_thr"],
            "valid_accuracy_at_0.5": block["acc_at_0_5"],
            "valid_f2_at_0.5": block["f2_at_0_5"],
            "valid_best_threshold_F2": float(block["best_thr"]),
            "n_train": int(len(y_train)),
            "n_valid": int(len(y_valid)),
        })

        curves_dir = Path("artifacts/curves")
        plot_and_log_curves(y_valid, p_valid, curves_dir)
        mlflow.log_artifacts(str(curves_dir), artifact_path="curves")

        models_dir.mkdir(parents=True, exist_ok=True)
        from joblib import dump
        dump(hgb, models_dir / "hgb_model.joblib")
        dump(preproc, models_dir / "preprocess.joblib")
        save_metadata(models_dir / "metadata.json", {
            "features_order": feats,
            "threshold_f2": thr_opt,
            "classes": [0, 1],
            "model_type": "HistGradientBoostingClassifier"
        })

        sig = infer_signature(pd.DataFrame(X_train, columns=feats), hgb.predict(X_train_t))
        mlflow.sklearn.log_model(hgb, 
                                 artifact_path="model", 
                                 signature=sig,
                                 input_example=X_train.iloc[:1])

        print(f"[HGB] AUROC={block['auroc']:.4f} | AUPRC={block['auprc']:.4f} | "
              f"ACC@bestF2={block['acc_at_best_f2_thr']:.4f} | F2@bestF2={block['f2_at_best_f2_thr']:.4f} "
              f"(thr={thr_opt:.3f}) | ACC@0.5={block['acc_at_0_5']:.4f} | F2@0.5={block['f2_at_0_5']:.4f}")

if __name__ == "__main__":
    main()
