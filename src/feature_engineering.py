from __future__ import annotations
import pandas as pd
import numpy as np

BASE_FEATS = [
    "ICULOS","Temp","BaseExcess","DBP","FiO2","Gender","Age","HCO3","HR",
    "HospAdmTime","Magnesium","O2Sat","Resp"
]

# Ventanas (en horas) típicas para señales en UCI
ROLL_WINDOWS = [3, 6]

def _per_patient_sort(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["patient_id","ICULOS"])

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve df con columnas originales + features nuevas:
    - Razones respiratorias: SF (SpO2/FiO2) y ROX = SF / Resp
    - Deltas/variaciones: Δ1h, Δ3h, Δ6h de señales clave
    - Estadísticos móviles: mean/std/min/max (3h, 6h)
    - Señales hemo-respiratorias compuestas
    """
    df = _per_patient_sort(df.copy())

    # --- Ratios respiratorios (SF/ROX) ---
    # FiO2 puede estar como 0..1; O2Sat 0..100
    eps = 1e-6
    df["sf_ratio"] = df["O2Sat"] / (df["FiO2"].clip(lower=eps))
    df["rox_index"] = df["sf_ratio"] / (df["Resp"].clip(lower=eps))

    # --- Índices hemodinámicos aproximados ---
    # No hay SBP en dataset; usamos proxies con DBP
    df["hr_dbp"] = df["HR"] / (df["DBP"].clip(lower=eps))     # proxy de “stress”
    df["pulse_pressure_proxy"] = (df["DBP"] * 1.5)            # heurística simple

    # --- Deltas y rollings para señales clave ---
    key_signals = ["Temp","HR","DBP","O2Sat","Resp","HCO3","BaseExcess","Magnesium","FiO2"]
    df_group = df.groupby("patient_id", group_keys=False)

    # Lags
    for col in key_signals:
        df[f"{col}_lag1"] = df_group[col].shift(1)
        df[f"{col}_lag3"] = df_group[col].shift(3)
        df[f"{col}_lag6"] = df_group[col].shift(6)

    # Deltas (valor actual - lag)
    for col in key_signals:
        for h in [1,3,6]:
            df[f"delta_{col}_{h}h"] = df[col] - df[f"{col}_lag{h}"]

    # Rolling stats por ventana
    for w in ROLL_WINDOWS:
        rw = df_group[key_signals].rolling(window=w, min_periods=1)
        stats = rw.mean().add_suffix(f"_mean_{w}h")
        stats["dummy"] = 0  # para asegurar merge por índice
        stds  = df_group[key_signals].rolling(window=w, min_periods=1).std().add_suffix(f"_std_{w}h")
        mins  = df_group[key_signals].rolling(window=w, min_periods=1).min().add_suffix(f"_min_{w}h")
        maxs  = df_group[key_signals].rolling(window=w, min_periods=1).max().add_suffix(f"_max_{w}h")
        # Concat preservando orden/índice
        roll = pd.concat([stats.drop(columns=["dummy"], errors="ignore"), stds, mins, maxs], axis=1)
        df = pd.concat([df, roll.reset_index(level=0, drop=True)], axis=1)

    # Interacciones útiles
    df["temp_hr"] = df["Temp"] * df["HR"]
    df["age_hr"]  = df["Age"] * df["HR"]
    df["sf_resp"] = df["sf_ratio"] * df["Resp"]  # sensibilidad a ventilación

    # Re-ordenar
    return df
