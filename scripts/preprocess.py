# scripts/preprocess.py
import argparse
import pandas as pd
from pathlib import Path

def main(args):
    """
    Función principal para ejecutar el preprocesamiento de datos.
    """
    # Define las rutas de entrada y salida
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Asegúrate de que el directorio de salida exista
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Cargando datos crudos desde {input_path}...")
    # --- Aquí va tu lógica para cargar los datos ---
    # Ejemplo: Cargar todos los archivos .psv de la carpeta raw
    raw_files = list(input_path.glob('*.psv'))
    df_list = [pd.read_csv(f, sep='|') for f in raw_files]
    df = pd.concat(df_list, ignore_index=True)
    print("Datos crudos cargados.")

    print("Iniciando preprocesamiento...")
    # --- Lógica de preprocesamiento ---
    # 1. Limpieza de datos (manejo de nulos, outliers, etc.)
    # Ejemplo: Imputar valores faltantes con la media o mediana
    # df.fillna(df.mean(), inplace=True)
    
    # 2. Ingeniería de características (crear nuevas variables)
    # Ejemplo: Crear variables de promedios móviles para signos vitales
    
    # 3. Normalización o escalado de variables numéricas
    # scaler = StandardScaler()
    # df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    print("Preprocesamiento completado.")

    # Guardar los datos procesados
    output_file = output_path / "processed_data.csv"
    df.to_csv(output_file, index=False)
    print(f"Datos procesados guardados en {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocesa los datos crudos del PhysioNet Challenge.")
    parser.add_argument("--input", required=True, help="Ruta a la carpeta con datos crudos (data/raw)")
    parser.add_argument("--output", required=True, help="Ruta para guardar los datos procesados (data/processed)")
    
    args = parser.parse_args()
    main(args)