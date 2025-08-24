# Predicción Temprana de Sepsis con Machine Learning

Este proyecto tiene como objetivo desarrollar un modelo de machine learning para la predicción temprana de sepsis en pacientes de unidades de cuidados intensivos (UCI). El desarrollo se basa en los datos y la problemática definidos en el [PhysioNet/Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/). 

## Integrantes del equipo

- Monica Alejandra Alvarez Carrillo (ma.alvarezc1@uniandes.edu.co)
- Daniel Eduardo Ayala Ramírez (de.ayala@uniandes.edu.co)
- Manuela Alejandra Hernández Otálora (ma.hernandezo1@uniandes.edu)
- Richard Stiv Murcia Huerfano (rs.murcia@uniandes.edu.co)

## Descripción del Problema

La sepsis es una condición crítica que requiere detección y tratamiento tempranos para mejorar las probabilidades de supervivencia del paciente.  Este proyecto se enfoca en crear un sistema que, utilizando datos clínicos, alerte sobre un posible cuadro de sepsis con al menos 6 horas de antelación, permitiendo una intervención médica oportuna. 

## Dataset

El conjunto de datos utilizado es el proporcionado por el **PhysioNet Challenge 2019**. Incluye datos de series temporales (signos vitales) y datos demográficos de más de 40,000 pacientes de UCI. 

- **Fuente Original:** [https://physionet.org/content/challenge-2019/1.0.0/](https://physionet.org/content/challenge-2019/1.0.0/)
- **Variables:** El dataset contiene 40 covariables que incluyen signos vitales, resultados de laboratorio y datos demográficos por cada hora de estancia del paciente en la UCI. 
- **Etiqueta (`SepsisLabel`):** La variable objetivo indica si el paciente desarrollará sepsis.

## Modelo Propuesto

Para abordar este problema, se propone un modelo basado en redes neuronales recurrentes, específicamente una arquitectura que combine **LSTM (Long Short-Term Memory)** con capas densas.

1.  **Capas LSTM:** Ideales para capturar patrones temporales en los datos de signos vitales y laboratorio. Manejan secuencias de longitud variable y pueden modelar dependencias a largo plazo.
2.  **Capas Densas (Dense):** Se utilizarán para procesar la información demográfica estática del paciente y para combinar las características extraídas por las capas LSTM.
3.  **Capa de Salida:** Una capa `sigmoid` para la clasificación binaria (Sepsis/No Sepsis).

Este enfoque híbrido permite aprovechar tanto la naturaleza secuencial de los datos como la información estática del paciente para una predicción más robusta.

## Estructura del Proyecto

El repositorio está organizado para seguir las mejores prácticas en proyectos de ciencia de datos, asegurando la reproducibilidad y la colaboración.

```
physionet-sepsis-forecasting/
├── data/
│   ├── raw/          # Datos originales inmutables (versionados con DVC)
│   ├── processed/    # Datos limpios y transformados (versionados con DVC)
│   └── inference/    # Nuevos datos para generar predicciones (no versionados con Git)
├── notebooks/        # Jupyter notebooks para exploración y experimentación
├── scripts/          # Scripts de Python para el pipeline (preprocesamiento, entrenamiento, etc.)
├── models/           # Modelos entrenados (versionados con DVC)
├── results/          # Métricas de evaluación y predicciones (salidas del pipeline)
├── dvc.yaml          # Define las etapas del pipeline de DVC
└── README.md         # Documentación general del proyecto
```

*   **`data/`**: Contiene todos los datos del proyecto.
    *   **`raw/`**: Los datos originales sin ninguna modificación. Esta carpeta debe ser tratada como de solo lectura.
    *   **`processed/`**: Datos limpios, transformados y listos para ser usados por el modelo. Son generados por el script `scripts/preprocess.py`.
    *   **`inference/`**: Aquí se deben colocar los nuevos datos sobre los que se desea ejecutar una predicción.
*   **`notebooks/`**: Cuadernos de Jupyter para análisis exploratorio, visualizaciones y pruebas de modelos. No forman parte del pipeline reproducible.
*   **`scripts/`**: Contiene todo el código fuente modularizado que compone el pipeline (preprocesamiento, entrenamiento, evaluación, predicción).
*   **`models/`**: Almacena los modelos serializados (ej. archivos `.pkl`) generados durante la etapa de entrenamiento.
*   **`results/`**: Guarda las salidas del pipeline, como los archivos JSON con las métricas de rendimiento y los CSV con las predicciones.
*   **`dvc.yaml`**: Archivo clave que define cada etapa del pipeline de Machine Learning, sus dependencias y salidas.

## Pipeline de Machine Learning con DVC

Este proyecto utiliza **DVC (Data Version Control)** para crear un pipeline reproducible y versionar los datos y modelos que son demasiado grandes para Git. El pipeline completo está definido en el archivo `dvc.yaml`.

```yaml
# dvc.yaml
stages:
  preprocess:
    desc: "Preprocesa los datos crudos y los guarda en la carpeta processed"
    cmd: python scripts/preprocess.py --input data/raw --output data/processed
    deps:
      - data/raw
      - scripts/preprocess.py
    outs:
      - data/processed

  train:
    desc: "Entrena el modelo usando los datos procesados"
    cmd: python scripts/train.py --input data/processed --output models
    deps:
      - data/processed
      - scripts/train.py
    outs:
      - models/model.pkl

  evaluate:
    desc: "Evalúa el modelo y genera métricas de rendimiento"
    cmd: python scripts/evaluate.py --model models/model.pkl --data data/processed --output results/metrics.json
    deps:
      - models/model.pkl
      - data/processed
      - scripts/evaluate.py
    metrics:
      - results/metrics.json:
          cache: false

  predict:
    desc: "Genera predicciones sobre nuevos datos usando el modelo entrenado"
    cmd: python scripts/predict.py --model models/model.pkl --input data/inference --output results/predictions.csv
    deps:
      - models/model.pkl
      - data/inference
      - scripts/predict.py
    outs:
      - results/predictions.csv
```

## Flujo de Trabajo para Colaboradores

Para empezar a trabajar, sigue estos pasos:

1.  **Clona el repositorio:**
    ```bash
    git clone https://github.com/rich-coding/physionet-sepsis-forecasting
    cd physionet-sepsis-forecasting
    ```

2.  **Instala las dependencias:**
    Se recomienda usar un entorno virtual.
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Descarga los datos y modelos versionados:**
    Este comando descarga los archivos rastreados por DVC (de `data/processed`, `models/`, etc.) desde el almacenamiento remoto configurado.
    ```bash
    dvc pull
    ```

4.  **Crea tu propia rama de trabajo:**
    **Nunca trabajes directamente sobre la rama `main`**. Cada vez que inicies una nueva tarea (una nueva funcionalidad, la corrección de un bug, etc.), crea una rama para aislar tus cambios.

    ```bash
    # 1. Asegúrate de estar en la rama principal y tener la última versión
    git checkout main
    git pull origin main

    # 2. Crea y muévete a tu nueva rama. Usa un nombre descriptivo.
    # Ejemplos: feature/agregar-modelo-lstm, fix/corregir-preprocesamiento
    git checkout -b feature/nombre-descriptivo-de-tu-rama
    ```

5.  **Realiza tus cambios:**
    Ahora que estás en tu propia rama, puedes modificar el código en `scripts/`, experimentar en `notebooks/`, o actualizar los datos en `data/raw/` de forma segura.

6.  **Reproduce el pipeline y versiona tus resultados:**
    Después de hacer cambios, ejecuta el pipeline. DVC se encargará de re-ejecutar solo las etapas que fueron afectadas por tus cambios.

    ```bash
    # Ejecuta el pipeline completo o una etapa específica (ej. dvc repro train)
    dvc repro
    ```
    Una vez que estés satisfecho con los resultados, guarda tus cambios con Git y DVC.

    ```bash
    # Añade los cambios en el código Y los archivos .dvc actualizados
    git add .

    # Confirma los cambios en Git
    git commit -m "Descripción clara y concisa de tus cambios"
    ```

7.  **Sube tus cambios y crea un Pull Request:**
    Para que tus cambios sean revisados e incorporados a la rama `main`, debes subirlos a GitHub y abrir un *Pull Request*.

    ```bash
    # 1. Sube los datos y modelos versionados por DVC al almacenamiento remoto
    dvc push

    # 2. Sube tu rama con los cambios de código a GitHub
    git push -u origin feature/nombre-descriptivo-de-tu-rama
    ```

8.  **Abre el Pull Request en GitHub:**
    *   Ve al repositorio en tu navegador.
    *   GitHub detectará automáticamente que subiste una nueva rama y te mostrará un botón para **"Compare & pull request"**.
    *   Haz clic, añade un título y una descripción detallada de tus cambios, y crea el Pull Request para que el equipo pueda revisarlo.
