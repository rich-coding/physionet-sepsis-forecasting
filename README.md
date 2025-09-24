# Predicción Temprana de Sepsis con Machine Learning

Este proyecto tiene como objetivo desarrollar un modelo de machine learning para la predicción temprana de sepsis en pacientes de unidades de cuidados intensivos (UCI). El desarrollo se basa en los datos y la problemática definidos en el [PhysioNet/Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/).

## Integrantes del equipo

- Monica Alejandra Alvarez Carrillo (ma.alvarezc1@uniandes.edu.co)
- Daniel Eduardo Ayala Ramírez (de.ayala@uniandes.edu.co)
- Manuela Alejandra Hernández Otálora (ma.hernandezo1@uniandes.edu)
- Richard Stiv Murcia Huerfano (rs.murcia@uniandes.edu.co)

## Descripción del Problema

La sepsis es una condición crítica, potencialmente mortal, que surge de la respuesta desregulada del cuerpo a una infección y es una de las principales causas de mortalidad en las UCI. Su detección temprana es un desafío clínico debido a que los síntomas iniciales son inespecíficos y pueden confundirse con otras condiciones. El retraso en el diagnóstico aumenta significativamente la mortalidad y los costos hospitalarios.

Este proyecto se enfoca en crear un sistema que, utilizando datos clínicos, alerte sobre un posible cuadro de sepsis con al menos 6 horas de antelación, permitiendo una intervención médica oportuna.

## Dataset

El conjunto de datos utilizado es el proporcionado por el **PhysioNet Challenge 2019**, que incluye datos de más de 40,000 estancias de pacientes en UCI de dos sistemas hospitalarios.

- **Fuente Original:** [https://physionet.org/content/challenge-2019/1.0.0/](https://physionet.org/content/challenge-2019/1.0.0/)
- **Variables:** El dataset contiene 40 covariables que incluyen signos vitales (p. ej., HR, O2Sat, Temp), resultados de laboratorio (p. ej., lactato, bilirrubina) y datos demográficos (edad, género) por cada hora de estancia del paciente en la UCI.
- **Etiqueta (`SepsisLabel`):** La variable objetivo indica si el paciente desarrollará sepsis, definida según los criterios de Sepsis-3.

### Preprocesamiento y Feature Engineering

Para preparar los datos para el modelado, se aplicó un preprocesamiento adaptado a cada tipo de modelo, considerando el fuerte desbalance de clases (98.2% no sepsis vs. 1.8% sepsis). Las estrategias clave incluyeron:
- **Imputación:** Se utilizó imputación por mediana para la Regresión Logística y LSTM para evitar sesgos, mientras que los modelos basados en árboles (HGB, XGBoost) manejaron valores faltantes de forma nativa.
- **Normalización:** Se aplicó `StandardScaler` para los modelos sensibles a la escala de las variables (Regresión Logística y LSTM).
- **Ingeniería de Características:** Se crearon nuevas variables para capturar la dinámica temporal y las relaciones fisiológicas:
    - **Índices clínicos:** SF ratio (SpO2/FiO2) y ROX index.
    - **Características temporales:** Lags (valores pasados) y deltas (cambios en el tiempo) a 1, 3 y 6 horas.
    - **Estadísticas móviles:** Media, desviación estándar, mínimo y máximo en ventanas de 3 y 6 horas.
    - **Interacciones no lineales:** Combinaciones de variables fisiológicamente relevantes (p. ej., temperatura × frecuencia cardíaca).

## Modelos Desarrollados

Se adoptó una estrategia de modelado progresiva, comparando varios enfoques para encontrar el mejor equilibrio entre interpretabilidad y rendimiento:

1.  **Regresión Logística (LR):** Modelo base por su simplicidad e interpretabilidad.
2.  **Histogram-based Gradient Boosting (HGB) y XGBoost:** Modelos de boosting capaces de capturar relaciones no lineales y manejar eficientemente valores faltantes.
3.  **Long Short-Term Memory (LSTM):** Red neuronal recurrente seleccionada por su capacidad para modelar secuencias y patrones temporales, una característica esencial de los datos de UCI. Para este modelo, los datos se estructuraron en secuencias utilizando una ventana deslizante de 8 horas.

## Evaluación

La evaluación de los modelos se centró en métricas robustas ante el desbalance de clases, priorizando la detección de casos positivos (pacientes con sepsis) y penalizando los falsos negativos.

- **Métricas Clave:**
    - **F2-Score:** Métrica que combina precisión y sensibilidad (recall), dando más importancia a la sensibilidad para minimizar los falsos negativos, lo cual es crítico en un contexto clínico.
    - **AUPRC (Area Under the Precision-Recall Curve):** Más informativa que el AUC-ROC en datasets con un gran desbalance de clases.

### Resultados

- La **Regresión Logística** sirvió como una línea base, pero su rendimiento fue limitado en este complejo problema.
- Los modelos **HGB y XGBoost** mostraron un rendimiento superior a la Regresión Logística, siendo HGB el que obtuvo la mayor sensibilidad, con un F2-score de 0.18 - 0.22.
- El modelo **LSTM** demostró una clara ventaja, alcanzando un **F1-score de 0.44**. Su arquitectura, diseñada para datos secuenciales, junto con una función de pérdida personalizada (DiceLoss), permitió un mejor equilibrio entre sensibilidad y especificidad, mostrando una tendencia de mejora progresiva durante el entrenamiento.

Finalmente, debido a restricciones de infraestructura (disponibilidad de GPU en el entorno de despliegue), se optó por desplegar el modelo **HGB**, ya que ofreció el mejor rendimiento entre los modelos de machine learning tradicionales y era computacionalmente más eficiente para el entorno de producción.

## Despliegue y Operacionalización

El proyecto se diseñó siguiendo las mejores prácticas de **MLOps** para garantizar la reproducibilidad, la trazabilidad y la colaboración.

- **Control de Versiones:** **Git** para el código y **DVC (Data Version Control)** para los datos y modelos.
- **Orquestación del Pipeline:** **DVC pipelines** (`dvc.yaml`) y `Makefile` para automatizar las etapas de preprocesamiento, entrenamiento, evaluación y predicción.
- **Seguimiento de Experimentos:** **MLflow** para registrar experimentos, parámetros, métricas y artefactos de los modelos.
- **Contenerización:** **Docker** para empaquetar la aplicación del modelo y el frontend, asegurando la consistencia entre los entornos de desarrollo y producción.
- **Infraestructura como Código (IaC):** **Terraform** para provisionar y gestionar la infraestructura en AWS (EC2, ECR).
- **Interfaz de Usuario:** Se desarrolló un **tablero interactivo** en Angular que permite al personal clínico monitorear las alertas de sepsis, visualizar el riesgo de los pacientes y gestionar la carga de trabajo de manera efectiva.

## Estructura del Proyecto

El repositorio está organizado para facilitar la reproducibilidad y la colaboración:

```
physionet-sepsis-forecasting/
├── app/              # Aplicación FastAPI para servir el modelo
├── aws/              # Scripts de despliegue en AWS (ECR)
├── data/             # Datos crudos, procesados y de inferencia (gestionados con DVC)
├── front/            # Código fuente del tablero en Angular
├── infra/            # Código de Terraform para la infraestructura
├── models/           # Modelos entrenados (gestionados con DVC)
├── notebooks/        # Jupyter notebooks para exploración y experimentación
├── results/          # Métricas de evaluación, gráficos y predicciones
├── scripts/          # Scripts de Python para el pipeline
├── src/              # Código fuente modularizado (feature engineering, métricas, etc.)
├── dvc.yaml          # Define las etapas del pipeline de DVC
├── Makefile          # Comandos para automatizar tareas comunes
├── Dockerfile        # Definición del contenedor para la aplicación
└── README.md         # Documentación general del proyecto
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
    ```bash
    dvc pull
    ```

4.  **Crea tu propia rama de trabajo:**
    **Nunca trabajes directamente sobre la rama `main`**.
    ```bash
    git checkout -b feature/nombre-descriptivo-de-tu-rama
    ```

5.  **Realiza tus cambios** en el código, notebooks o datos.

6.  **Reproduce el pipeline y versiona tus resultados:**
    ```bash
    dvc repro
    ```
    Guarda tus cambios con Git y DVC.
    ```bash
    git add .
    git commit -m "Descripción de tus cambios"
    ```

7.  **Sube tus cambios y crea un Pull Request:**
    ```bash
    dvc push
    git push -u origin feature/nombre-descriptivo-de-tu-rama
    ```