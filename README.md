# Predicción Temprana de Sepsis con Machine Learning

Este proyecto tiene como objetivo desarrollar un modelo de machine learning para la predicción temprana de sepsis en pacientes de unidades de cuidados intensivos (UCI). El desarrollo se basa en los datos y la problemática definidos en el [PhysioNet/Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/). 

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

## Instalación

Para ejecutar este proyecto, necesitarás Python 3.8+ y las siguientes librerías. Puedes instalarlas usando `pip`:

```bash
pip install pandas numpy scikit-learn tensorflow