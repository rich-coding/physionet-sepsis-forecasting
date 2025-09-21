import os
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # Barra de progreso para descargas

"""
Script para descargar todos los archivos *.psv de los conjuntos de datos
training_setA y training_setB del PhysioNet Challenge 2019.

Los archivos se guardan en las carpetas:
    r"c:\repos\physionet-sepsis-forecasting\data\training_setA"
    r"c:\repos\physionet-sepsis-forecasting\data\training_setB"

Si las carpetas no existen, el script las crea automáticamente.

Requiere: requests, beautifulsoup4, tqdm
Instalar con: pip install requests beautifulsoup4 tqdm
"""

# URLs de los conjuntos de datos en PhysioNet
URLS = {
    "training_setA": "https://physionet.org/files/challenge-2019/1.0.0/training/training_setA/",
    "training_setB": "https://physionet.org/files/challenge-2019/1.0.0/training/training_setB/"
}

# Carpetas locales donde se guardarán los archivos descargados
DEST_DIRS = {
    "training_setA": r"c:\repos\physionet-sepsis-forecasting\data\training_setA",
    "training_setB": r"c:\repos\physionet-sepsis-forecasting\data\training_setB"
}

def crear_carpeta_si_no_existe(path):
    """
    Crea la carpeta especificada si no existe.

    Args:
        path (str): Ruta de la carpeta a crear.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def obtener_lista_archivos_psv(url):
    """
    Obtiene la lista de archivos .psv disponibles en la URL dada.

    Args:
        url (str): URL de la página web que contiene los enlaces a los archivos .psv.

    Returns:
        list: Lista de nombres de archivos .psv encontrados en la página.
    """
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    archivos = []
    for enlace in soup.find_all("a"):
        href = enlace.get("href", "")
        if href.endswith(".psv"):
            archivos.append(href)
    return archivos

def descargar_archivo(url, destino):
    """
    Descarga un archivo desde la URL al destino especificado.

    Args:
        url (str): URL del archivo a descargar.
        destino (str): Ruta local donde se guardará el archivo.

    Returns:
        tuple: (ruta_destino, None) si la descarga fue exitosa,
               (ruta_destino, error) si hubo algún error.
    """
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        with open(destino, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return (destino, None)
    except Exception as e:
        return (destino, str(e))

def main():
    """
    Función principal del script.
    Descarga todos los archivos .psv de los conjuntos de datos especificados,
    mostrando una barra de progreso y un resumen al finalizar.
    """
    # Número de hilos para descargas concurrentes
    max_workers = 20
    for nombre, url in URLS.items():
        destino = DEST_DIRS[nombre]
        crear_carpeta_si_no_existe(destino)
        print(f"Descargando archivos de {nombre}...")
        archivos = obtener_lista_archivos_psv(url)
        tareas = []
        resultados = []
        # Descarga concurrente de archivos
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for archivo in archivos:
                url_archivo = url + archivo
                ruta_destino = os.path.join(destino, archivo)
                # Solo descarga si el archivo no existe localmente
                if not os.path.exists(ruta_destino):
                    tareas.append(executor.submit(descargar_archivo, url_archivo, ruta_destino))
            # Barra de progreso para descargas
            with tqdm(total=len(tareas), desc=f"{nombre}", unit="archivo") as pbar:
                for future in as_completed(tareas):
                    ruta, error = future.result()
                    archivo = os.path.basename(ruta)
                    if error:
                        resultados.append((archivo, "ERROR"))
                    else:
                        resultados.append((archivo, "OK"))
                    pbar.update(1)
        # Resumen de descargas
        print(f"Descarga de {nombre} completada.\nResumen:")
        ok = sum(1 for _, status in resultados if status == "OK")
        err = sum(1 for _, status in resultados if status == "ERROR")
        print(f"  Archivos descargados: {ok}")
        print(f"  Fallidos: {err}\n")

if __name__ == "__main__":
    main()