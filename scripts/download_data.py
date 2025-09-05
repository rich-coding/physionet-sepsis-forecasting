import os
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

BASE_URLS = {
    "A": "https://physionet.org/files/challenge-2019/1.0.0/training/training_setA/",
    "B": "https://physionet.org/files/challenge-2019/1.0.0/training/training_setB/"
}
RANGES = {
    "A": range(1, 20644),  # p000001 a p020643
    "B": range(100001, 120001) # p100001 a p120000
}
OUTPUT_DIR = "data/raw"

def download_file(url, local_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True, url
    except requests.exceptions.HTTPError as e:
        return False, url # File likely doesn't exist (404)

def main():
    print("Starting data download from PhysioNet...")
    
    # Decidamos qué set descargar.
    set_to_download = "A"  # Cambiar a "B" para descargar el otro set.

    target_dir = os.path.join(OUTPUT_DIR, f"training_set{set_to_download}")
    os.makedirs(target_dir, exist_ok=True)
    
    base_url = BASE_URLS[set_to_download]
    patient_range = RANGES[set_to_download]
    
    prefix = 'p0' if set_to_download == 'A' else 'p'
    
    tasks = []
    for i in patient_range:
        patient_id = f"{prefix}{str(i).zfill(6)}.psv"
        file_url = f"{base_url}{patient_id}"
        local_file_path = os.path.join(target_dir, patient_id)
        if not os.path.exists(local_file_path):
            tasks.append((file_url, local_file_path))

    if not tasks:
        print(f"All files for set {set_to_download} seem to be already downloaded.")
        return

    print(f"Found {len(tasks)} files to download for training set {set_to_download}...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(lambda p: download_file(*p), tasks), total=len(tasks)))
    
    success_count = sum(1 for success, url in results if success)
    fail_count = len(results) - success_count
    
    print(f"\n[DONE] Download complete.")
    print(f"Successfully downloaded: {success_count} files.")
    if fail_count > 0:
        print(f"Failed or missing files (404 Not Found): {fail_count}")

if __name__ == "__main__":
    main()