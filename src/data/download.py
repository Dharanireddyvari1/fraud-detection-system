# Kaggle datasets download script

import subprocess, zipfile
from pathlib import Path

def download_ieee_fraud():
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True) 
    print("Downloading the dataset - IEEE-CIS Fraud Detection ...")
    subprocess.run(["kaggle", "competitions", "download","-c",
                     "ieee-fraud-detection", "-p", str(data_dir)], check=True)
    
    zip_path = data_dir / "ieee-fraud-detection.zip"
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_dir)
    zip_path.unlink()  # Remove the zip file after extraction   
    
    for f in data_dir.iterdir():
        if f.is_file() and f.suffix == '.csv':
            print(f"Downloaded and extracted: {f.name}")

if __name__ == "__main__":
    download_ieee_fraud()