from pathlib import Path
import kagglehub
import pandas as pd

def download_galaxy_zoo_dataset() -> Path:
    path = kagglehub.dataset_download("jaimetrickz/galaxy-zoo-2-images")
    return Path(path)

def load_mapping_and_paths(dataset_path: Path):
    images_dir = dataset_path / "images_gz2" / "images"
    mapping_file = dataset_path / "gz2_filename_mapping.csv"

    mapping = pd.read_csv(mapping_file)
    return mapping, images_dir, mapping_file

def load_gz2_table():
    gz2_url = "https://zooniverse-data.s3.amazonaws.com/galaxy-zoo-2/zoo2MainSpecz.csv.gz"
    gz2 = pd.read_csv(gz2_url, compression="gzip")
    return gz2