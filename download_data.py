import tempfile
import gdown
import shutil
import os

def download_file_gdrive(url, download_path):
    """Download a file from Google Drive."""
    print(f"Downloading {url} to {download_path}")
    with tempfile.NamedTemporaryFile() as tmp_file:
        tmp_file_path = tmp_file.name
        gdown.download(url, tmp_file_path, quiet=True, fuzzy=True)
        assert os.path.getsize(tmp_file_path) > 0, f"Downloaded file is empty!"
        shutil.copy(tmp_file_path, download_path)



download_url = "https://drive.google.com/file/d/1jZQAWiii-DGwTqcQI6wehvsaxMB1AWSg/view?usp=sharing"
download_path = "./data/raw_data/0.txt"

download_file_gdrive(download_url, download_path)