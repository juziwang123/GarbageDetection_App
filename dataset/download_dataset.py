import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_garbage_dataset():
    # Initialize Kaggle API
    api = KaggleApi()
    try:
        api.authenticate()
        print("Kaggle API authenticated successfully.")
    except Exception as e:
        print(f"Error authenticating with Kaggle API: {e}")
        print("Please ensure you have placed your 'kaggle.json' file in the correct location (e.g., ~/.kaggle/ or C:\\Users\\<User>\\.kaggle\\).")
        return

    dataset_name = "mostafaabla/garbage-classification"  # A popular garbage classification dataset
    download_path = "dataset"
    
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    print(f"Downloading dataset '{dataset_name}' to '{download_path}'...")
    try:
        api.dataset_download_files(dataset_name, path=download_path, unzip=False) # Download zip first
        print("Download complete. Unzipping...")
        
        # Unzip manually
        zip_path = os.path.join(download_path, "garbage-classification.zip")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(download_path)
            print("Unzip complete.")
            os.remove(zip_path) # Remove zip file
        else:
            print(f"Zip file not found at {zip_path}")
            
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    download_garbage_dataset()
