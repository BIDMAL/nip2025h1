import os
import requests
import yaml

def load_config(config_file):
    config = None

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    return config

def download_file(source_path, save_path):
    # Send a GET request to the URL
    response = requests.get(source_path, stream=True)
    response.raise_for_status()  # Raise an error for bad status codes

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Write the file to the specified path
    with open(save_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

def download_sources(config):
    source_path = config['remote_sources'][0] + config['remote_filenames'][0]
    save_path = config['local_path'] + config['remote_filenames'][0]
    download_file(source_path, save_path)

if __name__ == "__main__":
    config = load_config('config.yaml')
    download_sources(config)