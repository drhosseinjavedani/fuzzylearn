import os
from io import StringIO

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

load_dotenv()


def read_data_from_gdrive_or_local(
    name_on_env, fixed_part="https://drive.google.com/uc?export=download&id="
):
    shared_link = os.getenv(name_on_env)
    if "http://" in name_on_env:
        file_id = shared_link.split("/")[-2]
        download_url = fixed_part + file_id
        url2 = requests.get(download_url).text
        csv_raw = StringIO(url2)
        data = pd.read_csv(csv_raw)
    else:
        data = pd.read_csv(shared_link)
    return data


def read_yaml_file(file_path):
    with open(file_path, "r") as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error while reading YAML file: {e}")
            return None
