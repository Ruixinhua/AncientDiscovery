import logging
import math
import os
import zipfile
import shutil
from pathlib import Path

import requests
from tqdm import tqdm

from util import tools

log = logging.getLogger(__name__)
dataset_url = "https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/ksk47h2hsh-2.zip"


def maybe_download(url, filename=None, work_directory=".", expected_bytes=None):
    """Download a file if it is not already downloaded.

    Args:
        filename (str): File name.
        work_directory (str): Working directory.
        url (str): URL of the file to download.
        expected_bytes (int): Expected file size in bytes.

    Returns:
        str: File path of the file downloaded.
    """
    if filename is None:
        filename = url.split("/")[-1]
    os.makedirs(work_directory, exist_ok=True)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):

        r = requests.get(url, stream=True)
        total_size = int(r.headers.get("content-length", 0))
        block_size = 1024
        num_iterables = math.ceil(total_size / block_size)

        tools.print_log(f"Download dataset at {filepath}")
        with open(filepath, "wb") as file:
            for data in tqdm(r.iter_content(block_size), total=num_iterables, unit="KB", desc="Download"):
                file.write(data)
    else:
        log.info("File {} already downloaded".format(filepath))
    if expected_bytes is not None:
        statinfo = os.stat(filepath)
        if statinfo.st_size != expected_bytes:
            os.remove(filepath)
            raise IOError("Failed to verify {}".format(filepath))

    return filepath


def unzip_file(zip_src, dst_dir, clean_zip_file=True):
    """Unzip a file

    Args:
        zip_src (str): Zip file.
        dst_dir (str): Destination folder.
        clean_zip_file (bool): Whether or not to clean the zip file.
    """
    fz = zipfile.ZipFile(zip_src, "r")

    for file in tqdm(fz.namelist(), total=len(fz.namelist()), desc="Unzip"):
        fz.extract(file, dst_dir)
    fz.close()
    if clean_zip_file:
        os.remove(zip_src)


def download_dataset(data_path):
    """Download resources.

    Args:
        data_path (str): Path to download the resources.
    """
    os.makedirs(data_path, exist_ok=True)
    file_path = maybe_download(dataset_url, work_directory=data_path)
    unzip_file(file_path, data_path)
    unzip_file(os.path.join(data_path, "datasets.zip"), data_path)

