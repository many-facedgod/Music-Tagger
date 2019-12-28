from __future__ import print_function

import hashlib
import requests
import shutil

from os import mkdir, remove
from os.path import join, isdir, isfile
from zipfile import ZipFile

mp3_urls = ['http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.001',
            'http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.002',
            'http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.003']

tags_url = 'http://mi.soi.city.ac.uk/datasets/magnatagatune/annotations_final.csv'

data_path = '../data'

mp3_hash = 'f12add0eb4cd6c2a3ab7e4afaf7b7467'
tags_hash = 'f04fa01752a8cc64f6e1ca142a0fef1d'

mp3_file = join(data_path, 'mp3.zip')
tags_file = join(data_path, 'tags.csv')


def download_file(url, file):
    """Downloads a file from the url into the directory."""
    print(url)
    with requests.get(url, stream=True) as r:
        shutil.copyfileobj(r.raw, file)


def md5(filename):
    with open(filename, "rb") as f:
        file_hash = hashlib.md5()
        chunk = f.read(1048576 * 50)  # 50 MB
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(10485760 * 50)
        return file_hash.hexdigest()


def download_data():
    """Download the MagnaTagATune Dataset"""
    if not isdir(data_path):
        mkdir(data_path)
    print('Downloading data...')
    if isfile(mp3_file) and md5(mp3_file) == mp3_hash:
        print('Already exists')
    else:
        with open(mp3_file, 'wb') as f:
            for url in mp3_urls:
                download_file(url, f)
        assert md5(mp3_file) == mp3_hash
    print('Downloading tags...')
    if isfile(tags_file) and md5(tags_file) == tags_hash:
        print('Already exists')
    else:
        with open(tags_file, 'wb') as f:
            download_file(tags_url, f)
        assert md5(tags_file) == tags_hash


def unzip_data():
    """Unzip the downloaded data."""
    print('Unzipping files')
    with ZipFile(mp3_file) as z:
        z.extractall(data_path)
    remove(mp3_file)


def main():
    download_data()
    unzip_data()


if __name__ == '__main__':
    main()
