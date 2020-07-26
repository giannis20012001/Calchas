# Modules import
import os
import gzip
import glob
import shutil
import requests
from os import listdir
from os.path import isfile, join


# ======================================================================================================================
# Functions space
# ======================================================================================================================
def download(url: str, dest_folder: str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
    file_path = os.path.join(dest_folder, filename)

    r = requests.get(url, stream=True)
    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))


def extract_all_files_to_dir(source_dir: str, dest_dir: str):
    for src_name in glob.glob(os.path.join(source_dir, '*.gz')):
        base = os.path.basename(src_name)
        dest_name = os.path.join(dest_dir, base[:-3])
        with gzip.open(src_name, 'rb') as infile:
            with open(dest_name, 'wb') as outfile:
                for line in infile:
                    outfile.write(line)

        print('Added: ' + dest_name)


def extract_file_to_dir(source_file: str, dest_file: str):
    with gzip.open(source_file, 'rb') as f_in:
        with open(dest_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    print('Added: ' + dest_file)


# ======================================================================================================================
# Main function
# ======================================================================================================================
def main():
    # First step download raw data
    # Download the file from `url`, save it in a temporary directory and get the
    year = 2002
    while year <= 2020:
        # path to it in the `file_name` variable:
        file_name = "nvdcve-1.1-" + str(year) + ".json.gz"
        url = "https://nvd.nist.gov/feeds/json/cve/1.1/" + file_name
        # Download the file from `url` and save it locally under `file_name` if it does not exist:
        if not os.path.isfile('data/raw_data/' + file_name):
            download(url, dest_folder='data/raw_data/')
        year = year + 1

    # Second step extract data from .gz files
    file_names = [f for f in listdir('data/json_data') if isfile(join('data/json_data', f))]
    if not file_names:
        extract_all_files_to_dir('data/raw_data/', 'data/json_data/')
    else:
        expected_file_names = [f for f in listdir('data/raw_data') if isfile(join('data/raw_data', f))]
        expected_file_names = [s.strip('.gz') for s in expected_file_names]
        for file in expected_file_names:
            if not os.path.isfile('data/json_data/' + file):
                extract_file_to_dir('data/raw_data/' + file + '.gz', 'data/json_data/' + file)


if __name__ == "__main__":
    main()
