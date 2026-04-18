"""
Download only the data files needed from each dataset.
Run from the project root:  python data/download.py
Re-running is safe — already-downloaded files are skipped.
"""

import os
import subprocess
import sys
import urllib.request
import json

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")


# Each entry: (url, destination filename)
# Using raw GitHub URLs to grab only the data files
DATASETS = {
    "mustard": [
        (
            "https://raw.githubusercontent.com/soujanyaporia/MUStARD/master/data/sarcasm_data.json",
            "sarcasm_data.json",
        ),
    ],
    "sarcasm_v2": [
        # This repo stores data in a less standard way — clone is easier
        # We do a sparse checkout of just the data folder
    ],
    "isarcasmeval": [
        (
            "https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/train/train.En.csv",
            "train.En.csv",
        ),
        (
            "https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/test/task_A_En_test.csv",
            "task_A_En_test.csv",
        ),
    ],
    "csc": [
        (
            "https://raw.githubusercontent.com/CoPsyN/CSC/main/data_full.csv",
            "data_full.csv",
        ),
    ],
    "sarc": [
        # SARC data is not in the GitHub repo itself — it must be
        # downloaded separately. The repo only has evaluation code.
        # See: https://nlp.cs.princeton.edu/SARC/2.0/
    ],
    "news_headlines": [
        (
            "https://raw.githubusercontent.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection/master/Sarcasm_Headlines_Dataset.json",
            "Sarcasm_Headlines_Dataset.json",
        ),
    ],
}

# Datasets that need special handling (clone or external download)
CLONE_REPOS = {
    "sarcasm_v2": "https://github.com/soraby/sarcasm2",
}


def download_file(url, dest):
    if os.path.isfile(dest):
        print(f"  [skip]  {os.path.basename(dest)} already exists")
        return
    print(f"  [fetch] {os.path.basename(dest)}")
    try:
        tmp = dest + ".tmp"
        urllib.request.urlretrieve(url, tmp)
        # Catch silently-saved 404 HTML pages
        with open(tmp, "rb") as f:
            if f.read(15).lower().startswith(b"<!doctype html"):
                os.remove(tmp)
                raise ValueError(f"got HTML response — check URL: {url}")
        os.rename(tmp, dest)
    except Exception as e:
        print(f"  [error] {e}", file=sys.stderr)
        sys.exit(1)


def clone_repo(name, url):
    """Shallow-clone a repo when direct file download isn't practical."""
    dest = os.path.join(RAW_DIR, name)
    if os.path.isdir(dest):
        print(f"  [skip]  {name} repo already cloned")
        return
    print(f"  [clone] {name} <- {url}")
    result = subprocess.run(["git", "clone", "--depth", "1", url, dest])
    if result.returncode != 0:
        print(f"  [error] failed to clone {name}", file=sys.stderr)
        sys.exit(1)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    # Download individual data files
    for name, files in DATASETS.items():
        if not files:
            continue
        dest_dir = os.path.join(RAW_DIR, name)
        os.makedirs(dest_dir, exist_ok=True)
        print(f"[{name}]")
        for url, filename in files:
            download_file(url, os.path.join(dest_dir, filename))

    # Clone repos where direct download isn't practical
    for name, url in CLONE_REPOS.items():
        print(f"[{name}]")
        clone_repo(name, url)

    # Special note for SARC
    sarc_dir = os.path.join(RAW_DIR, "sarc")
    os.makedirs(sarc_dir, exist_ok=True)
    sarc_readme = os.path.join(sarc_dir, "README.md")
    if not os.path.isfile(sarc_readme):
        with open(sarc_readme, "w") as f:
            f.write(
                "# SARC Dataset\n\n"
                "The SARC 2.0 data files must be downloaded manually from:\n"
                "https://nlp.cs.princeton.edu/SARC/2.0/\n\n"
                "Download the following files into this directory:\n"
                "- train-balanced.csv.bz2\n"
                "- test-balanced.csv.bz2\n"
                "- comments.json.bz2\n"
            )
        print("[sarc]")
        print("  [note]  SARC requires manual download — see data/raw/sarc/README.md")

    print("\nAll datasets downloaded. Next step: run data/preprocessing.py")


if __name__ == "__main__":
    main()