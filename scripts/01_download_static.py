import subprocess
import os

repos = {
    "data/raw/statsbomb": "https://github.com/statsbomb/open-data.git",
    "data/raw/openfootball": "https://github.com/openfootball/world-cup.git",
}

for path, url in repos.items():
    if not os.path.exists(path):
        print(f"Cloning {url}...")
        subprocess.run(["git", "clone", "--depth=1", url, path])
        print(f"Done → {path}")
    else:
        print(f"Already exists, skipping: {path}")