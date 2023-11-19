import os
from utils import read_json, fetch_audio


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")


def main():
    links_json_file = os.path.join(PROJECT_DIR, "links.json")
    links_json = read_json(links_json_file)
    for name, link in links_json.items():
        output = os.path.join(DATASET_DIR, "Mixture", name, "Mixture.mp3")
        if os.path.exists(output):
            continue
        os.makedirs(os.path.dirname(output), exist_ok=True)
        fetch_audio(output, link)


if __name__ == "__main__":
    main()
