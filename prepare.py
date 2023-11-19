import os
from utils import read_json, fetch_audio, extract_vocal
from joblib import Parallel, delayed
from tqdm import tqdm


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")


def _show_errors(results):
    count = 0
    for result in results:
        if result[0]:
            continue
        print(result[1])
        count += 1
    if count >= 0:
        print("Failed count: ", count)


def save_audio(name, link):
    output = os.path.join(DATASET_DIR, "original", name, "Mixture.mp3")
    if os.path.exists(output):
        return True, None
    try:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        fetch_audio(output, link)
        return True, None
    except:
        return False, f"Download failed {name}: {link}"


def extract_audio(name):
    fp = os.path.join(DATASET_DIR, "original", name, "Mixture.mp3")
    output = os.path.join(DATASET_DIR, "vocal", name, "Mixture.wav")
    if os.path.exists(output):
        return True, None
    try:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        extract_vocal(fp, output)
        return True, None
    except:
        return False, f"Download failed {name}: {fp}"


def main():
    links_json_file = os.path.join(PROJECT_DIR, "links.json")
    links_json = read_json(links_json_file)
    results = Parallel(n_jobs=4)(
        delayed(save_audio)(name, link) for name, link in tqdm(links_json.items())
    )
    _show_errors(results)
    results = Parallel(n_jobs=4)(
        delayed(extract_audio)(name) for name, link in tqdm(links_json.items())
    )
    _show_errors(results)


if __name__ == "__main__":
    main()
