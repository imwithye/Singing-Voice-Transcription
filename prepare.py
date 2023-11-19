import os
from utils import read_json, fetch_audio
from joblib import Parallel, delayed
from tqdm import tqdm


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")


def save_audio(name, link):
    output = os.path.join(DATASET_DIR, "original", name, "Mixture.mp3")
    if os.path.exists(output):
        return name, link, True
    try:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        fetch_audio(output, link)
        return name, link, True
    except:
        return name, link, False


def main():
    links_json_file = os.path.join(PROJECT_DIR, "links.json")
    links_json = read_json(links_json_file)
    results = Parallel(n_jobs=4)(
        delayed(save_audio)(name, link) for name, link in tqdm(links_json.items())
    )
    count = 0
    for result in results:
        if result[2]:
            continue
        print("Dowload failed: ", result[0], result[1])
        count += 1
    print("Failed count: ", count)


if __name__ == "__main__":
    main()
