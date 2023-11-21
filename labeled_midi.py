import os
from utils import read_json, save_json
from tqdm import tqdm
from utils import notes2mid

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")

LABELED_JSON_FILE = os.path.join(DATASET_DIR, "labeled.json")
LABELED_JSON = read_json(LABELED_JSON_FILE)


def get_midis():
    count = 0
    for i in tqdm(range(1, len(LABELED_JSON) + 1)):
        if str(i) not in LABELED_JSON:
            continue
        notes = LABELED_JSON[str(i)]
        vocals = os.path.join(DATASET_DIR, "train", str(i), "Vocals.wav")
        if os.path.exists(vocals):
            count += 1
            output = os.path.join(DATASET_DIR, "train", str(i), "Vocals.mid")
            if not os.path.exists(output):
                mid = notes2mid(notes)
                mid.save(output)
            output = os.path.join(DATASET_DIR, "train", str(i), "Vocals.json")
            if not os.path.exists(output):
                save_json(output, notes)
            continue
        vocals = os.path.join(DATASET_DIR, "valid", str(i), "Vocals.wav")
        if os.path.exists(vocals):
            count += 1
            output = os.path.join(DATASET_DIR, "valid", str(i), "Vocals.mid")
            if not os.path.exists(output):
                mid = notes2mid(notes)
            output = os.path.join(DATASET_DIR, "valid", str(i), "Vocals.json")
            if not os.path.exists(output):
                save_json(output, notes)
            continue
    print(f"Done, {count} songs midi generated in total")


if __name__ == "__main__":
    get_midis()
