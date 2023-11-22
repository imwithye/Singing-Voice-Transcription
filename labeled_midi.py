import os
from utils import (
    read_json,
    save_json,
    TRAIN_DATASET_DIR,
    VALID_DATASET_DIR,
    LABELS_JSON_FILE,
)
from tqdm import tqdm
from utils import notes2mid

LABELS_JSON = read_json(LABELS_JSON_FILE)


def get_midis():
    count = 0
    for i in tqdm(range(1, len(LABELS_JSON) + 1)):
        if str(i) not in LABELS_JSON:
            continue
        notes = LABELS_JSON[str(i)]
        vocals = os.path.join(TRAIN_DATASET_DIR, str(i), "Vocals.wav")
        if os.path.exists(vocals):
            count += 1
            output = os.path.join(TRAIN_DATASET_DIR, str(i), "Vocals.mid")
            if not os.path.exists(output):
                mid = notes2mid(notes)
                mid.save(output)
            output = os.path.join(TRAIN_DATASET_DIR, str(i), "Vocals.json")
            if not os.path.exists(output):
                save_json(output, notes)
            continue
        vocals = os.path.join(VALID_DATASET_DIR, str(i), "Vocals.wav")
        if os.path.exists(vocals):
            count += 1
            output = os.path.join(VALID_DATASET_DIR, str(i), "Vocals.mid")
            if not os.path.exists(output):
                mid = notes2mid(notes)
            output = os.path.join(VALID_DATASET_DIR, str(i), "Vocals.json")
            if not os.path.exists(output):
                save_json(output, notes)
            continue
    print(f"Done, {count} songs midi generated in total")


if __name__ == "__main__":
    get_midis()
