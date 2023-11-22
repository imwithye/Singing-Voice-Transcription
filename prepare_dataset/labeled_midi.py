import os
from utils import (
    read_json,
    save_json,
    get_vocals_filepath,
    TRAIN_DATASET_DIR,
    VALID_DATASET_DIR,
    LABELS_JSON_FILE,
)
from tqdm import tqdm
from utils import notes2mid
from music21 import midi

LABELS_JSON = read_json(LABELS_JSON_FILE)


def get_midis():
    count = 0
    for i in tqdm(range(1, len(LABELS_JSON) + 1)):
        if str(i) not in LABELS_JSON:
            continue
        notes = LABELS_JSON[str(i)]
        vocals = get_vocals_filepath(i)
        if vocals is None or not os.path.exists(vocals):
            continue
        count += 1
        
        output = os.path.join(os.path.dirname(vocals), "Vocals.json")
        save_json(output, notes)

        output = os.path.join(os.path.dirname(vocals), "Vocals.mid")
        mid = notes2mid(notes)
        mid.save(output)
    print(f"Done, {count} songs midi generated in total")


if __name__ == "__main__":
    get_midis()
