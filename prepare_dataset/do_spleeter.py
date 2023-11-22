import os
from utils import read_json, TRAIN_DATASET_DIR, VALID_DATASET_DIR, LINKS_JSON_FILE
import librosa
import numpy as np
import soundfile
from tqdm import tqdm
from spleeter.separator import Separator

LINKS = read_json(LINKS_JSON_FILE)

separator = Separator("spleeter:2stems")


def do_spleeter(filepath, output):
    y, sr = librosa.core.load(filepath, sr=None, mono=True)
    if sr != 44100:
        y = librosa.core.resample(y=y, orig_sr=sr, target_sr=44100)
    waveform = np.expand_dims(y, axis=1)
    prediction = separator.separate(waveform)
    voc = librosa.core.to_mono(prediction["vocals"].T)
    voc = np.clip(voc, -1.0, 1.0)
    soundfile.write(output, voc, 44100, subtype="PCM_16")


def get_vocals():
    count = 0
    for idx in tqdm(range(1, len(LINKS) + 1)):
        input = os.path.join(TRAIN_DATASET_DIR, str(idx), "Mixture.mp3")
        if os.path.exists(input):
            count += 1
            dirname = os.path.dirname(input)
            output = os.path.join(dirname, "Vocals.wav")
            if not os.path.exists(output):
                do_spleeter(input, output)
        input = os.path.join(VALID_DATASET_DIR, str(idx), "Mixture.mp3")
        if os.path.exists(input):
            count += 1
            dirname = os.path.dirname(input)
            output = os.path.join(dirname, "Vocals.wav")
            if not os.path.exists(output):
                do_spleeter(input, output)
    print(f"Done, {count} songs spleeted in total")


if __name__ == "__main__":
    get_vocals()
