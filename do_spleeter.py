import os
from utils import read_json
import librosa
import numpy as np
import soundfile
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")

LINKS_JSON_FILE = os.path.join(PROJECT_DIR, "links.json")
LINKS = read_json(LINKS_JSON_FILE)

separator = Separator("spleeter:2stems")


def do_spleeter(filepath):
    dirname = os.path.dirname(filepath)
    output = os.path.join(dirname, "Vocals.mp3")
    y, sr = librosa.core.load(filepath, sr=None, mono=True)
    if sr != 44100:
        y = librosa.core.resample(y=y, orig_sr=sr, target_sr=44100)
    waveform = np.expand_dims(y, axis=1)
    prediction = separator.separate(waveform)
    vocal = librosa.core.to_mono(prediction["vocals"].T)
    vocal = np.clip(vocal, -1.0, 1.0)
    soundfile.write(output, vocal, 44100, subtype="PCM_16")


if __name__ == "__main__":
    for idx in range(1, len(LINKS) + 1):
        train = os.path.join(DATASET_DIR, "train", str(idx), "Mixture.mp3")
        if os.path.exists(train):
            do_spleeter(train)
        valid = os.path.join(DATASET_DIR, "valid", str(idx), "Mixture.mp3")
        if os.path.exists(valid):
            do_spleeter(valid)
