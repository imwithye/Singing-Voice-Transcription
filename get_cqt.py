import os
import torch
import librosa
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from utils import read_json, get_vocals_filepath, LINKS_JSON_FILE

LINKS = read_json(LINKS_JSON_FILE)


def compute_cqt(idx):
    input = get_vocals_filepath(idx)
    if input is None or not os.path.exists(input):
        return
    output = os.path.join(os.path.dirname(input), "CQT.pt")
    if os.path.exists(output):
        return

    y, sr = librosa.core.load(input, sr=None, mono=True)
    if sr != 44100:
        y = librosa.core.resample(y=y, orig_sr=sr, target_sr=44100)
    y = librosa.util.normalize(y)
    cqt_feature = np.abs(
        librosa.cqt(
            y,
            sr=44100,
            hop_length=1024,
            fmin=librosa.midi_to_hz(36),
            n_bins=84 * 2,
            bins_per_octave=12 * 2,
            filter_scale=1.0,
        )
    ).T
    cqt = torch.tensor(cqt_feature, dtype=torch.float).unsqueeze(1)
    torch.save(cqt, output)


def get_cqt():
    print("Computing CQT")
    Parallel(n_jobs=4)(
        delayed(compute_cqt)(idx) for idx in tqdm(range(1, len(LINKS) + 1))
    )


if __name__ == "__main__":
    get_cqt()
