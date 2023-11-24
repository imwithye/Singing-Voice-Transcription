import os
import shutil
from utils import read_json, DATASET_DIR, LINKS_JSON_FILE
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path
import yt_dlp

LINKS = read_json(LINKS_JSON_FILE)


def fetch_audio(filepath: str, link: str):
    class Logger(object):
        def debug(self, msg):
            pass

        def warning(self, msg):
            pass

        def error(self, msg):
            pass

    fp = Path(filepath)
    ydl_opts = {
        "outtmpl": str(fp.with_suffix("")) + ".%(ext)s",
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "320",
            }
        ],
        "logger": Logger(),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])


def get_youtube(idx, dir):
    output = os.path.join(DATASET_DIR, dir, str(idx), "Mixture.mp3")
    if os.path.exists(output):
        return True, None
    try:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        link = LINKS[str(idx)]
        fetch_audio(output, link)
        return True, None
    except:
        return False, f"Download failed for idx {idx}: {link}"


def cleanup(dir):
    for idx in range(1, len(LINKS) + 1):
        output = os.path.join(DATASET_DIR, dir, str(idx), "Mixture.mp3")
        parent = os.path.dirname(output)
        if not os.path.exists(output) and os.path.exists(parent):
            shutil.rmtree(parent)


def get_audios():
    print("Downloading training set")
    # song #1~#400 are training set
    results = Parallel(n_jobs=4)(
        delayed(get_youtube)(idx, "train") for idx in tqdm(range(1, 401))
    )
    failed = [r for r in results if not r[0]]
    print(
        f"Failed to download {len(failed)} songs, {len(results) - len(failed)} songs downloaded"
    )
    for f in failed:
        print(f[1])

    print("Downloading valid set")
    # song #401~ are valid set
    results = Parallel(n_jobs=4)(
        delayed(get_youtube)(idx, "valid") for idx in tqdm(range(401, len(LINKS) + 1))
    )
    failed = [r for r in results if not r[0]]
    print(
        f"Failed to download {len(failed)} songs, {len(results) - len(failed)} songs downloaded"
    )
    for f in failed:
        print(f[1])

    print("Cleaning up")
    # remove empty directories
    cleanup("train")
    cleanup("valid")


if __name__ == "__main__":
    get_audios()
