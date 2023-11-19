from __future__ import unicode_literals
from pathlib import Path
import json
import yt_dlp
from tqdm import tqdm


def read_json(json_file: str):
    with open(json_file, "r") as f:
        return json.load(f)


def save_json(json_file: str, data: dict):
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)


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
        "logger": Logger(),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(link, download=True)
