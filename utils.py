from __future__ import unicode_literals
from pathlib import Path
import numpy as np
import json
import librosa
import yt_dlp
import soundfile
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter


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
        ydl.extract_info(link, download=True)


def extract_vocal(filepath: str, target_filepath: str):
    separator = Separator("spleeter:2stems")
    # y, sr = librosa.core.load(filepath, sr=None)
    # if sr != 44100:
    #     y = librosa.core.resample(y=y, orig_sr=sr, target_sr=44100)
    # waveform = np.expand_dims(y, axis=1)

    audio_loader = AudioAdapter.default()
    sample_rate = 44100
    waveform, _ = audio_loader.load(filepath, sample_rate=sample_rate)
    prediction = separator.separate(waveform)
    vocal = librosa.core.to_mono(prediction["vocals"].T)
    vocal = np.clip(vocal, -1.0, 1.0)
    soundfile.write(target_filepath, vocal, 44100, subtype="PCM_16")
