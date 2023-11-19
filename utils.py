from __future__ import unicode_literals
import json
import yt_dlp


def read_json(json_file):
    with open(json_file, "r") as f:
        return json.load(f)


def save_json(json_file, data):
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)


def download_youtube(link):
    ydl_opts = {}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])
