import os
import json
import mido
import torch
from music21 import midi

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
TRAIN_DATASET_DIR = os.path.join(DATASET_DIR, "train")
VALID_DATASET_DIR = os.path.join(DATASET_DIR, "valid")
LINKS_JSON_FILE = os.path.join(DATASET_DIR, "links.json")
LABELS_JSON_FILE = os.path.join(DATASET_DIR, "labels.json")


def read_json(json_file: str):
    with open(json_file, "r") as f:
        return json.load(f)


def save_json(json_file: str, data: dict):
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)


def music21_from_midi(fp):
    mf = midi.MidiFile()
    mf.open(fp)
    mf.read()
    mf.close()
    s = midi.translate.midiFileToStream(mf)
    return s


def get_vocals_filepath(idx):
    train = os.path.join(TRAIN_DATASET_DIR, str(idx), "Vocals.wav")
    if os.path.exists(train):
        return train
    valid = os.path.join(VALID_DATASET_DIR, str(idx), "Vocals.wav")
    if os.path.exists(valid):
        return valid
    return None


def get_midi_filepath(idx):
    train = os.path.join(TRAIN_DATASET_DIR, str(idx), "Vocals.mid")
    if os.path.exists(train):
        return train
    valid = os.path.join(VALID_DATASET_DIR, str(idx), "Vocals.mid")
    if os.path.exists(valid):
        return valid
    return None


def get_cqt_filepath(idx):
    train = os.path.join(TRAIN_DATASET_DIR, str(idx), "CQT.pt")
    if os.path.exists(train):
        return train
    valid = os.path.join(VALID_DATASET_DIR, str(idx), "CQT.pt")
    if os.path.exists(valid):
        return valid
    return None


def notes2mid(notes):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    mid.ticks_per_beat = 480
    new_tempo = mido.bpm2tempo(120.0)

    track.append(mido.MetaMessage("set_tempo", tempo=new_tempo))
    track.append(mido.Message("program_change", program=0, time=0))

    cur_total_tick = 0

    for note in notes:
        if note[2] == 0:
            continue
        note[2] = int(round(note[2]))

        ticks_since_previous_onset = int(
            mido.second2tick(note[0], ticks_per_beat=480, tempo=new_tempo)
        )
        ticks_current_note = int(
            mido.second2tick(note[1] - 0.0001, ticks_per_beat=480, tempo=new_tempo)
        )
        note_on_length = ticks_since_previous_onset - cur_total_tick
        note_off_length = ticks_current_note - note_on_length - cur_total_tick

        track.append(
            mido.Message("note_on", note=note[2], velocity=100, time=note_on_length)
        )
        track.append(
            mido.Message("note_off", note=note[2], velocity=100, time=note_off_length)
        )
        cur_total_tick = cur_total_tick + note_on_length + note_off_length

    return mid
