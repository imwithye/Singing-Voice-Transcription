import os

PROJECT_DIR = os.getcwd()
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
LABELS_JSON_FILE= os.path.join(DATASET_DIR, "labeled.json")
print("Project directory: ", PROJECT_DIR)
print("Dataset directory: ", DATASET_DIR)
print("Labels file: ", LABELS_JSON_FILE)

from get_youtube import get_audios
from do_spleeter import get_vocals
from labeled_midi import get_midis

# Get audio from youtube
get_audios()
# Get vocals from audio
get_vocals()
# Get midi from labeled vocals
get_midis()

from baseline import AudioDataset
from utils import save_pkl

print("Generating dataset in pkl file...")

target_path = os.path.join(DATASET_DIR, "train.pkl")
if not os.path.exists(target_path):
    dataset = AudioDataset(gt_path=LABELS_JSON_FILE, data_dir=os.path.join(DATASET_DIR, "train"))
    save_pkl(target_path, dataset)

target_path = os.path.join(DATASET_DIR, "valid.pkl")
if not os.path.exists(target_path):
    dataset = AudioDataset(gt_path=LABELS_JSON_FILE, data_dir=os.path.join(DATASET_DIR, "valid"))
    save_pkl(target_path, dataset)
