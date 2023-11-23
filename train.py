import os
import torch
from utils import DEVICE, PROJECT_DIR, DATASET_DIR, TRAIN_DATASET_DIR, VALID_DATASET_DIR, LABELS_JSON_FILE

print("Project directory: ", PROJECT_DIR)
print("Dataset directory: ", DATASET_DIR)
print("Labels file: ", LABELS_JSON_FILE)

MODEL_PATH = os.path.join(PROJECT_DIR, "effnet", "models", "1005_e_4")
MODELS_SAVE_DIR = os.path.join(PROJECT_DIR, "models")

from effnet import EffNetPredictor, EffNetb0, ResNet18, WideResidualNet, EFFNET_STATE_DICT

def load_model(name):
    if name == "effnet":
        model = EffNetb0().to(DEVICE)
        model.load_state_dict(torch.load(EFFNET_STATE_DICT, map_location=DEVICE), strict=False)
    if name == "resnet":
        model = ResNet18().to(DEVICE)
    if name == "wideresnet":
        model = WideResidualNet().to(DEVICE)
    return model

predictor = EffNetPredictor(device=DEVICE, model=load_model("effnet"))
predictor.fit(
    train_dataset_dir=TRAIN_DATASET_DIR,
    valid_dataset_dir=VALID_DATASET_DIR,
    labels_filepath=LABELS_JSON_FILE,
    model_dir=MODELS_SAVE_DIR,
    batch_size=1024,
    valid_batch_size=200,
    epoch=10,
    lr=1e-4,
    save_every_epoch=1,
    save_prefix="effnet2",
    train_file_limit=100000,
    valid_file_limit=100000,
)
