from utils import PROJECT_DIR, DATASET_DIR, LABELS_JSON_FILE

print("Project directory: ", PROJECT_DIR)
print("Dataset directory: ", DATASET_DIR)
print("Labels file: ", LABELS_JSON_FILE)

import os
from utils import TRAIN_DATASET_DIR, VALID_DATASET_DIR, LABELS_JSON_FILE

MODEL_PATH = os.path.join(PROJECT_DIR, "effnet", "models", "1005_e_4")
MODELS_SAVE_DIR = os.path.join(PROJECT_DIR, "models")

from effnet import EffNetPredictor
from utils import DEVICE

# predictor = EffNetPredictor(device=DEVICE, model_path=MODEL_PATH)
predictor = EffNetPredictor(device=DEVICE)
predictor.fit(
    train_dataset_dir=TRAIN_DATASET_DIR,
    valid_dataset_dir=VALID_DATASET_DIR,
    labels_filepath=LABELS_JSON_FILE,
    model_dir=MODELS_SAVE_DIR,
    batch_size=256,
    valid_batch_size=200,
    epoch=1,
    lr=1e-4,
    save_every_epoch=1,
    save_prefix="resnet",
    train_file_limit=100000,
    valid_file_limit=10,
)
