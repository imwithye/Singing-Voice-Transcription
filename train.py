import os
import argparse
from net import Predictor
from utils import (
    DEVICE,
    PROJECT_DIR,
    DATASET_DIR,
    TRAIN_DATASET_DIR,
    VALID_DATASET_DIR,
    LABELS_JSON_FILE,
    load_model,
)

print("Project directory: ", PROJECT_DIR)
print("Dataset directory: ", DATASET_DIR)
print("Labels file: ", LABELS_JSON_FILE)

MODEL_PATH = os.path.join(PROJECT_DIR, "effnet", "models", "1005_e_4")
MODELS_SAVE_DIR = os.path.join(PROJECT_DIR, "models")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--net")
    parser.add_argument("--train-file-limit")
    parser.add_argument("--valid-file-limit")
    args = parser.parse_args()

    net = "effnet" if args.net is None else args.net
    train_file_limit = (
        1000000 if args.train_file_limit is None else int(args.train_file_limit)
    )
    valid_file_limit = (
        1000000 if args.valid_file_limit is None else int(args.valid_file_limit)
    )
    print()
    print("Net:", net)
    print("Train File Limit:", train_file_limit)
    print("Valid File Limit:", valid_file_limit)
    print()

    predictor = Predictor(device=DEVICE, model=load_model(net))
    predictor.fit(
        train_dataset_dir=TRAIN_DATASET_DIR,
        valid_dataset_dir=VALID_DATASET_DIR,
        model_dir=MODELS_SAVE_DIR,
        batch_size=512,
        valid_batch_size=200,
        epoch=10,
        lr=1e-4,
        save_every_epoch=1,
        save_prefix=net,
        train_file_limit=train_file_limit,
        valid_file_limit=valid_file_limit,
    )
