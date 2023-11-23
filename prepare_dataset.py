if __name__ == "__main__":
    from prepare_dataset import prepare_dataset
    from utils import PROJECT_DIR, DATASET_DIR, LABELS_JSON_FILE

    print("Project directory: ", PROJECT_DIR)
    print("Dataset directory: ", DATASET_DIR)
    print("Labels file: ", LABELS_JSON_FILE)

    prepare_dataset()

    from net import AudioDataset
    from utils import TRAIN_DATASET_DIR, VALID_DATASET_DIR

    dataset = AudioDataset(data_dir=TRAIN_DATASET_DIR, limit=10)
    print("Train dataset length: ", len(dataset))
    dataset = AudioDataset(data_dir=VALID_DATASET_DIR, limit=10)
    print("Valid dataset length: ", len(dataset))
