import os
from effnet import SeqDataset, EffNetPredictor
from utils import save_json, DEVICE, PROJECT_DIR, VALID_DATASET_DIR

MODEL_PATH = os.path.join(PROJECT_DIR, "effnet", "models", "1005_e_4")
MODELS_SAVE_DIR = os.path.join(PROJECT_DIR, "models")

def predict(idx):
    assert idx > 400
    cqt_feature_path = os.path.join(VALID_DATASET_DIR, str(idx), "CQT_feature.pt")
    test_dataset = SeqDataset(cqt_feature_path, str(idx))

    predictor = EffNetPredictor(device=DEVICE, model_path=MODEL_PATH)
    results = predictor.predict(
        test_dataset,
        onset_thres=0.4,
        offset_thres=0.5,
    )
    data = results[str(idx)]

    output = os.path.join(VALID_DATASET_DIR, str(idx), "Vocals_predict.json")
    save_json(output, data)

if __name__ == "__main__":
    predict(401)