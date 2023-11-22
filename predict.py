import os
from tqdm import tqdm
from effnet import SeqDataset, EffNetPredictor
from utils import read_json, save_json, notes2mid, DEVICE, PROJECT_DIR, VALID_DATASET_DIR

MODEL_PATH = os.path.join(PROJECT_DIR, "effnet", "models", "1005_e_4")
MODELS_SAVE_DIR = os.path.join(PROJECT_DIR, "models")


def predict(predictor, idx):
    output = os.path.join(VALID_DATASET_DIR, str(idx), "Vocals_predict.json")
    if os.path.exists(output):
        return read_json(output)

    cqt_feature_path = os.path.join(VALID_DATASET_DIR, str(idx), "CQT_feature.pt")
    test_dataset = SeqDataset(cqt_feature_path, str(idx))
    results = predictor.predict(
        test_dataset,
        onset_thres=0.4,
        offset_thres=0.5,
    )
    data = results[str(idx)]
    save_json(output, data)
    mid = notes2mid(data)
    output = os.path.join(VALID_DATASET_DIR, str(idx), "Vocals_predict.mid")
    mid.save(output)

    return data


if __name__ == "__main__":
    results = {}
    predictor = EffNetPredictor(device=DEVICE, model_path=MODEL_PATH)
    for the_dir in tqdm(os.listdir(VALID_DATASET_DIR)):
        result = predict(predictor, the_dir)
        results[the_dir] = result
    save_json("predict.json", results)
