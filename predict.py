import os
from tqdm import tqdm
from effnet import SeqDataset, EffNetPredictor
from utils import save_json, notes2mid, DEVICE, PROJECT_DIR, VALID_DATASET_DIR
from midi2audio import FluidSynth

MODEL_PATH = os.path.join(PROJECT_DIR, "effnet", "models", "1005_e_4")
MODELS_SAVE_DIR = os.path.join(PROJECT_DIR, "models")
fs = FluidSynth()

def predict(predictor, idx):
    cqt_feature_path = os.path.join(VALID_DATASET_DIR, str(idx), "CQT_feature.pt")
    test_dataset = SeqDataset(cqt_feature_path, str(idx))
    results = predictor.predict(
        test_dataset,
        onset_thres=0.4,
        offset_thres=0.5,
    )
    data = results[str(idx)]

    output = os.path.join(VALID_DATASET_DIR, str(idx), "Vocals_predict.json")
    save_json(output, data)
    mid = notes2mid(data)
    output = os.path.join(VALID_DATASET_DIR, str(idx), "Vocals_predict.mid")
    mid.save(output)
    wav_output = os.path.join(VALID_DATASET_DIR, str(idx), "Vocals_predict.wav")
    fs.midi_to_audio(output, wav_output)


if __name__ == "__main__":
    predictor = EffNetPredictor(device=DEVICE, model_path=MODEL_PATH)
    for the_dir in tqdm(os.listdir(VALID_DATASET_DIR)):
        predict(predictor, the_dir)
