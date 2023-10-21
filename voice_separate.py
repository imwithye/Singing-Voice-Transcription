import os
import librosa
import soundfile
import numpy as np
from spleeter.separator import Separator
from tqdm import tqdm

if __name__ == '__main__':

    dataset_dir = "./MIR-ST500/songs"

    separator = Separator('spleeter:2stems')

    for i, the_dir in tqdm(enumerate(os.listdir(dataset_dir), 1)):
        # jump over processed songs
        if i <= 29:
            continue
        
        print("start processing: %s" % the_dir)
        # support audio file as mp3 or m4a format
        mix_path_mp3 = os.path.join(dataset_dir, the_dir, "Mixture.mp3")
        mix_path_m4a = os.path.join(dataset_dir, the_dir, "Mixture.m4a")
        if os.path.exists(mix_path_mp3):
            mix_path = mix_path_mp3
        else:
            mix_path = mix_path_m4a
            
        y, sr = librosa.core.load(mix_path, sr=None)
        if sr != 44100:
            y = librosa.core.resample(y=y, orig_sr=sr, target_sr=44100)
        waveform = np.expand_dims(y, axis=1) # reorganize for separator's input

        prediction = separator.separate(waveform)

        # Mono Conversion (preprocess for later use)
        voc = librosa.core.to_mono(prediction["vocals"].T)
        # common practice to ensure values are valid for audio processing and playback
        voc = np.clip(voc, -1.0, 1.0)
        # save voice as new file
        soundfile.write(os.path.join(dataset_dir, the_dir, "Vocal.wav"), voc, 44100, subtype='PCM_16')

        # for accompaniment
        # acc = librosa.core.to_mono(prediction["accompaniment"].T)
        # acc = np.clip(acc, -1.0, 1.0)
        # soundfile.write(os.path.join(dataset_dir, the_dir, "Inst.wav"), acc, 44100, subtype='PCM_16')
