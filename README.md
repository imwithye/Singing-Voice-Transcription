# Project Group 17 Singing-Voice-Transcription
Singing Voice Transcription for CS4347/CS5647

This repo contains our implementation of the Singing Voice Transcription pipeline.

## TL;NR

To see the result, run

```
python evaluate.py
```

## Install Dependencies

Do note that Python version must be 3.10

```
apt install ffmpeg
pip install -r requirements.txt
```

## Prepare the database

By running following script, it will download the dataset from YouTube.

```
python prepare_dataset.py
```

You may run this scripts multi times due to network issues.

## Train the model

```
python train.py --net <network> [--train-file-limit 100] [--valid-file-limit 100]
```

We implement 3 diffrenet networks

1. effnet
2. resnet
3. wideresnet

As training on the full dataset is slow and requires 4090 with at least 20GB VRAM. You may choose to limit the training file size with `--train-file-limit` parameter.

The trained model will be saved to the `models` directory. 

We have saved our trained models for evaluation.

## Prediction

```
python predict.py --net <network> --model <model_name>
```

The trained model will be saved with epoch index. You may check the `models` folder. To predict with certain checkpoint, for example

```
python predict.py --net effnet --model effnet_10
python predict.py --net resnet --model resnet_10
python predict.py --net wideresnet --model wideresnet_10
```

## Evaluation

```
python evaluate.py
```
