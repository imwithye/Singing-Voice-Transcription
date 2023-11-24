# Project Group 17 Singing-Voice-Transcription

Zhong Wenjun, A0255945R, e0962957@u.nus.edu
Luo Dan, A0222782E, e0563823@u.nus.edu
Bruce Ng, A0218204R, e0544240@u.nus.edu
Gong Yiwei, A0228568N, e0674552@u.nus.edu

Singing Voice Transcription for CS4347/CS5647

This repo contains our implementation of the Singing Voice Transcription pipeline.

Visualization of the result is available at `result_visualize.ipynb`.

## TL;NR

To see the result, run

```
python evaluate.py
```

To run the prediction, run

```
python predict.py --net effnet --model effnet_10
python predict.py --net resnet --model resnet_3
python predict.py --net resnet --model resnet_10
```

### Results

```
Evaluate Metrics of predict_effnet_10.json
         Precision Recall F1-score
COnPOff  0.523467 0.564335 0.541600
COnP     0.749959 0.807238 0.774895
COn      0.834252 0.894593 0.859785
gt note num: 30734.0 tr note num: 32753.0
song number: 98

Evaluate Metrics of predict_effnet_4.json
         Precision Recall F1-score
COnPOff  0.479675 0.524396 0.499333
COnP     0.715617 0.780670 0.743750
COn      0.814368 0.886349 0.844941
gt note num: 30734.0 tr note num: 33343.0
song number: 98

Evaluate Metrics of predict_resnet_10.json
         Precision Recall F1-score
COnPOff  0.395889 0.495712 0.438723
COnP     0.642451 0.807253 0.712909
COn      0.727555 0.912077 0.806169
gt note num: 30734.0 tr note num: 38407.0
song number: 98

Evaluate Metrics of predict_resnet_3.json
         Precision Recall F1-score
COnPOff  0.525057 0.564603 0.542553
COnP     0.758485 0.816162 0.783798
COn      0.838055 0.899006 0.864261
gt note num: 30734.0 tr note num: 32858.0
song number: 98

Evaluate Metrics of predict_resnet_5.json
         Precision Recall F1-score
COnPOff  0.471846 0.544128 0.504113
COnP     0.708213 0.818592 0.757293
COn      0.790457 0.910861 0.843566
gt note num: 30734.0 tr note num: 35243.0
song number: 98
```

###

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
