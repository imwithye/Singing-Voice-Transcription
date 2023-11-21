import torch
import torch.nn as nn


def main():
    device = "cpu"
    if torch.cuda.is_available():
        device = "gpu"
    print("Device:", device)
