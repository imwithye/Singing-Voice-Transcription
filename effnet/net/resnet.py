import torch.nn as nn
import torch
from torchvision.models import resnet18


class ResNet18(nn.Module):
    def __init__(self, pitch_class=12, pitch_octave=4):
        super(ResNet18, self).__init__()
        self.model_name = "resnet"
        self.pitch_octave = pitch_octave
        self.pitch_class = pitch_class

        # Create model
        self.resnet = resnet18()
        # adjust the first conv layer to accept one-channel input, instead of 3 (colored image)
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Modify last linear layer
        self.fc_in_features = self.resnet.fc.in_features  # NOTE: might be wrong as input channel changed from 3 to 1
        self.resnet.fc = nn.Linear(
            in_features = self.fc_in_features, 
            out_features = 2 + pitch_class + pitch_octave + 2)\
        


    def forward(self, x):
        out = self.resnet(x)
        # print(out.shape)
        # [batch, output_size]

        onset_logits = out[:, 0]
        offset_logits = out[:, 1]

        pitch_out = out[:, 2:]

        pitch_octave_logits = pitch_out[:, 0 : self.pitch_octave + 1]
        pitch_class_logits = pitch_out[:, self.pitch_octave + 1 :]

        return onset_logits, offset_logits, pitch_octave_logits, pitch_class_logits


if __name__ == "__main__":
    from torchsummary import summary

    model = ResNet18().cuda()  # NOTE: the device type
    summary(model, input_size=(1, 11, 168))