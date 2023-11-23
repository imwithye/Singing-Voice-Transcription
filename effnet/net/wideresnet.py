import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualZeroPaddingBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        first_block=False,
        down_sample=False,
        up_sample=False,
    ):
        super(ResidualZeroPaddingBlock, self).__init__()
        self.first_block = first_block
        self.down_sample = down_sample
        self.up_sample = up_sample

        if self.up_sample:
            self.upsampling = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=2 if self.down_sample else 1,
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.skip_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=2 if self.down_sample else 1,
        )

        # Initialize the weights and biases
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0.1)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0.1)
        nn.init.xavier_uniform_(self.skip_conv.weight)

    def forward(self, x):
        if self.first_block:
            x = F.relu(x)
            if self.up_sample:
                x = self.upsampling(x)
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
            if x.shape != out.shape:
                x = self.skip_conv(x)
        else:
            out = F.relu(self.conv1(x))
            out = F.relu(self.conv2(out))

        return x + out


class WideResidualBlocks(nn.Module):
    def __init__(
        self, in_channels, out_channels, n, down_sample=False, up_sample=False
    ):
        super(WideResidualBlocks, self).__init__()
        self.blocks = nn.Sequential(
            *[
                ResidualZeroPaddingBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    first_block=(i == 0),
                    down_sample=down_sample,
                    up_sample=up_sample,
                )
                for i in range(n)
            ]
        )

    def forward(self, x):
        return self.blocks(x)


class WideResidualNet(nn.Module):
    def __init__(self, pitch_class=12, pitch_octave=4):
        super(WideResidualNet, self).__init__()
        self.model_name = "wideresnet"
        self.pitch_octave = pitch_octave
        self.pitch_class = pitch_class
        # target number of classes in the last linear layer
        self.n_features = 2 + pitch_class + pitch_octave + 2
        # Create model
        
        self.wideresnet = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # input channel = 1 
            WideResidualBlocks(16, 32, n=1, down_sample=False),
            WideResidualBlocks(32, 64, n=1, down_sample=True),
            WideResidualBlocks(64, 128, n=1, down_sample=True),
            nn.ReLU(),
            nn.Flatten(),  # output shape (2*16128)?
            # NOTE: 16128 might be wrong (it was for input-channel = 3)
            nn.Linear(16128, self.n_features, bias=False),   
            nn.Sigmoid(),  
        )

    def forward(self, x):
        out = self.wideresnet(x)
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

    model = WideResidualNet().cuda()  # NOTE: the device type
    summary(model, input_size=(1, 11, 168))
