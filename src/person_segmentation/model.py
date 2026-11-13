from pathlib import Path

import torch
from torch import nn
from torchvision import models
from torchvision.models import segmentation


class DoubleConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class DownBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(input_channels, output_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class UpBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(input_channels, input_channels // 2, 2, stride=2)
        self.conv = DoubleConv(input_channels // 2 + skip_channels, output_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = self.up(inputs)
        height_difference = skip.shape[-2] - inputs.shape[-2]
        width_difference = skip.shape[-1] - inputs.shape[-1]
        inputs = nn.functional.pad(
            inputs,
            [
                width_difference // 2,
                width_difference - width_difference // 2,
                height_difference // 2,
                height_difference - height_difference // 2,
            ],
        )
        return self.conv(torch.cat((skip, inputs), dim=1))


class UNet(nn.Module):
    def __init__(self, input_channels: int = 3, output_channels: int = 1) -> None:
        super().__init__()
        self.input = DoubleConv(input_channels, 32)
        self.down1 = DownBlock(32, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)
        self.up1 = UpBlock(512, 256, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up3 = UpBlock(128, 64, 64)
        self.up4 = UpBlock(64, 32, 32)
        self.output = nn.Conv2d(32, output_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        first = self.input(inputs)
        second = self.down1(first)
        third = self.down2(second)
        fourth = self.down3(third)
        bottleneck = self.down4(fourth)
        output = self.up1(bottleneck, fourth)
        output = self.up2(output, third)
        output = self.up3(output, second)
        output = self.up4(output, first)
        return self.output(output)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PERSON_CLASS_INDEX = 15


class Normalize(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs - self.mean) / self.std


class DecoderBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv = DoubleConv(input_channels + skip_channels, output_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = nn.functional.interpolate(
            inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.conv(torch.cat((skip, inputs), dim=1))


class ResNetUNet(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet34(weights=weights)
        self.normalize = Normalize()
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.up4 = DecoderBlock(512, 256, 256)
        self.up3 = DecoderBlock(256, 128, 128)
        self.up2 = DecoderBlock(128, 64, 64)
        self.up1 = DecoderBlock(64, 64, 64)
        self.refine = DoubleConv(64, 32)
        self.output = nn.Conv2d(32, 1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        size = inputs.shape[-2:]
        x = self.normalize(inputs)
        s0 = self.stem(x)
        s1 = self.layer1(self.pool(s0))
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        out = self.up4(s4, s3)
        out = self.up3(out, s2)
        out = self.up2(out, s1)
        out = self.up1(out, s0)
        out = nn.functional.interpolate(out, size=size, mode="bilinear", align_corners=False)
        return self.output(self.refine(out))


class DeepLabPerson(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = (
            segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
            if pretrained
            else None
        )
        self.network = segmentation.deeplabv3_resnet50(
            weights=weights, weights_backbone=None, aux_loss=True
        )
        self.normalize = Normalize()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.network(self.normalize(inputs))["out"]
        person = logits[:, PERSON_CLASS_INDEX : PERSON_CLASS_INDEX + 1]
        keep = [i for i in range(logits.shape[1]) if i != PERSON_CLASS_INDEX]
        others = torch.logsumexp(logits[:, keep], dim=1, keepdim=True)
        return person - others


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    if name == "unet":
        return UNet()
    if name == "resnet34":
        return ResNetUNet(pretrained=pretrained)
    if name == "deeplab":
        return DeepLabPerson(pretrained=pretrained)
    raise ValueError(f"Unknown model: {name}")


def load_trained(path: str | Path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location=device)
    model = build_model(checkpoint.get("model_name", "unet"), pretrained=False)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint
