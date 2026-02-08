"""
MobileFaceNet Architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import (
    BatchNorm1d,
    BatchNorm2d,
    Conv2d,
    Linear,
    Module,
    PReLU,
    Sequential,
)


class Flatten(Module):
    """Flatten layer."""

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor.view(input_tensor.size(0), -1)


class ConvBlock(Module):
    """Standard Convolution Block with BatchNorm and PReLU."""

    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel: tuple[int, int] = (1, 1),
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = Conv2d(
            in_c,
            out_channels=out_c,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class LinearBlock(Module):
    """Convolution Block acting as Linear projection (1x1 Conv) + BatchNorm."""

    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel: tuple[int, int] = (1, 1),
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = Conv2d(
            in_c,
            out_channels=out_c,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return x


class DepthWise(Module):
    """Depth-wise Separable Convolution Block."""

    def __init__(
        self,
        in_c: int,
        out_c: int,
        residual: bool = False,
        kernel: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (2, 2),
        padding: tuple[int, int] = (1, 1),
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = ConvBlock(
            in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1)
        )
        self.conv_dw = ConvBlock(
            groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride
        )
        self.project = LinearBlock(
            groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1)
        )
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual:
            short_cut = x
        out = self.conv(x)
        out = self.conv_dw(out)
        out = self.project(out)
        if self.residual:
            output = short_cut + out
        else:
            output = out
        return output


class Residual(Module):
    """Sequence of Residual DepthWise blocks."""

    def __init__(
        self,
        c: int,
        num_block: int,
        groups: int,
        kernel: tuple[int, int] = (3, 3),
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        modules = []
        for _ in range(num_block):
            modules.append(
                DepthWise(
                    c,
                    c,
                    residual=True,
                    kernel=kernel,
                    padding=padding,
                    stride=stride,
                    groups=groups,
                )
            )
        self.model = Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class GNAP(Module):
    """Global Norm-Aware Pooling."""

    def __init__(self, embedding_size: int) -> None:
        super().__init__()
        assert embedding_size == 512
        self.bn1 = BatchNorm2d(512, affine=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn2 = BatchNorm1d(512, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(x)
        x_norm = torch.norm(x, 2, 1, True)
        x_norm_mean = torch.mean(x_norm)
        weight = x_norm_mean / x_norm
        x = x * weight
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        feature = self.bn2(x)
        return feature


class GDC(Module):
    """Global Depthwise Convolution."""

    def __init__(self, embedding_size: int) -> None:
        super().__init__()
        self.conv_6_dw = LinearBlock(
            512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0)
        )
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_6_dw(x)
        x = self.conv_6_flatten(x)
        x = self.linear(x)
        x = self.bn(x)
        return x


class MobileFaceNet(Module):
    """MobileFaceNet model architecture."""

    def __init__(
        self,
        input_size: list[int] | tuple[int, int],
        embedding_size: int = 512,
        output_name: str = "GDC",
    ) -> None:
        super().__init__()
        assert output_name in ["GNAP", "GDC"]
        assert input_size[0] in [112]

        self.conv1 = ConvBlock(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = ConvBlock(
            64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64
        )
        self.conv_23 = DepthWise(
            64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128
        )
        self.conv_3 = Residual(
            64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv_34 = DepthWise(
            64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256
        )
        self.conv_4 = Residual(
            128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv_45 = DepthWise(
            128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512
        )
        self.conv_5 = Residual(
            128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv_6_sep = ConvBlock(
            128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0)
        )

        if output_name == "GNAP":
            self.output_layer = GNAP(512)
        else:
            self.output_layer = GDC(embedding_size)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)

        conv_features = self.conv_6_sep(out)
        out = self.output_layer(conv_features)

        # Original code returns (output, feature_map)
        return out, conv_features
