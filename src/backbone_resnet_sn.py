import torch
import torch.nn as nn
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
from torch.nn import Dropout2d
from torchvision.models.resnet import conv1x1
from torchvision.models.resnet import conv3x3

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Implement ResNet50-SNGP model in pytorch. This code was originally in tensorflow provided by:
'https://github.com/google/uncertainty-baselines/blob/master/uncertainty_baselines/models/resnet50_sngp.py'
"""
    
# Use batch normalization defaults from Pytorch.
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

class Bottleneck(nn.Module):
    """Bottleneck block.

    Bottleneck in torchvision places the stride for downsampling at 3x3
    convolution(self.conv2) while original implementation places the stride at the
    first 1x1 convolution(self.conv1) according to "Deep residual learning for
    image recognition"https://arxiv.org/abs/1512.03385.
    This variant is also known as ResNet V1.5 and improves accuracy according to
    https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    """

    expansion: int = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 dropout_rate: float = 0.1) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when
        # stride != 1.
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = Dropout2d(p=dropout_rate, inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        out = self.dropout(out)

        return out


class ResNetBackbone(nn.Module):
    def __init__(self, inplanes, block, layers, num_classes = 10):
        super().__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            # need exclude the first conv2d in layer0 from this following hook
            if isinstance(m, nn.Conv2d):
                m.register_full_backward_hook(self.spectral_norm_hook)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes, eps=BATCH_NORM_EPSILON, momentum=BATCH_NORM_DECAY),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x

    def spectral_norm_hook(self, module, grad_input, grad_output):
        # applied to linear layer weights after gradient descent updates
        with torch.no_grad():
            norm = torch.linalg.matrix_norm(module.weight, 2)
            if self.norm_multiplier < norm:
                module.weight = nn.Parameter(self.norm_multiplier * module.weight / norm)
