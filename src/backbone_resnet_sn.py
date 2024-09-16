import torch
import torch.nn as nn

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Implement ResNet50-SNGP model in pytorch. This code was originally in tensorflow provided by:
'https://github.com/google/uncertainty-baselines/blob/master/uncertainty_baselines/models/resnet50_sngp.py'
"""
    
# Use batch normalization defaults from Pytorch.
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)

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
        x = self.fc(x)

        return x

    def spectral_norm_hook(self, module, grad_input, grad_output):
        # applied to linear layer weights after gradient descent updates
        with torch.no_grad():
            norm = torch.linalg.matrix_norm(module.weight, 2)
            if self.norm_multiplier < norm:
                module.weight = nn.Parameter(self.norm_multiplier * module.weight / norm)
