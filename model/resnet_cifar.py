import math
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, affine=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)
        return out


class ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, width_factor=1, num_classes=10, first_stride=1, affine=True, large_weight=False):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16 * width_factor
        self.conv1 = nn.Conv2d(3, 16 * width_factor, kernel_size=first_stride * 2 + 1,
                               stride=first_stride, padding=first_stride, bias=False)
        self.bn1 = nn.BatchNorm2d(16 * width_factor, affine=affine)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16 * width_factor, layers[0], affine=affine)
        self.layer2 = self._make_layer(block, 32 * width_factor, layers[1], stride=2, affine=affine)
        self.layer3 = self._make_layer(block, 64 * width_factor, layers[2], stride=2, affine=affine)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * width_factor * block.expansion, num_classes, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) and affine:
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if large_weight:
            self.fc.weight.data *= 15

    def _make_layer(self, block, planes, blocks, stride=1, affine=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, affine=affine))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet_cifar(depth, width_factor=1, output_dim=10, first_stride=1, affine=True, large_weight=False):
    assert depth > 2 and depth % 6 == 2, 'Unsupported depth for ResNet (CIFAR)'
    block_num = depth // 6
    model = ResNet_Cifar(BasicBlock, [block_num, block_num, block_num],
                         width_factor, output_dim, first_stride, affine=affine, large_weight=large_weight)
    return model
