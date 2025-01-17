"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
import pdb


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class CheckLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, batch):
        batch = batch.permute(0, 2, 3, 1)
        output = self.linear(batch)
        return output.permute(0, 3, 1, 2)

class ResNet(nn.Module):

    def __init__(
        self, block, num_block, num_classes=100,
        # variant_name='1', pos_emb_dim=0, softmax_temp=1, variant_locs=[], stochastic_stride=False, stride=1
    ):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)) # ? -> 32
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1) # 32 -> 32
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2) # 32 -> 16
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2) # 16 -> 8
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2) # 8 -> 4
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.tensor_dimension = [
            [], [32, 64], [32, 64], [16, 128], [8, 256], [4, 512]
        ]

        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def get_network_structure(self):
        network_blocks = [
            self.conv1,
            self.conv2_x,
            self.conv3_x,
            self.conv4_x,
            self.conv5_x,
            self.avg_pool,
            torch.nn.Flatten(start_dim=1),
            self.fc
        ]

        spatial_shapes = [
            [32, 32, 64],
            [32, 32, 64],
            [16, 16, 128],
            [8, 8, 256],
            [4, 4, 512],
            [1, 1, 512],
        ]
        return network_blocks, spatial_shapes

        # self.network_layers = [
        #     self.conv1,
        #     self.conv2_x,
        #     self.conv3_x,
        #     self.conv4_x,
        #     self.conv5_x,
        #     self.avg_pool,
        #     torch.nn.Flatten(start_dim=1),
        #     self.fc
        # ]

        # self.backbone_layers = {
        #     0: self.conv1,
        #     1: self.conv2_x,
        #     2: self.conv3_x,
        #     3: self.conv4_x,
        #     4: self.conv5_x,
        #     5: self.avg_pool,
        #     6: torch.nn.Flatten(start_dim=1),
        #     7: self.fc
        # }
        # self.network_layers = []

        # self.variant_name = variant_name
        # self.variant_locs = variant_locs
        # if self.variant_name == 'fc':
        #     for injection_loc in self.variant_locs:
        #         _, n_channels = self.tensor_dimension[injection_loc]

        #         self.check_fc = CheckLinear(n_channels, n_channels)
        #         self.check_bn = nn.BatchNorm2d(n_channels)
        #         self.network_layers = self.network_layers[:variant_loc] + [self.check_fc, self.check_bn] + self.network_layers[variant_loc:]
        # elif self.variant_name != 'baseline':
        #     # pdb.set_trace()
        #     spatial_dim, n_channels = self.tensor_dimension[self.variant_loc]
        #     self.csam = csam.ConvolutionalSelfAttention(
        #         spatial_shape=[spatial_dim, spatial_dim, n_channels],
        #         filter_size=3,
        #         approach_args={
        #             'approach_name': variant_name,
        #             'pos_emb_dim': pos_emb_dim,
        #             'softmax_temp': softmax_temp,
        #             'stochastic_stride': stochastic_stride,
        #             'stride': stride
        #         },
        #         )
        #     self.csam_bn = nn.BatchNorm2d(n_channels)
        #     self.network_layers = self.network_layers[:variant_loc] + [self.csam, self.csam_bn] + self.network_layers[variant_loc:]

    # def generate_structure(self):
    #     backbone_layers = {
    #         0: self.conv1,
    #         1: self.conv2_x,
    #         2: self.conv3_x,
    #         3: self.conv4_x,
    #         4: self.conv5_x,
    #         5: self.avg_pool,
    #         6: torch.nn.Flatten(start_dim=1),
    #         7: self.fc
    #     }
    #     network_layers = []
    #     for injection_loc in self.variant_locs:
    #         spatial_dim, n_channels = self.tensor_dimension[injection_loc]


    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        # pdb.set_trace()
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        # if self.use_csam:
        #     # import pdb;pdb.set_trace()
        #     output = self.csam(output.permute(0, 2, 3, 1))
        #     output = output.permute(0, 3, 1, 2)
        #     output = self.csam_bn(output)
        output = self.conv4_x(output)

        output = self.conv5_x(output)

        # if self.use_csam:
        #     # import pdb;pdb.set_trace()
        #     output = self.csam(output.permute(0, 2, 3, 1))
        #     output = output.permute(0, 3, 1, 2)
        #     output = self.csam_bn(output)

        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        # pdb.set_trace()
        # output = x
        # for layer in self.network_layers:
        #     output = layer(output)

        return output

def resnet18(
    # variant_name='1',
    # pos_emb_dim=0,
    # softmax_temp=1,
    # variant_loc=-1,
    # stochastic_stride=False,
    # stride=1
    ):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2],
    # variant_name=variant_name, pos_emb_dim=pos_emb_dim,
    # softmax_temp=softmax_temp, variant_loc=variant_loc,
    # stochastic_stride=stochastic_stride, stride=stride
    )

def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])
