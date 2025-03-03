import math
from tinyolo.blocks import SimCSPSPPF, ConvBNReLU, BiFusion
from tinygrad import Tensor, nn
from typing import Tuple, List, Callable


class QARepVGGBlock:
    """
    RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://arxiv.org/abs/2212.01593
    """
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        self.bn = nn.BatchNorm2d(out_channels)
        self.rbr_dense = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.rbr_dense_bn = nn.BatchNorm2d(out_channels)
        self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=1, bias=False)
        self.identity = True if out_channels == in_channels and stride == 1 else False

    def __call__(self, x: Tensor) -> Tensor:
        return (self.bn(self.rbr_dense_bn(self.rbr_dense(x)) + self.rbr_1x1(x) + x if self.identity else 0)).relu()

def repeat_block(in_chnls, out_chnls, n) -> List[Callable[[Tensor], Tensor]]:
    return [QARepVGGBlock(in_chnls, out_chnls)] + [QARepVGGBlock(out_chnls, out_chnls) for _ in range(n - 1)] if n > 1 else []

class EfficientRep():
    def __init__(self, channels_list, num_repeats):
        self.stem = [QARepVGGBlock(3, channels_list[0], stride=2)]
        self.ERBlock_2 = [QARepVGGBlock(channels_list[0], channels_list[1], 2)] + repeat_block(channels_list[1], channels_list[1], num_repeats[1])
        self.ERBlock_3 = [QARepVGGBlock(channels_list[1], channels_list[2], 2)] + repeat_block(channels_list[2], channels_list[2], num_repeats[2])
        self.ERBlock_4 = [QARepVGGBlock(channels_list[2], channels_list[3], 2)] + repeat_block(channels_list[3], channels_list[3], num_repeats[3])
        self.ERBlock_5 = [QARepVGGBlock(channels_list[3], channels_list[4], 2)] + repeat_block(channels_list[4], channels_list[4], num_repeats[4]) + \
            + [SimCSPSPPF(channels_list[4], channels_list[4], kernel_size=5)]

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        x1 = x.sequential(self.stem)
        x2 = x1.sequential(self.ERBlock_2)
        x3 = x2.sequential(self.ERBlock_3)
        x4 = x3.sequential(self.ERBlock_4)
        x5 = x4.sequential(self.ERBlock_5)
        return (x2, x3, x4, x5)
    

class RepBiFPANNeck(nn.Module):
    """RepBiFPANNeck Module
    """
    # [64, 128, 256, 512, 1024]
    # [256, 128, 128, 256, 256, 512]

    def __init__(
        self,
        channels_list=None,
        num_repeats=None,
        block=QARepVGGBlock
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        self.reduce_layer0 = ConvBNReLU(
            in_channels=channels_list[4], # 1024
            out_channels=channels_list[5], # 256
            kernel_size=1,
            stride=1
        )

        self.Bifusion0 = BiFusion(
            in_channels=[channels_list[3], channels_list[2]], # 512, 256
            out_channels=channels_list[5], # 256
        )
        self.Rep_p4 = RepBlock(
            in_channels=channels_list[5], # 256
            out_channels=channels_list[5], # 256
            n=num_repeats[5],
            block=block
        )

        self.reduce_layer1 = ConvBNReLU(
            in_channels=channels_list[5], # 256
            out_channels=channels_list[6], # 128
            kernel_size=1,
            stride=1
        )

        self.Bifusion1 = BiFusion(
            in_channels=[channels_list[2], channels_list[1]], # 256, 128
            out_channels=channels_list[6], # 128
        )

        self.Rep_p3 = RepBlock(
            in_channels=channels_list[6], # 128
            out_channels=channels_list[6], # 128
            n=num_repeats[6],
            block=block
        )

        self.downsample2 = ConvBNReLU(
            in_channels=channels_list[6], # 128
            out_channels=channels_list[7], # 128
            kernel_size=3,
            stride=2
        )

        self.Rep_n3 = RepBlock(
            in_channels=channels_list[6] + channels_list[7], # 128 + 128
            out_channels=channels_list[8], # 256
            n=num_repeats[7],
            block=block
        )

        self.downsample1 = ConvBNReLU(
            in_channels=channels_list[8], # 256
            out_channels=channels_list[9], # 256
            kernel_size=3,
            stride=2
        )

        self.Rep_n4 = RepBlock(
            in_channels=channels_list[5] + channels_list[9], # 256 + 256
            out_channels=channels_list[10], # 512
            n=num_repeats[8],
            block=block
        )


    def forward(self, input):

        (x3, x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])
        pan_out2 = self.Rep_p3(f_concat_layer1)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        outputs = [pan_out2, pan_out1, pan_out0]

        return outputs

# https://github.com/meituan/YOLOv6/blob/main/configs/qarepvgg/yolov6s_qa.py
def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor

class YOLOv6s:
  def __init__(self, w, d, num_classes): #width_multiple, depth_multiple
    num_repeat_backbone = [1, 6, 12, 18, 6]
    num_repeat_neck = [12, 12, 12, 12]
    num_repeat = [(max(round(i * d), 1) if i > 1 else i) for i in (num_repeat_backbone + num_repeat_neck)]
    channels_list_backbone = [64, 128, 256, 512, 1024]
    channels_list_neck = [256, 128, 128, 256, 256, 512]
    channels_list = [make_divisible(i * w, 8) for i in (channels_list_backbone + channels_list_neck)]
    self.backbone = EfficientRep(in_channels=3, channels_list=channels_list, num_repeats=num_repeat)
    self.neck = RepBiFPANNeck(channels_list=channels_list, num_repeats=num_repeat)

  def __call__(self, x: Tensor) -> Tensor:
    x = self.backbone(x)
    x = self.neck(*x)
    return self.head(x)