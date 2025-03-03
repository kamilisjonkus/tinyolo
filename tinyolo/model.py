import math
from tinygrad import Tensor, nn
from typing import Tuple, List, Callable


class ConvBNReLU:
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=1, bias=False):
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=groups)
    self.bn = nn.BatchNorm2d(out_channels)

  def __call__(self, x: Tensor) -> Tensor:
    return self.bn(self.conv(x)).relu()

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
  
class CSPSPPFModule:
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, in_channels, out_channels):
        c_ = int(out_channels * 0.5)  # hidden channels
        self.cv1 = ConvBNReLU(in_channels, c_, 1, 1)
        self.cv2 = ConvBNReLU(in_channels, c_, 1, 1)
        self.cv3 = ConvBNReLU(c_, c_, 3, 1)
        self.cv4 = ConvBNReLU(c_, c_, 1, 1)
        self.cv5 = ConvBNReLU(4 * c_, c_, 1, 1)
        self.cv6 = ConvBNReLU(c_, c_, 3, 1)
        self.cv7 = ConvBNReLU(2 * c_, out_channels, 1, 1)

    def __call__(self, x: Tensor) -> Tensor:
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y0 = self.cv2(x)
        y1 = x1.max_pool2d(5, 1, padding=2)
        y2 = y1.max_pool2d(5, 1, padding=2)
        y3 = self.cv6(self.cv5(Tensor.cat([x1, y1, y2, y2.max_pool2d(5, 1, padding=2)], 1)))
        return self.cv7(Tensor.cat((y0, y3), dim=1))

class BiFusion:
    def __init__(self, in_channels, out_channels):
        self.cv1 = ConvBNReLU(in_channels[0], out_channels, 1, 1)
        self.cv2 = ConvBNReLU(in_channels[1], out_channels, 1, 1)
        self.cv3 = ConvBNReLU(out_channels * 3, out_channels, 1, 1)
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.downsample = ConvBNReLU(out_channels, out_channels, 3, 2)

    def __call__(self, x: Tuple[Tensor]) -> Tensor:
        x0 = self.upsample(x[0])
        x1 = self.cv1(x[1])
        x2 = self.downsample(self.cv2(x[2]))
        return self.cv3(Tensor.cat((x0, x1, x2), dim=1))

class EfficientRep:
    def __init__(self, channels_list, num_repeats):
        self.stem = [QARepVGGBlock(3, channels_list[0], stride=2)]
        self.ERBlock_2 = [QARepVGGBlock(channels_list[0], channels_list[1], 2)] + repeat_block(channels_list[1], channels_list[1], num_repeats[1])
        self.ERBlock_3 = [QARepVGGBlock(channels_list[1], channels_list[2], 2)] + repeat_block(channels_list[2], channels_list[2], num_repeats[2])
        self.ERBlock_4 = [QARepVGGBlock(channels_list[2], channels_list[3], 2)] + repeat_block(channels_list[3], channels_list[3], num_repeats[3])
        self.ERBlock_5 = [QARepVGGBlock(channels_list[3], channels_list[4], 2)] + repeat_block(channels_list[4], channels_list[4], num_repeats[4]) + \
            + [CSPSPPFModule(channels_list[4], channels_list[4])]

    def __call__(self, x: Tensor) -> Tuple[Tensor]:
        x1 = x.sequential(self.stem)
        x2 = x1.sequential(self.ERBlock_2)
        x3 = x2.sequential(self.ERBlock_3)
        x4 = x3.sequential(self.ERBlock_4)
        x5 = x4.sequential(self.ERBlock_5)
        return (x2, x3, x4, x5)

class RepBiFPANNeck:
    def __init__(self, channels_list, num_repeats):
        self.reduce_layer0 = ConvBNReLU(channels_list[4], channels_list[5], 1, 1)
        self.Bifusion0 = BiFusion([channels_list[3], channels_list[2]], channels_list[5])
        self.Rep_p4 = repeat_block(channels_list[5], channels_list[5], num_repeats[5])
        self.reduce_layer1 = ConvBNReLU(channels_list[5], channels_list[6], 1, 1)
        self.Bifusion1 = BiFusion([channels_list[2], channels_list[1]], channels_list[6])
        self.Rep_p3 = repeat_block(channels_list[6], channels_list[6], num_repeats[6])
        self.downsample2 = ConvBNReLU(channels_list[6], channels_list[7], 3, 2)
        self.Rep_n3 = repeat_block(channels_list[6] + channels_list[7], channels_list[8], num_repeats[7])
        self.downsample1 = ConvBNReLU(channels_list[8], channels_list[9], 3, 2)
        self.Rep_n4 = repeat_block(channels_list[5] + channels_list[9], channels_list[10], num_repeats[8])

    def __call__(self, x: Tuple[Tensor]) -> Tuple[Tensor]:

        (x3, x2, x1, x0) = x

        fpn_out0 = self.reduce_layer0(x0)
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])
        f_out0 = f_concat_layer0.sequential(self.Rep_p4)

        fpn_out1 = self.reduce_layer1(f_out0)
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])
        pan_out2 = f_concat_layer1.sequential(self.Rep_p3)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = Tensor.cat([down_feat1, fpn_out1], 1)
        pan_out1 = p_concat_layer1.sequential(self.Rep_n3)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = Tensor.cat([down_feat0, fpn_out0], 1)
        pan_out0 = p_concat_layer2.sequential(self.Rep_n4)

        return (pan_out2, pan_out1, pan_out0)

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