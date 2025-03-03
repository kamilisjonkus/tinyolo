import math
from tinygrad import Tensor, nn
from typing import Tuple, List, Callable


class ConvBNReLU:
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=1, bias=False):
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=groups)
    self.bn = nn.BatchNorm2d(out_channels)

  def __call__(self, x: Tensor) -> Tensor:
    return self.bn(self.conv(x)).relu()
  
class ConvBNSiLU:
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=1, bias=False):
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=groups)
    self.bn = nn.BatchNorm2d(out_channels)

  def __call__(self, x: Tensor) -> Tensor:
    return self.bn(self.conv(x)).silu()

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
    

class EffiDeHead:
    '''Efficient Decoupled Head
    '''
    def __init__(self, num_classes, channels_list):  # detection layer
        chx = [6, 8, 10]
        head_layers = [
          ConvBNSiLU(channels_list[chx[0]], channels_list[chx[0]], 1, 1),
          ConvBNSiLU(channels_list[chx[0]], channels_list[chx[0]], 3, 1),
          ConvBNSiLU(channels_list[chx[0]], channels_list[chx[0]], 3, 1),
          nn.Conv2d(channels_list[chx[0]], num_classes, 1),
          nn.Conv2d(channels_list[chx[0]], 4, 1),
          ConvBNSiLU(channels_list[chx[1]], channels_list[chx[1]], 1, 1),
          ConvBNSiLU(channels_list[chx[1]], channels_list[chx[1]], 3, 1),
          ConvBNSiLU(channels_list[chx[1]], channels_list[chx[1]], 3, 1),
          nn.Conv2d(channels_list[chx[1]], num_classes, 1),
          nn.Conv2d(channels_list[chx[1]], 4, 1),
          ConvBNSiLU(channels_list[chx[2]], channels_list[chx[2]], 1, 1),
          ConvBNSiLU(channels_list[chx[2]], channels_list[chx[2]], 3, 1),
          ConvBNSiLU(channels_list[chx[2]], channels_list[chx[2]], 3, 1),
          nn.Conv2d(channels_list[chx[2]], num_classes, 1),
          nn.Conv2d(channels_list[chx[2]], 4, 1)
        ]
        self.nc = num_classes  # number of classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.grid = [Tensor.zeros(1)] * 3
        self.prior_prob = 1e-2
        stride = [8, 16, 32]
        self.stride = Tensor.tensor(stride)
        self.proj_conv = nn.Conv2d(1, 1, 1, bias=False)
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0

        # Init decouple head
        self.stems = []
        self.cls_convs = []
        self.reg_convs = []
        self.cls_preds = []
        self.reg_preds = []

        # Efficient decoupled head layers
        for i in range(3):
            idx = i*5
            self.stems.append(head_layers[idx])
            self.cls_convs.append(head_layers[idx+1])
            self.reg_convs.append(head_layers[idx+2])
            self.cls_preds.append(head_layers[idx+3])
            self.reg_preds.append(head_layers[idx+4])


    def __call__(self, x):
        if self.training:
            cls_score_list = []
            reg_distri_list = []

            for i in range(self.nl):
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)

                cls_output = Tensor.sigmoid(cls_output)
                cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))
                reg_distri_list.append(reg_output.flatten(2).permute((0, 2, 1)))

            cls_score_list = Tensor.cat(cls_score_list, axis=1)
            reg_distri_list = Tensor.cat(reg_distri_list, axis=1)

            return x, cls_score_list, reg_distri_list
        else:
            cls_score_list = []
            reg_dist_list = []

            for i in range(self.nl):
                b, _, h, w = x[i].shape
                l = h * w
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)
                cls_output = Tensor.sigmoid(cls_output)
                cls_score_list.append(cls_output.reshape([b, self.nc, l]))
                reg_dist_list.append(reg_output.reshape([b, 4, l]))

            cls_score_list = Tensor.cat(cls_score_list, axis=-1).permute(0, 2, 1)
            reg_dist_list = Tensor.cat(reg_dist_list, axis=-1).permute(0, 2, 1)
            anchor_points, stride_tensor = generate_anchors(
                x, self.stride, self.grid_cell_size, self.grid_cell_offset, device=x[0].device, is_eval=True, mode='af')
            pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format='xywh')
            pred_bboxes *= stride_tensor
            return Tensor.cat(
                [
                    pred_bboxes,
                    Tensor.ones((b, pred_bboxes.shape[1], 1), device=pred_bboxes.device, dtype=pred_bboxes.dtype),
                    cls_score_list
                ],
                axis=-1)

class YOLOv6s:
  def __init__(self, w, d, num_classes): #width_multiple, depth_multiple
    num_repeat_backbone = [1, 6, 12, 18, 6]
    num_repeat_neck = [12, 12, 12, 12]
    num_repeat = [(max(round(i * d), 1) if i > 1 else i) for i in (num_repeat_backbone + num_repeat_neck)]
    channels_list_backbone = [64, 128, 256, 512, 1024]
    channels_list_neck = [256, 128, 128, 256, 256, 512]
    channels_list = [math.ceil(i * w / 8) * 8 for i in (channels_list_backbone + channels_list_neck)]
    self.backbone = EfficientRep(in_channels=3, channels_list=channels_list, num_repeats=num_repeat)
    self.neck = RepBiFPANNeck(channels_list=channels_list, num_repeats=num_repeat)
    self.head = EffiDeHead(num_classes, channels_list)

  def __call__(self, x: Tensor) -> Tensor:
    x = self.backbone(x)
    x = self.neck(*x)
    return self.head(x)