import math
from tinygrad import Tensor, nn
from typing import Tuple, List, Callable


class ConvBNAct:
  def __init__(self, in_ch, out_ch, kernel_size, stride, padding=0, activation=None):
    self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
    self.bn = nn.BatchNorm2d(out_ch)
    self.activation = activation

  def __call__(self, x: Tensor) -> Tensor:
    x = self.bn(self.conv(x))
    if self.activation == None: return x
    elif self.activation == 'relu': return Tensor.relu(x)
    elif self.activation == 'silu': return Tensor.silu(x)
    else: raise ValueError(f"Unsupported activation: {self.activation}")
  
class RepVGGBlock:
  '''RepVGGBlock is a basic rep-style block, including training and deploy status
  This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
  Quantization-Aware version: https://arxiv.org/abs/2212.01593
  '''
  def __init__(self, in_ch, out_ch, stride=1):
    self.rbr_identity_bn = nn.BatchNorm2d(in_ch) if out_ch == in_ch and stride == 1 else None
    self.rbr_3x3 = ConvBNAct(in_ch, out_ch, 3, stride, 1)
    self.rbr_1x1 = ConvBNAct(in_ch, out_ch, 1, stride, 0)

  def __call__(self, x: Tensor) -> Tensor:
    if self.rbr_identity_bn is None: identity_out = 0
    else: identity_out = self.rbr_identity_bn(x)
    return Tensor.relu(self.rbr_3x3(x) + self.rbr_1x1(x) + identity_out)

def repeat_block(in_ch, out_ch, n) -> List[Callable[[Tensor], Tensor]]:
    return [RepVGGBlock(in_ch, out_ch)] + [RepVGGBlock(out_ch, out_ch) for _ in range(n - 1)] if n > 1 else []
  
class CSPSPPFModule:
  # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
  def __init__(self, in_ch, out_ch):
    inner_ch = int(out_ch * 0.5)
    self.cv1 = ConvBNAct(in_ch, inner_ch, 1, 1, activation='relu')
    self.cv2 = ConvBNAct(in_ch, inner_ch, 1, 1, activation='relu')
    self.cv3 = ConvBNAct(inner_ch, inner_ch, 3, 1, activation='relu')
    self.cv4 = ConvBNAct(inner_ch, inner_ch, 1, 1, activation='relu')
    self.cv5 = ConvBNAct(4 * inner_ch, inner_ch, 1, 1, activation='relu')
    self.cv6 = ConvBNAct(inner_ch, inner_ch, 3, 1, activation='relu')
    self.cv7 = ConvBNAct(2 * inner_ch, out_ch, 1, 1, activation='relu')

  def __call__(self, x: Tensor) -> Tensor:
    x1 = self.cv4(self.cv3(self.cv1(x)))
    y0 = self.cv2(x)
    y1 = x1.max_pool2d(5, 1, padding=2)
    y2 = y1.max_pool2d(5, 1, padding=2)
    y3 = y2.max_pool2d(5, 1, padding=2)
    y4 = self.cv6(self.cv5(Tensor.cat(x1, y1, y2, y3, dim=1)))
    return self.cv7(Tensor.cat(y0, y4, dim=1))

class BiFusion:
  def __init__(self, in_ch, out_ch):
    self.cv1 = ConvBNAct(in_ch[0], out_ch, 1, 1, activation='relu')
    self.cv2 = ConvBNAct(in_ch[1], out_ch, 1, 1, activation='relu')
    self.cv3 = ConvBNAct(out_ch * 3, out_ch, 1, 1, activation='relu')
    self.upsample = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=2, stride=2)
    self.downsample = ConvBNAct(out_ch, out_ch, 3, 2, activation='relu')

  def __call__(self, x: Tuple[Tensor]) -> Tensor:
    x0 = self.upsample(x[0])
    x1 = self.cv1(x[1])
    x2 = self.downsample(self.cv2(x[2]))
    return self.cv3(Tensor.cat(x0, x1, x2, dim=1))

class EfficientRep:
  def __init__(self, ch_list, num_repeats):
    self.stem = RepVGGBlock(3, ch_list[0], 2)
    self.ERBlock_2 = [RepVGGBlock(ch_list[0], ch_list[1], 2)] + repeat_block(ch_list[1], ch_list[1], num_repeats[1])
    self.ERBlock_3 = [RepVGGBlock(ch_list[1], ch_list[2], 2)] + repeat_block(ch_list[2], ch_list[2], num_repeats[2])
    self.ERBlock_4 = [RepVGGBlock(ch_list[2], ch_list[3], 2)] + repeat_block(ch_list[3], ch_list[3], num_repeats[3])
    self.ERBlock_5 = [RepVGGBlock(ch_list[3], ch_list[4], 2)] + repeat_block(ch_list[4], ch_list[4], num_repeats[4]) 
    self.csp = CSPSPPFModule(ch_list[4], ch_list[4])

  def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    x1 = self.stem(x)
    x2 = x1.sequential(self.ERBlock_2)
    x3 = x2.sequential(self.ERBlock_3)
    x4 = x3.sequential(self.ERBlock_4)
    x5 = self.csp(x4.sequential(self.ERBlock_5))
    return (x2, x3, x4, x5)

class RepBiFPANNeck:
  def __init__(self, ch_list, num_repeats):
    self.reduce_layer0 = ConvBNAct(ch_list[4], ch_list[5], 1, 1, activation='relu')
    self.Bifusion0 = BiFusion([ch_list[3], ch_list[2]], ch_list[5])
    self.Rep_p4 = repeat_block(ch_list[5], ch_list[5], num_repeats[5])
    self.reduce_layer1 = ConvBNAct(ch_list[5], ch_list[6], 1, 1, activation='relu')
    self.Bifusion1 = BiFusion([ch_list[2], ch_list[1]], ch_list[6])
    self.Rep_p3 = repeat_block(ch_list[6], ch_list[6], num_repeats[6])
    self.downsample2 = ConvBNAct(ch_list[6], ch_list[7], 3, 2, activation='relu')
    self.Rep_n3 = repeat_block(ch_list[6] + ch_list[7], ch_list[8], num_repeats[7])
    self.downsample1 = ConvBNAct(ch_list[8], ch_list[9], 3, 2, activation='relu')
    self.Rep_n4 = repeat_block(ch_list[5] + ch_list[9], ch_list[10], num_repeats[8])

  def __call__(self, x: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
    (x3, x2, x1, x0) = x
    fpn_out0 = self.reduce_layer0(x0)
    fpn_out1 = self.reduce_layer1(self.Bifusion0([fpn_out0, x1, x2]).sequential(self.Rep_p4))
    pan_out2 = self.Bifusion1([fpn_out1, x2, x3]).sequential(self.Rep_p3)
    pan_out1 = Tensor.cat(self.downsample2(pan_out2), fpn_out1, dim=1).sequential(self.Rep_n3)
    pan_out0 = Tensor.cat(self.downsample1(pan_out1), fpn_out0, dim=1).sequential(self.Rep_n4)
    return (pan_out2, pan_out1, pan_out0)
    
class EffiDeHead:
  '''Efficient Decoupled Head
  '''
  def __init__(self, num_classes, ch_list):
    self.stems = self.cls_convs = self.reg_convs = self.cls_preds = self.reg_preds = []
    chx = [6, 8, 10]
    for i in range(3):
      ch = ch_list[chx[i]]
      self.stems.append(ConvBNAct(ch, ch, 1, 1, activation='silu'))
      self.cls_convs.append(ConvBNAct(ch, ch, 3, 1, activation='silu'))
      self.reg_convs.append(ConvBNAct(ch, ch, 3, 1, activation='silu'))
      self.cls_preds.append(nn.Conv2d(ch, num_classes, 1))
      self.reg_preds.append(nn.Conv2d(ch, 4, 1))

  def __call__(self, x: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tuple[Tensor, Tensor, Tensor], Tensor, Tensor]:
    cls_score_list = reg_distri_list = []
    for i in range(3):
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
    return x, Tensor.cat(cls_score_list, axis=1), Tensor.cat(reg_distri_list, axis=1)

class YOLOv6:
  def __init__(self, w, d, num_classes): #width_multiple, depth_multiple
    num_repeat_backbone = [1, 6, 12, 18, 6]
    num_repeat_neck = [12, 12, 12, 12]
    num_repeat = [(max(round(i * d), 1) if i > 1 else i) for i in (num_repeat_backbone + num_repeat_neck)]
    ch_list_backbone = [64, 128, 256, 512, 1024]
    ch_list_neck = [256, 128, 128, 256, 256, 512]
    ch_list = [math.ceil(i * w / 8) * 8 for i in (ch_list_backbone + ch_list_neck)]
    self.backbone = EfficientRep(ch_list=ch_list, num_repeats=num_repeat)
    self.neck = RepBiFPANNeck(ch_list=ch_list, num_repeats=num_repeat)
    self.head = EffiDeHead(num_classes, ch_list)

  def __call__(self, x: Tensor) -> Tensor:
    x = self.backbone(x)
    x = self.neck(*x)
    return self.head(x)
  
YOLOv6n = lambda num_classes=80: YOLOv6(0.25, 0.33, num_classes=num_classes)
YOLOv6s = lambda num_classes=80: YOLOv6(0.50, 0.33, num_classes=num_classes)