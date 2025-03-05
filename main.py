from tinyolo.model import YOLOv6s
from tinygrad.tensor import Tensor

mdl = YOLOv6s()
dummy_frame = Tensor.rand(1, 3, 640, 640)
predictions = mdl(dummy_frame)
print(predictions)