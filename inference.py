from ultralytics import YOLO
import onnx

# 载入pytorch模型
model = YOLO('runs/pose/train8/weights/best.pt')
# model.export(format='onnx', opset=11)
model.export(format="engine", device=0)

# onnx_model = onnx.load('runs/pose/train8/weights/best.onnx')
# onnx.checker.check_model(onnx_model)
#
# print(onnx.helper.printable_graph(onnx_model.graph))