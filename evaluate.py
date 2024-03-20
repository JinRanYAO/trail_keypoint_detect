from ultralytics import YOLO

model = YOLO('runs/pose/train-raw-01/weights/best.pt')

evaluate_results = model.val(data='fisheye01-pose.yaml')