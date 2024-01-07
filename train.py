from ultralytics import YOLO
import cv2
import os
import numpy as np
from collections import defaultdict

# Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('e2e-pose.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
#
# Train the model
results = model.train(data='fisheye01-pose.yaml', epochs=100, imgsz=640, batch=128)