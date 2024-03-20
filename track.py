from ultralytics import YOLO
import cv2
import os
import numpy as np
from tracker import KalmanFilter

model = YOLO('/home/yons/ultralytics/runs/pose/train8/weights/best.pt')
# model = YOLO('/home/yons/ultralytics/runs/pose/train8/weights/best-sim.onnx')
seq_path = "/mnt/pool1/yaotong/fisheye_dataset/scene_01/images/2023-12-13-15-34-57"

prev_image = None
prev_image_gray = None
prev_means = None
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

trackers = []
trackers_init = [False, False, False]
means = []
covariances = []

for img_path in sorted(os.listdir(seq_path)):
    curr_image = cv2.imread(os.path.join(seq_path, img_path))
    curr_image_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
    results = model(curr_image)
    boxes = results[0].boxes.xywh.cpu().squeeze(0)
    w, h = boxes[2], boxes[3]
    curr_keypoints = results[0].keypoints.data.cpu().numpy().reshape(-1, 3)

    for i in range(curr_keypoints.shape[0]):
        if curr_keypoints[i, 2] >= 0.5 and trackers_init[i] == False:
            kf = KalmanFilter()
            mean, cov = kf.initiate(curr_keypoints[i, :2], w, h)
            trackers.append(kf)
            means.append(mean)
            covariances.append(cov)
            trackers_init[i] = True
            continue
        elif curr_keypoints[i, 2] >= 0.5:
            means[i], covariances[i] = trackers[i].predict(means[i], covariances[i], w, h)
            means[i], covariances[i] = trackers[i].update(means[i], covariances[i], curr_keypoints[i, :2], w, h, mea_type='detect')

    if prev_image is not None:

        curr_keypoints_LK, status, _ = cv2.calcOpticalFlowPyrLK(prev_image_gray, curr_image_gray, np.array(prev_means)[:, :2].astype(np.float32), None, **lk_params)

        curr_keypoints_LK = curr_keypoints_LK[status.flatten() == 1]

        for i in range(curr_keypoints_LK.shape[0]):
            cv2.circle(curr_image, (int(curr_keypoints_LK[i, 0]), int(curr_keypoints_LK[i, 1])), 2, (255, 0, 0), -1)
            means[i], covariances[i] = trackers[i].update(means[i], covariances[i], curr_keypoints_LK[i, :], w, h, mea_type='LK')

    curr_keypoints = curr_keypoints[curr_keypoints[:, 2] >= 0.2][:, :2].astype(np.float32)

    for i in range(curr_keypoints.shape[0]):
        cv2.circle(curr_image, (int(curr_keypoints[i, 0]), int(curr_keypoints[i, 1])), 2, (0, 0, 0), -1)

    for i in range(len(means)):
        cv2.circle(curr_image, (int(means[i][0]), int(means[i][1])), 2, (255, 255, 255), -1)

    cv2.imshow("YOLOv8 Tracking", curr_image)
    cv2.waitKey(800)

    prev_image = curr_image
    prev_image_gray = curr_image_gray
    prev_means = means

cv2.destroyAllWindows()
